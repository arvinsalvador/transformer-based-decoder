"""Fail-closed FULL selection and cross-artifact validation. Never evaluates or trains."""

import csv
import math
import re
import sqlite3
from pathlib import Path

import torch

from src.evaluation.identity import read_json, verify
from src.evaluation.metrics import compare
from src.experiments.plan import digest
from src.experiments.preflight import audit_data
from src.experiments.registry import check_seal
from src.training.reproducibility import file_hash
from src.transformer.config import TransformerConfig
from src.transformer.inspection import inspect
from src.transformer.model import DecoderOnlyTransformer
from src.trigram.serialization import load_model


def inside(path, root):
    path = Path(path).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("Experiment artifact path escapes its experiment directory")
    return path


def validate_full(settings, directory):
    status = read_json(directory / "status.json")
    resolved = read_json(directory / "experiment_plan.json")
    if digest(resolved) != status["plan_fingerprint"]:
        raise ValueError("Experiment plan fingerprint mismatch")
    full = next((s for s in status["scales"] if s["scale"] == "full"), None)
    if not full or full["status"] != "COMPLETED":
        raise ValueError("FULL has not completed")
    if not full.get("primary_final_result"):
        raise ValueError("FULL is not designated primary")
    audit = audit_data(settings, resolved["experiment"], status["audit"]["dataset_manifest"])
    if (
        audit["split_fingerprints"] != status["audit"]["split_fingerprints"]
        or audit["counts"] != status["audit"]["counts"]
    ):
        raise ValueError("Canonical data changed since experiment")
    if full["actual_train_documents"] != audit["counts"]["train"]:
        raise ValueError("FULL did not use all canonical training documents")
    for stage in ("trigram", "dry_run", "transformer", "evaluation"):
        record = full["stages"][stage]
        if record["status"] != "COMPLETED":
            raise ValueError(f"FULL {stage} stage is incomplete")
        for path in record["artifacts"]:
            inside(path, directory)
        check_seal(record)
    comparison_path = inside(full["comparison_path"], directory)
    comparison = read_json(comparison_path)
    if comparison["status"] != "COMPLETED" or comparison["scope"] != "FULL TEST SET":
        raise ValueError("Primary FULL evaluation was incomplete or limited")
    if set(comparison["results"]) != {"trigram", "transformer"}:
        raise ValueError("FULL comparison must contain both models")
    model_root = directory / "scales/full/models"
    manifests = {
        "trigram": read_json(model_root / "trigram/trigram_manifest.json"),
        "transformer": read_json(model_root / "transformer/model_manifest.json"),
    }
    tokenizer, _, identity = verify(
        settings,
        settings.paths["DATA_DIR"] / "splits/test.jsonl",
        settings.paths["MODEL_DIR"] / "tokenizer",
        manifests,
        audit["dataset_manifest"],
        inside(full["subset_manifest_path"], directory),
    )
    for key in (
        "test_fingerprint",
        "tokenizer_fingerprint",
        "dataset_fingerprint",
        "subset_fingerprint",
    ):
        if identity[key] != comparison["fingerprints"][key]:
            raise ValueError(f"Comparison {key} mismatch")
    for name, result in comparison["results"].items():
        for key in identity:
            if key.endswith("fingerprint") and result.get(key) != identity[key]:
                raise ValueError(f"Individual {name} {key} mismatch")
        if (
            result["document_limit"] is not None
            or result["scope"] != "FULL TEST SET"
            or result["documents"] != audit["counts"]["test"]
        ):
            raise ValueError("Primary result is a development-limited test evaluation")
        for key in (
            "average_nll",
            "perplexity",
            "training_duration_seconds",
            "model_size_bytes",
            "prediction_events",
            "evaluation_seconds",
            "total_nll",
        ):
            value = result.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid final metric {name}.{key}")
        if (
            result["perplexity"] < 1
            or result["model_size_bytes"] <= 0
            or result["prediction_events"] < result["documents"]
        ):
            raise ValueError("Implausible final metrics")
        for key in ("peak_training_ram_mb", "peak_training_vram_mb", "tokens_per_second"):
            if result.get(key) is not None and (
                type(result[key]) not in (int, float)
                or not math.isfinite(result[key])
                or result[key] < 0
            ):
                raise ValueError("Invalid resource measurement")
        filename = "trigram_counts.sqlite" if name == "trigram" else "best_model.pt"
        if result["model_fingerprint"] != file_hash(model_root / name / filename):
            raise ValueError("Model fingerprint mismatch")
        if result["model_size_bytes"] != (model_root / name / filename).stat().st_size:
            raise ValueError("Model artifact size mismatch")
        if result["training_duration_seconds"] != manifests[name]["training_duration_seconds"]:
            raise ValueError("Recorded training time mismatch")
        if not math.isclose(
            result["average_nll"], result["total_nll"] / result["prediction_events"]
        ) or not math.isclose(math.log(result["perplexity"]), result["average_nll"]):
            raise ValueError("NLL/perplexity arithmetic mismatch")
        individual = read_json(comparison_path.parent / f"{name}_metrics.json")
        for key in (
            "average_nll",
            "perplexity",
            "prediction_events",
            "test_fingerprint",
            "tokenizer_fingerprint",
        ):
            if individual[key] != result[key]:
                raise ValueError("Individual metrics disagree with comparison")
    tf = manifests["transformer"]
    with torch.device("meta"):
        model = DecoderOnlyTransformer(TransformerConfig(**tf["architecture"]))
    architecture = inspect(model)
    if (
        architecture["architecture_fingerprint"] != tf["architecture_fingerprint"]
        or architecture["total_parameters"]
        != comparison["results"]["transformer"]["parameter_count"]
        or model.config.vocab_size != tokenizer.get_vocab_size()
    ):
        raise ValueError("Transformer architecture/parameter mismatch")
    model.load_state_dict(
        torch.load(
            model_root / "transformer/best_model.pt", map_location="meta", weights_only=True
        ),
        strict=True,
    )
    with load_model(model_root / "trigram/trigram_counts.sqlite") as trigram:
        if (
            trigram.connection.execute("PRAGMA quick_check").fetchone()[0] != "ok"
            or trigram.vocabulary_size != tokenizer.get_vocab_size()
        ):
            raise ValueError("Trigram database integrity failure")
    training_summary = full["stages"]["transformer"]["result"]
    if training_summary["status"] not in ("COMPLETED", "EARLY_STOPPED") or training_summary.get(
        "dry_run"
    ):
        raise ValueError("Final Transformer has no actual completed training evidence")
    for key in ("gpu_peak_mb", "process_ram_mb"):
        value = training_summary.get("resources", {}).get(key)
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
        ):
            raise ValueError("Invalid saved training resource measurement")
    relative, interpretation = compare(
        comparison["results"]["trigram"], comparison["results"]["transformer"]
    )
    if relative != comparison["relative"]:
        raise ValueError("Saved relative metrics disagree with model results")
    summary = read_json(directory / "summary.json")
    matrix = read_json(directory / "experiment_matrix.json")
    if (
        not summary["primary_final_result"]
        or summary["full_result"] != comparison
        or summary["rows"] != matrix["rows"]
        or matrix["plan_fingerprint"] != status["plan_fingerprint"]
        or matrix["experiment_id"] != status["experiment_id"]
        or matrix["resolved_plan"] != resolved
    ):
        raise ValueError("Primary summary/matrix mismatch")
    matrix_full = next(row for row in matrix["rows"] if row["scale"] == "full")
    for name, result in comparison["results"].items():
        for key in ("average_nll", "perplexity", "training_duration_seconds", "model_size_bytes"):
            if matrix_full[f"{name}_{key}"] != result[key]:
                raise ValueError("FULL matrix metrics mismatch")
    with (directory / "experiment_matrix.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != len(matrix["rows"]) or any(
        any(csv_row.get(key) != ("" if value is None else str(value)) for key, value in row.items())
        for csv_row, row in zip(csv_rows, matrix["rows"], strict=True)
    ):
        raise ValueError("Experiment matrix CSV disagrees with JSON")
    warnings = []
    if resolved["settings"]["environment"] != "gpu" or training_summary.get("device") != "cuda":
        warnings.append(
            "Final artifacts record local/CPU training; scientifically usable, "
            "but not evidence of a GPU run."
        )
    if resolved["settings"]["training"].get("max_steps") is not None:
        warnings.append(
            "Training used an optimizer-step limit; disclose this bounded experiment policy."
        )
    curves = []
    for scale in status["scales"]:
        row = {
            "scale": scale["scale"],
            "status": scale["status"],
            "train_documents": scale["actual_train_documents"],
        }
        if scale["status"] == "COMPLETED":
            try:
                for path in scale["stages"]["evaluation"]["artifacts"]:
                    inside(path, directory)
                check_seal(scale["stages"]["evaluation"])
                other = read_json(inside(scale["comparison_path"], directory))
                if other["fingerprints"]["test_fingerprint"] != identity["test_fingerprint"]:
                    raise ValueError("Supporting scale test fingerprint mismatch")
                row["results"] = other["results"]
                row["training_resources"] = (
                    scale["stages"]["transformer"].get("result", {}).get("resources", {})
                )
            except (ValueError, OSError, KeyError) as exc:
                row.update(status="FAIL_INTEGRITY", message=str(exc))
                warnings.append(
                    f"Supporting scale {scale['scale']} failed integrity; "
                    "retained as failed, not hidden."
                )
        curves.append(row)
    generation = read_json(comparison_path.parent / "generation_results.json")
    if generation["evaluation_id"] != comparison["evaluation_id"]:
        raise ValueError("Generation evaluation ID mismatch")
    prompts = {
        name: [item["prompt"] for item in generation["results"] if item["model_type"] == name]
        for name in ("trigram", "transformer")
    }
    if not prompts["trigram"] or prompts["trigram"] != prompts["transformer"]:
        raise ValueError("Saved generation prompts are not paired")
    return {
        "status": "PASS",
        "experiment_id": status["experiment_id"],
        "comparison": comparison,
        "counts": audit["counts"],
        "total_corpus_documents": audit["total_corpus_documents"],
        "tokenizer_vocabulary": tokenizer.get_vocab_size(),
        "relative": relative,
        "interpretation": interpretation,
        "warnings": warnings,
        "learning_curve": curves,
        "generation": generation["results"],
        "resolved_plan": resolved,
        "fingerprints": identity,
        "training_resources": training_summary.get("resources", {}),
    }


def load_results(settings, experiment=None, latest_complete=False):
    root = settings.paths["EXPERIMENT_DIR"] / "final"
    if experiment:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment):
            return {"status": "FAIL", "message": "Unsafe experiment ID"}
        try:
            return validate_full(settings, root / experiment)
        except (
            ValueError,
            OSError,
            KeyError,
            RuntimeError,
            StopIteration,
            TypeError,
            sqlite3.Error,
        ) as exc:
            return {"status": "FAIL", "message": str(exc), "experiment_id": experiment}
    candidates = []
    for path in root.glob("*/status.json"):
        try:
            status = read_json(path)
            if any(s["scale"] == "full" and s["status"] == "COMPLETED" for s in status["scales"]):
                candidates.append((status.get("updated_at", ""), path.parent.name))
        except (ValueError, OSError, KeyError):
            continue
    if not candidates:
        return {"status": "NOT_RUN", "message": "FINAL EXPERIMENT NOT YET EXECUTED"}
    if len(candidates) > 1 and not latest_complete:
        return {
            "status": "WARNING",
            "message": "Multiple FULL experiments; select --experiment or --latest-complete",
        }
    # Latest is based on recorded update time, never on metric quality. Invalid latest fails closed.
    return load_results(settings, max(candidates)[1])
