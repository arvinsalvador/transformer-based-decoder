"""Read-only evaluation orchestration; no training APIs or optimizer are imported."""

import copy
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import psutil
import torch

from src.config.evaluation import validate_evaluation
from src.config.settings import PROJECT_ROOT
from src.evaluation.adapters import score_transformer, score_trigram
from src.evaluation.artifacts import table, write_csv, write_json
from src.evaluation.generation import generate_comparison, load_prompts
from src.evaluation.identity import FingerprintError, read_json, training_metadata, verify
from src.evaluation.metrics import compare
from src.training.monitoring import environment
from src.training.precision import select_precision
from src.training.reproducibility import file_hash
from src.transformer.config import TransformerConfig
from src.transformer.inspection import inspect
from src.transformer.model import DecoderOnlyTransformer
from src.trigram.serialization import load_model
from src.utils.device import detect_device


class EvaluationFailure(RuntimeError):
    def __init__(self, summary):
        self.summary = summary
        super().__init__(summary["message"])


def evaluate_models(
    settings,
    test,
    tokenizer_path,
    *,
    trigram_path=None,
    transformer_path=None,
    model="both",
    output=None,
    limit_documents=None,
    full_test=False,
    prompts_path=None,
    dataset_manifest=None,
    subset_manifest=None,
):
    cfg = copy.deepcopy(settings.values["evaluation"])
    if limit_documents is not None:
        cfg["document_limit"] = limit_documents
    if full_test:
        if limit_documents is not None:
            raise ValueError("Choose full test or document limit, not both")
        cfg["document_limit"] = None
    root = settings.paths["EXPERIMENT_DIR"].resolve()
    parent = Path(output or root / "evaluation").resolve()
    if parent != root and root not in parent.parents:
        raise ValueError("Evaluation output must be beneath EXPERIMENT_DIR")
    evaluation_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:10]
    directory = parent / evaluation_id
    directory.mkdir(parents=True)
    summary = {
        "evaluation_id": evaluation_id,
        "status": "RUNNING",
        "started_at": datetime.now(UTC).isoformat(),
        "run_directory": str(directory),
        "document_limit": cfg["document_limit"],
        "scope": "FULL TEST SET" if cfg["document_limit"] is None else "DEVELOPMENT SUBSET",
    }
    write_json(directory / "summary.json", summary)
    start, stage, models, results = perf_counter(), "FAILED_CONFIGURATION", {}, {}
    try:
        validate_evaluation(cfg)
        if model not in ("both", "trigram", "transformer"):
            raise ValueError("model must be both, trigram, or transformer")
        names = ["trigram", "transformer"] if model == "both" else [model]
        prompts_path = Path(prompts_path or PROJECT_ROOT / "config/evaluation_prompts.yaml")
        prompts = load_prompts(prompts_path)
        info = detect_device(settings.values["device"] if "transformer" in names else "cpu")
        device = torch.device(info.selected_device)
        precision = select_precision(device, cfg["precision"], True)
        env = {**environment(info, precision), "system_ram_bytes": psutil.virtual_memory().total}
        write_json(directory / "environment.json", env)
        directories = {
            "trigram": Path(trigram_path or settings.paths["MODEL_DIR"] / "trigram"),
            "transformer": Path(transformer_path or settings.paths["MODEL_DIR"] / "transformer"),
        }
        manifest_names = {"trigram": "trigram_manifest.json", "transformer": "model_manifest.json"}
        filenames = {"trigram": "trigram_counts.sqlite", "transformer": "best_model.pt"}
        stage = "FAILED_MODEL_LOAD"
        manifests = {name: read_json(directories[name] / manifest_names[name]) for name in names}
        stage = "FAILED_FINGERPRINT"
        tokenizer, special, identity = verify(
            settings, test, tokenizer_path, manifests, dataset_manifest, subset_manifest
        )
        stage = "FAILED_MODEL_LOAD"
        watched = [
            Path(test),
            Path(tokenizer_path) / "tokenizer.json",
            Path(tokenizer_path) / "tokenizer_manifest.json",
            prompts_path,
        ]
        if dataset_manifest:
            watched.append(Path(dataset_manifest))
        if subset_manifest:
            watched.extend([Path(subset_manifest), Path(read_json(subset_manifest)["subset_path"])])
        watched += [
            directories[name] / file
            for name in names
            for file in (manifest_names[name], filenames[name])
        ]
        original_hashes = {path: file_hash(path) for path in watched}
        summary.update(**identity, models_evaluated=names)
        write_json(
            directory / "resolved_config.json",
            {
                "evaluation": cfg,
                "model": model,
                "fingerprints": identity,
                "prompts": prompts,
                "prompts_fingerprint": original_hashes[prompts_path],
                "random_seed": settings.values["random_seed"],
            },
        )
        metadata = {}
        stage = "FAILED_MODEL_LOAD"
        for name in names:
            manifest, path = manifests[name], directories[name] / filenames[name]
            metadata[name] = {
                **identity,
                **training_metadata(manifest),
                "model_fingerprint": original_hashes[path],
                "model_size_bytes": path.stat().st_size,
                "model_manifest_path": str(directories[name] / manifest_names[name]),
                "model_artifact_path": str(path),
                "training_run_id": manifest.get("run_id"),
                "best_optimizer_step": manifest.get("best_optimizer_step"),
                "best_epoch": manifest.get("best_epoch"),
                "training_validation_batch_limit": manifest.get("validation_batch_limit"),
                "environment": env,
            }
            if name == "trigram":
                loaded = load_model(path)
                models[name] = loaded
                if (loaded.vocabulary_size, loaded.bos_id, loaded.eos_id, loaded.add_k) != (
                    tokenizer.get_vocab_size(),
                    special["bos_token_id"],
                    special["eos_token_id"],
                    manifest["add_k"],
                ):
                    raise FingerprintError("Trigram database identity mismatch")
                metadata[name].update(
                    backend="sqlite",
                    training_backend=manifest.get("storage_backend"),
                    smoothing=manifest.get("smoothing"),
                    add_k=loaded.add_k,
                    parameter_count=None,
                    **loaded.statistics(),
                )
            else:
                loaded = DecoderOnlyTransformer(TransformerConfig(**manifest["architecture"]))
                architecture = inspect(loaded)
                if (
                    architecture["architecture_fingerprint"] != manifest["architecture_fingerprint"]
                    or loaded.config.vocab_size != tokenizer.get_vocab_size()
                    or loaded.config.pad_token_id != special["pad_token_id"]
                ):
                    raise FingerprintError("Transformer architecture fingerprint mismatch")
                loaded.load_state_dict(
                    torch.load(path, weights_only=True, map_location="cpu"), strict=True
                )
                loaded.requires_grad_(False).eval().to(device)
                models[name] = loaded
                metadata[name].update(
                    parameter_count=architecture["total_parameters"],
                    architecture_fingerprint=architecture["architecture_fingerprint"],
                    architecture=manifest["architecture"],
                )
        stage = "FAILED_EVALUATION"
        for name in names:
            # A separate streaming pass has identical input bytes, tokenizer, limit and order.
            if file_hash(test) != identity["test_fingerprint"]:
                raise FingerprintError("Test split changed during evaluation")
            common = (models[name], tokenizer, test, cfg["document_limit"], special, metadata[name])
            result = (
                score_trigram(*common)
                if name == "trigram"
                else score_transformer(*common, cfg, device, precision)
            )
            results[name] = {
                "evaluation_id": evaluation_id,
                "scope": summary["scope"],
                "document_limit": cfg["document_limit"],
                **result,
            }
            write_json(
                directory / f"{name}_metrics.json",
                {"status": "PARTIAL_UNTIL_SUMMARY_COMPLETED", **results[name]},
            )
        generated = generate_comparison(
            models,
            tokenizer,
            prompts,
            special,
            cfg["generation"],
            device,
            precision,
            settings.values["random_seed"],
        )
        for path, digest in original_hashes.items():
            if file_hash(path) != digest:
                raise FingerprintError(f"Read-only input changed during evaluation: {path}")
        relative, statements = (
            compare(results["trigram"], results["transformer"])
            if model == "both"
            else ({}, ["Single-model evaluation; no primary comparison performed."])
        )
        comparison = {
            "evaluation_id": evaluation_id,
            "status": "COMPLETED",
            "scope": summary["scope"],
            "fingerprints": identity,
            "results": results,
            "table": table(results),
            "relative": relative,
            "interpretation": statements,
            "generation_settings": cfg["generation"],
            "prompts_fingerprint": original_hashes[prompts_path],
            "environment": env,
        }
        write_json(
            directory / "generation_results.json",
            {"evaluation_id": evaluation_id, "results": generated},
        )
        write_csv(directory / "comparison.csv", comparison["table"], evaluation_id, identity)
        write_json(directory / "comparison.json", comparison)
        summary.update(
            status="COMPLETED",
            test_documents=next(iter(results.values()))["documents"],
            prediction_events=next(iter(results.values()))["prediction_events"],
            comparison_path=str(directory / "comparison.json"),
        )
    except KeyboardInterrupt:
        summary.update(
            status="INTERRUPTED",
            message="Evaluation interrupted; no complete comparison is available.",
        )
    except Exception as exc:
        message = str(exc)
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            message += (
                " Lower evaluation.transformer_batch_size; no examples were silently skipped."
            )
        summary.update(
            status="FAILED_FINGERPRINT" if isinstance(exc, FingerprintError) else stage,
            message=message,
            partial_models=list(results),
        )
    finally:
        for name, loaded in models.items():
            if name == "trigram":
                loaded.close()
        summary.update(
            completed_at=datetime.now(UTC).isoformat(), duration_seconds=perf_counter() - start
        )
        write_json(directory / "summary.json", summary)
    if summary["status"].startswith("FAILED"):
        raise EvaluationFailure(summary)
    return summary
