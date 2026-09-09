"""Explicit sequential stage execution, isolated scale artifacts and resumable gates."""

import copy
import gc
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import torch

from src.evaluation.artifacts import write_json
from src.evaluation.identity import read_json
from src.evaluation.service import evaluate_models
from src.experiments.plan import digest
from src.experiments.preflight import audit_data, preflight, storage_checks
from src.experiments.registry import check_seal, log, save, seal
from src.experiments.subsets import prepare_subset
from src.training.reproducibility import file_hash
from src.training.trainer import run_training
from src.trigram.trainer import train


def scale_settings(settings, directory):
    values = copy.deepcopy(settings.values)
    values["trigram"]["storage_backend"] = "sqlite"
    paths = {
        **settings.paths,
        "MODEL_DIR": directory / "models",
        "CHECKPOINT_DIR": directory / "checkpoints",
        "CANONICAL_TOKENIZER_DIR": settings.paths["MODEL_DIR"] / "tokenizer",
    }
    return replace(settings, values=values, paths=paths)


def training_resume(directory, settings):
    candidates = []
    for path in (directory / "training").glob("*/summary.json"):
        result = read_json(path)
        if not result.get("dry_run"):
            candidates.append((path.stat().st_mtime_ns, result))
    if not candidates:
        return None, None
    result = max(candidates, key=lambda pair: pair[0])[1]
    checkpoint = settings.paths["CHECKPOINT_DIR"] / result["run_id"] / "latest.pt"
    return result, checkpoint if checkpoint.exists() else None


def run_scale(
    settings, plan, audit, directory, scale, experiment_directory, persist, *, dry_only=False
):
    local = scale_settings(settings, directory)
    subset, subset_manifest, provenance = prepare_subset(
        settings,
        directory / "subset",
        scale,
        plan["subset_seed"],
        audit["split_fingerprints"]["train"],
    )
    scale["subset_fingerprint"] = provenance["subset_fingerprint"]
    scale["subset_manifest_path"] = str(subset_manifest)
    persist()
    write_json(
        experiment_directory / "subset_manifest.json",
        {
            "scales": [
                {
                    "scale": s["scale"],
                    "manifest": s.get("subset_manifest_path"),
                    "subset_fingerprint": s.get("subset_fingerprint"),
                }
                for s in read_json(experiment_directory / "status.json")["scales"]
            ]
        },
    )
    token_dir = settings.paths["MODEL_DIR"] / "tokenizer"
    validation = settings.paths["DATA_DIR"] / "splits/validation.jsonl"
    stages = scale.setdefault("stages", {})
    for name in ("trigram", "dry_run", "transformer", "evaluation"):
        if dry_only and name != "dry_run":
            continue
        if name == "trigram" and not plan["run_trigram"]:
            continue
        if name in ("dry_run", "transformer") and not plan["run_transformer"]:
            continue
        if name == "evaluation" and not plan["evaluate_test"]:
            continue
        previous = stages.get(name, {})
        if previous.get("status") == "COMPLETED":
            check_seal(previous)
            continue
        # Check completed predecessors too; changed artifacts never silently resume.
        for record in stages.values():
            if record.get("status") == "COMPLETED":
                check_seal(record)
        storage_checks([directory], plan["minimum_free_disk_gb"])
        for split, expected in audit["split_fingerprints"].items():
            if file_hash(settings.paths["DATA_DIR"] / f"splits/{split}.jsonl") != expected:
                raise ValueError("Canonical split changed during experiment")
        scale["status"] = {
            "trigram": "RUNNING_TRIGRAM",
            "dry_run": "RUNNING_TRANSFORMER",
            "transformer": "RUNNING_TRANSFORMER",
            "evaluation": "RUNNING_EVALUATION",
        }[name]
        stages[name] = {"status": "RUNNING"}
        persist()
        log(experiment_directory, scale["scale"], name, "RUNNING")
        try:
            if name == "trigram":
                result = train(
                    subset,
                    token_dir,
                    local,
                    dataset_fingerprint=audit["dataset_fingerprint"],
                    subset_manifest=subset_manifest,
                    overwrite=bool(previous),
                )
                files = [
                    local.paths["MODEL_DIR"] / "trigram" / filename
                    for filename in (
                        "trigram_counts.sqlite",
                        "trigram_manifest.json",
                        "training_statistics.json",
                    )
                ]
            elif name in ("dry_run", "transformer"):
                old, checkpoint = (
                    training_resume(directory, local) if name == "transformer" else (None, None)
                )
                if old and old["status"] in ("COMPLETED", "EARLY_STOPPED"):
                    result = old
                else:
                    result = run_training(
                        local,
                        subset,
                        validation,
                        token_dir,
                        output=directory / "training",
                        dry_run=name == "dry_run",
                        resume=checkpoint,
                        dataset_manifest=audit["dataset_manifest"],
                        subset_manifest=subset_manifest,
                    )
                if result["status"] not in ("COMPLETED", "EARLY_STOPPED"):
                    stages[name] = {"status": result["status"], "result": result}
                    scale["status"] = result["status"]
                    persist()
                    return False
                files = (
                    [Path(result["run_directory"]) / "summary.json"]
                    if name == "dry_run"
                    else [
                        local.paths["MODEL_DIR"] / "transformer" / filename
                        for filename in ("best_model.pt", "model_manifest.json")
                    ]
                )
            else:
                result = evaluate_models(
                    local,
                    settings.paths["DATA_DIR"] / "splits/test.jsonl",
                    token_dir,
                    output=directory / "evaluation",
                    full_test=True,
                    dataset_manifest=audit["dataset_manifest"],
                    subset_manifest=subset_manifest,
                )
                if result["status"] != "COMPLETED":
                    stages[name] = {"status": result["status"], "result": result}
                    scale["status"] = result["status"]
                    persist()
                    return False
                scale["comparison_path"] = result["comparison_path"]
                files = [
                    Path(result["run_directory"]) / filename
                    for filename in (
                        "summary.json",
                        "comparison.json",
                        "comparison.csv",
                        "generation_results.json",
                        "trigram_metrics.json",
                        "transformer_metrics.json",
                        "environment.json",
                        "resolved_config.json",
                    )
                ]
            stages[name] = {
                "status": "COMPLETED",
                "result": result,
                "artifacts": seal(files),
            }
            persist()
            log(experiment_directory, scale["scale"], name, "COMPLETED")
        except KeyboardInterrupt:
            scale["status"] = "INTERRUPTED"
            stages[name] = {"status": "INTERRUPTED"}
            persist()
            return False
        except Exception as exc:
            detail = getattr(exc, "summary", {})
            status = detail.get("status", "FAILED_OTHER")
            scale["status"] = (
                status if status in ("FAILED_OOM", "FAILED_NONFINITE") else "FAILED_OTHER"
            )
            stages[name] = {"status": scale["status"], "message": str(exc), "result": detail}
            persist()
            log(experiment_directory, scale["scale"], name, scale["status"])
            return False
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not dry_only:
        scale["status"] = "COMPLETED"
        scale["primary_final_result"] = scale["scale"] == "full" and plan["evaluate_test"]
    else:
        scale["status"] = "PENDING"
    persist()
    return True


def run_experiment(
    settings,
    plan,
    *,
    mode="plan",
    selected=None,
    output=None,
    resume=None,
    confirm_large_run=False,
    dataset_manifest=None,
):
    if mode not in ("plan", "preflight", "dry-run-only", "execute"):
        raise ValueError("Unknown experiment mode")
    parent = Path(output or settings.paths["EXPERIMENT_DIR"] / "final").resolve()
    if settings.paths["EXPERIMENT_DIR"].resolve() not in parent.parents:
        raise ValueError("Experiment output must be beneath EXPERIMENT_DIR")
    audit = audit_data(settings, plan, dataset_manifest)
    resolved = {
        "orchestrator_policy_version": 1,
        "controlled_trigram_backend": "sqlite",
        "experiment": plan,
        "settings": settings.values,
        "paths": {k: str(v.resolve()) for k, v in settings.paths.items()},
        "data": audit,
        "tokenizer_fingerprint": file_hash(
            settings.paths["MODEL_DIR"] / "tokenizer/tokenizer.json"
        ),
    }
    fingerprint = digest(resolved)
    chosen = [str(s) for s in selected] if selected else [s["scale"] for s in audit["scales"]]
    if len(chosen) != len(set(chosen)) or not set(chosen).issubset(
        {s["scale"] for s in audit["scales"]}
    ):
        raise ValueError("Selected scales must be unique and present in the plan")
    if mode == "plan":
        return {
            "status": "PLANNED",
            "audit": audit,
            "selected": chosen,
            "plan_fingerprint": fingerprint,
            "training_launched": False,
        }
    if resume and not re.fullmatch(r"[A-Za-z0-9_-]+", resume):
        raise ValueError("Resume requires a safe experiment ID")
    experiment_id = (
        resume or f"{plan['name']}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    )
    directory = parent / experiment_id
    if resume:
        status = read_json(directory / "status.json")
        if status["plan_fingerprint"] != fingerprint:
            raise ValueError(
                "Resume plan/configuration/data fingerprint mismatch; start a new experiment"
            )
    else:
        directory.mkdir(parents=True, exist_ok=False)
        status = {
            "experiment_id": experiment_id,
            "plan_fingerprint": fingerprint,
            "audit": audit,
            "status": "PLANNED",
            "scales": copy.deepcopy(audit["scales"]),
            "run_directory": str(directory),
        }
        write_json(directory / "experiment_plan.json", resolved)

    def persist():
        save(directory, status)

    # Exclusive-writer lock: stale locks after hard termination require explicit user removal.
    lock = directory / ".writer.lock"
    with lock.open("x"):
        pass
    try:
        if resume:
            status = read_json(directory / "status.json")
        persist()
        try:
            report = preflight(settings, plan, audit, directory)
            write_json(directory / "preflight.json", report)
            write_json(directory / "environment.json", report["environment"])
            status["status"] = "READY"
            persist()
        except Exception as exc:
            status.update(status="PREFLIGHT_FAILED", message=str(exc))
            write_json(
                directory / "preflight.json", {"status": "PREFLIGHT_FAILED", "message": str(exc)}
            )
            persist()
            return status
        if mode == "preflight":
            return status
        todo = [
            s for s in status["scales"] if s["scale"] in chosen and s["status"] != "NOT_APPLICABLE"
        ]
        if (
            mode == "execute"
            and not confirm_large_run
            and any(
                s["scale"] == "full"
                or s["actual_train_documents"] >= plan["large_run_threshold_documents"]
                for s in todo
                if s["status"] != "COMPLETED"
            )
        ):
            status["message"] = (
                "No training launched: --confirm-large-run is required for FULL "
                "or threshold-sized scales"
            )
            persist()
            return status
        if mode == "dry-run-only":
            todo = todo[:1]
        status["status"] = "RUNNING"
        status.pop("failed_scale", None)
        persist()
        for scale in todo:
            if scale["status"] == "COMPLETED":
                for record in scale["stages"].values():
                    check_seal(record)
                continue
            if not run_scale(
                settings,
                plan,
                audit,
                directory / "scales" / scale["scale"],
                scale,
                directory,
                persist,
                dry_only=mode == "dry-run-only",
            ):
                status["failed_scale"] = scale["scale"]
                status["status"] = (
                    "INTERRUPTED"
                    if scale["status"] == "INTERRUPTED"
                    else "PARTIAL"
                    if any(s["status"] == "COMPLETED" for s in status["scales"])
                    else "FAILED"
                )
                persist()
                return status
        status["status"] = (
            "READY"
            if mode == "dry-run-only"
            else "COMPLETED"
            if all(s["status"] in ("COMPLETED", "NOT_APPLICABLE") for s in status["scales"])
            else "PARTIAL"
        )
        persist()
        return status
    except KeyboardInterrupt:
        status["status"] = "INTERRUPTED"
        persist()
        return status
    except Exception as exc:
        status.update(status="FAILED", message=str(exc))
        persist()
        return status
    finally:
        lock.unlink()
