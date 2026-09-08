"""Single-device training with exact target normalization and resumable step boundaries."""

import copy
import math
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import torch

from src.config.settings import ConfigurationError, validate_config
from src.training.checkpoint import atomic_json, export_best, load_checkpoint, save_checkpoint
from src.training.data import SequenceDataset, close_loader, loader
from src.training.monitoring import environment, history_row, perplexity, resources
from src.training.optimization import (
    NonFiniteError,
    WarmupCosine,
    finish_accumulation,
    loss_sum,
    optimizer_for,
)
from src.training.precision import autocast, make_scaler, select_precision
from src.training.reproducibility import restore_rng, rng_state, seed_all, verify_inputs
from src.training.validation import validate
from src.transformer.factory import build
from src.transformer.inspection import inspect
from src.utils.device import DeviceUnavailableError, detect_device


class TrainingFailure(RuntimeError):
    def __init__(self, summary):
        self.summary = summary
        super().__init__(summary["message"])


def _safe_child(root, path):
    path, root = Path(path).resolve(), Path(root).resolve()
    if root not in path.parents:
        raise ValueError(f"Output must be beneath {root}")
    return path


def run_training(
    settings,
    train_path,
    validation_path,
    tokenizer_path,
    *,
    output=None,
    resume=None,
    max_steps=None,
    epochs=None,
    dry_run=False,
    run_name=None,
    overwrite_export=False,
    dataset_manifest=None,
    validation_batches=None,
):
    """Run bounded development or configured training. No test path is accepted.

    Resume replays/skips deterministic batches to the last completed optimizer
    boundary, then restores training RNGs. Dataset order settings must be unchanged.
    """
    values = copy.deepcopy(settings.values)
    if max_steps is not None:
        values["training"]["max_steps"] = max_steps
    if epochs is not None:
        values["training"]["epochs"] = epochs
    settings = replace(settings, values=values)
    cfg = values["training"]
    saved = load_checkpoint(resume) if resume else None
    if dry_run and resume:
        raise ValueError("Dry run does not resume persistent training")
    run_id = (
        saved["run_id"]
        if saved
        else datetime.now(UTC).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:10]
    )
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("Invalid checkpoint run ID")
    base = Path(output or settings.paths["EXPERIMENT_DIR"] / "training")
    directory = _safe_child(settings.paths["EXPERIMENT_DIR"], base / run_id)
    checkpoint_dir = _safe_child(
        settings.paths["CHECKPOINT_DIR"], settings.paths["CHECKPOINT_DIR"] / run_id
    )
    export_dir = settings.paths["MODEL_DIR"] / "transformer"
    if resume and Path(resume).resolve().parent != checkpoint_dir:
        raise ValueError("Resume checkpoint must belong to configured CHECKPOINT_DIR/run_id")
    directory.mkdir(parents=True, exist_ok=bool(saved))
    summary = {
        "run_id": run_id,
        "run_name": run_name,
        "status": "RUNNING",
        "dry_run": dry_run,
        "started_at": datetime.now(UTC).isoformat(),
        "run_directory": str(directory),
    }
    atomic_json(directory / "summary.json", summary)
    train_loader = validation_loader = model = optimizer = None
    device = torch.device("cpu")
    started = None
    in_step = False
    state = {
        "step": 0,
        "epoch": 0,
        "batch_in_epoch": 0,
        "microbatches": 0,
        "valid_tokens": 0,
        "sequences": 0,
        "nll_sum": 0.0,
        "best_nll": float("inf"),
        "early_reference": float("inf"),
        "bad_epochs": 0,
        "best_epoch": None,
        "best_step": None,
        "elapsed_seconds": 0.0,
    }
    safe_rng = None
    checkpoint = None
    fingerprints = {}
    try:
        validate_config(values)
        info = detect_device(values["device"])
        device = torch.device(info.selected_device)
        precision = select_precision(device, cfg["precision"], cfg["mixed_precision"])
        summary.update(device=info.selected_device, precision=precision)
        special, fingerprints, warnings = verify_inputs(
            train_path, validation_path, tokenizer_path, settings, dataset_manifest
        )
        if (
            not dry_run
            and not overwrite_export
            and any(
                (export_dir / name).exists() for name in ("best_model.pt", "model_manifest.json")
            )
        ):
            raise ValueError("Export exists; use --overwrite-export or a separate MODEL_DIR")
        seed_all(values["random_seed"], cfg["deterministic"])
        model, _ = build(settings, tokenizer_path)
        architecture = inspect(model)
        fingerprints["architecture_fingerprint"] = architecture["architecture_fingerprint"]
        if architecture["total_parameters"] > 100_000_000:
            warnings.append(
                "Architecture exceeds 100M parameters; substantial resources may be required."
            )
        summary.update(
            fingerprints=fingerprints,
            warnings=warnings,
            parameters=architecture["total_parameters"],
        )
        if saved:
            if saved["fingerprints"] != fingerprints:
                raise ValueError(
                    "Resume dataset/tokenizer/architecture/vocabulary fingerprint mismatch"
                )
            old = saved["config"]
            # Allow longer stopping limits; optimizer schedule keeps its saved horizon.
            current_order = {k: v for k, v in cfg.items() if k not in ("epochs", "max_steps")}
            old_order = {
                k: v for k, v in old["training"].items() if k not in ("epochs", "max_steps")
            }
            if current_order != old_order or values["random_seed"] != old["random_seed"]:
                raise ValueError("Resume training configuration/order differs from checkpoint")
            if saved["precision"] != precision:
                raise ValueError("Resume precision differs from checkpoint")
            model.load_state_dict(saved["model"])
            state = saved["state"]
            if cfg["max_steps"] is not None and cfg["max_steps"] <= state["step"]:
                raise ValueError("Resume max_steps must exceed saved optimizer step")
        model.to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_dataset = SequenceDataset(
            train_path,
            tokenizer_path,
            special,
            values["model"]["context_length"],
            cfg["sequence_stride"],
            shuffle=cfg["shuffle_buffer_size"],
            seed=values["random_seed"],
        )
        validation_dataset = SequenceDataset(
            validation_path,
            tokenizer_path,
            special,
            values["model"]["context_length"],
            values["model"]["context_length"],
            seed=values["random_seed"],
        )
        train_loader = loader(train_dataset, cfg, special["pad_token_id"])
        validation_loader = loader(
            validation_dataset, cfg, special["pad_token_id"], validation=True
        )
        optimizer = optimizer_for(model, cfg)
        scaler = make_scaler(precision)
        if dry_run:
            total_steps = 1
        elif saved:
            total_steps = saved["scheduler"]["total_steps"]
        elif cfg["max_steps"] is not None:
            total_steps = cfg["max_steps"]
        else:
            # Count only sequences, streaming; worker-specific final batches matter.
            workers = max(1, cfg["num_workers"])
            counts = [0] * workers
            for sample in train_dataset.records(0, 1):
                counts[sample["document_index"] % workers] += 1
            batches = sum(math.ceil(count / cfg["batch_size"]) for count in counts)
            total_steps = math.ceil(batches / cfg["gradient_accumulation_steps"]) * cfg["epochs"]
        scheduler = WarmupCosine(
            optimizer, total_steps, cfg["warmup_steps"], cfg["min_learning_rate_ratio"]
        )
        if saved:
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            if scaler is not None:
                scaler.load_state_dict(saved["scaler"])
            restore_rng(saved["rng"])
        safe_rng = rng_state()
        atomic_json(
            directory / "resolved_config.json",
            {
                "settings": values,
                "architecture": model.config.to_dict(),
                "fingerprints": fingerprints,
                "device": str(device),
                "precision": precision,
                "effective_sequence_batch": cfg["batch_size"] * cfg["gradient_accumulation_steps"],
                "scheduler_total_steps": total_steps,
                "validation_batch_limit": validation_batches,
            },
        )
        atomic_json(directory / "environment.json", environment(info, precision))

        def payload():
            return {
                "format_version": 1,
                "run_id": run_id,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "state": copy.deepcopy(state),
                "rng": safe_rng,
                "config": values,
                "architecture": model.config.to_dict(),
                "fingerprints": fingerprints,
                "precision": precision,
            }

        def checkpoint(**options):
            save_checkpoint(
                checkpoint_dir, payload(), keep=cfg["checkpoint"]["keep_last_n"], **options
            )

        initial_elapsed = state["elapsed_seconds"]
        started = perf_counter()

        def log(validation=None, gradient=None):
            elapsed = initial_elapsed + perf_counter() - started
            row = {
                "epoch": state["epoch"],
                "optimizer_step": state["step"],
                "train_nll": state["nll_sum"] / max(1, state["valid_tokens"]),
                "validation_nll": validation["nll"] if validation else None,
                "validation_perplexity": validation["perplexity"] if validation else None,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "valid_tokens": state["valid_tokens"],
                "sequences": state["sequences"],
                "microbatches": state["microbatches"],
                "tokens_per_second": state["valid_tokens"] / max(elapsed, 1e-12),
                "sequences_per_second": state["sequences"] / max(elapsed, 1e-12),
                "elapsed_seconds": elapsed,
                "gradient_norm": gradient,
                **resources(device),
            }
            history_row(directory / "history.csv", row)
            with (directory / "training.log").open("a", encoding="utf-8") as stream:
                stream.write(
                    f"step={state['step']} epoch={state['epoch']} train_nll={row['train_nll']}\n"
                )

        if dry_run:
            batch = next(iter(train_loader), None)
            if batch is None:
                raise ValueError("Training split is empty")
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(device, precision):
                loss, tokens = loss_sum(
                    model(batch["input_ids"], batch["attention_mask"]), batch["labels"]
                )
            (scaler.scale(loss / tokens) if scaler is not None else loss / tokens).backward()
            if scaler is not None:
                scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            if not torch.isfinite(norm):
                raise NonFiniteError("Nonfinite dry-run gradient")
            summary.update(
                status="COMPLETED",
                loss=float(loss.detach()) / tokens,
                forward="PASS",
                backward="PASS",
                optimizer_steps=0,
                batch_shape=list(batch["input_ids"].shape),
                gradient_norm=float(norm),
            )
        else:
            stop = False
            for epoch in range(state["epoch"], cfg["epochs"]):
                train_dataset.set_epoch(epoch)
                offset = state["batch_in_epoch"] if epoch == state["epoch"] else 0
                state["epoch"] = epoch
                optimizer.zero_grad(set_to_none=True)
                iterator = iter(train_loader)
                # Replay costs I/O/tokenization but does not mutate model or optimizer.
                for _ in range(offset):
                    if next(iterator, None) is None:
                        raise ValueError("Checkpoint batch cursor exceeds training stream")
                restore_rng(safe_rng)
                group_tokens = group_sequences = group_micro = 0
                group_loss = 0.0
                exhausted = True

                def step_group(group_tokens, group_sequences, group_micro, group_loss):
                    nonlocal in_step, safe_rng
                    in_step = True
                    norm = finish_accumulation(
                        model, optimizer, scaler, group_tokens, cfg["max_grad_norm"]
                    )
                    scheduler.step()
                    state["step"] += 1
                    state["batch_in_epoch"] += group_micro
                    state["microbatches"] += group_micro
                    state["valid_tokens"] += group_tokens
                    state["sequences"] += group_sequences
                    state["nll_sum"] += group_loss
                    state["elapsed_seconds"] = initial_elapsed + perf_counter() - started
                    safe_rng = rng_state()
                    in_step = False
                    if state["step"] % cfg["logging_steps"] == 0:
                        log(gradient=norm)
                    if state["step"] % cfg["checkpoint"]["save_every_steps"] == 0:
                        checkpoint(archive=True)

                model.train()
                for batch in iterator:
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    with autocast(device, precision):
                        loss, tokens = loss_sum(
                            model(batch["input_ids"], batch["attention_mask"]), batch["labels"]
                        )
                    (scaler.scale(loss) if scaler is not None else loss).backward()
                    group_tokens += tokens
                    group_sequences += batch["input_ids"].shape[0]
                    group_micro += 1
                    group_loss += float(loss.detach())
                    if group_micro == cfg["gradient_accumulation_steps"]:
                        step_group(group_tokens, group_sequences, group_micro, group_loss)
                        group_tokens = group_sequences = group_micro = 0
                        group_loss = 0.0
                        if cfg["max_steps"] is not None and state["step"] >= cfg["max_steps"]:
                            exhausted = False
                            stop = True
                            break
                if group_micro:
                    step_group(group_tokens, group_sequences, group_micro, group_loss)
                if not state["valid_tokens"]:
                    raise ValueError("Training split is empty")
                from itertools import islice

                batches = (
                    islice(validation_loader, validation_batches)
                    if validation_batches
                    else validation_loader
                )
                metrics = validate(model, batches, device, precision)
                improved = metrics["nll"] < state["best_nll"]
                if improved:
                    state.update(best_nll=metrics["nll"], best_epoch=epoch, best_step=state["step"])
                early = cfg["early_stopping"]
                if metrics["nll"] < state["early_reference"] - early["min_delta"]:
                    state["early_reference"] = metrics["nll"]
                    state["bad_epochs"] = 0
                else:
                    state["bad_epochs"] += 1
                state["elapsed_seconds"] = initial_elapsed + perf_counter() - started
                log(validation=metrics)
                if exhausted:
                    state["epoch"], state["batch_in_epoch"] = epoch + 1, 0
                safe_rng = rng_state()
                checkpoint(best=improved, archive=cfg["checkpoint"]["save_every_epoch"])
                summary["validation"] = metrics
                if early["enabled"] and state["bad_epochs"] >= early["patience"]:
                    summary["status"] = "EARLY_STOPPED"
                    break
                if stop or (cfg["max_steps"] is not None and state["step"] >= cfg["max_steps"]):
                    break
            if summary["status"] == "RUNNING":
                summary["status"] = "COMPLETED"
            if not (checkpoint_dir / "best.pt").is_file():
                raise ValueError("No validated best checkpoint available to export")
            best = load_checkpoint(checkpoint_dir / "best.pt")
            export_best(
                export_dir,
                best,
                {
                    "model_type": "decoder_only_transformer",
                    "run_id": run_id,
                    **fingerprints,
                    "architecture": model.config.to_dict(),
                    "parameter_count": architecture["total_parameters"],
                    "best_validation_nll": best["state"]["best_nll"],
                    "best_validation_perplexity": perplexity(best["state"]["best_nll"]),
                    "best_epoch": best["state"]["best_epoch"],
                    "best_optimizer_step": best["state"]["best_step"],
                    "precision": precision,
                    "training_duration_seconds": state["elapsed_seconds"],
                    "training_valid_tokens": state["valid_tokens"],
                    "pytorch_version": str(torch.__version__),
                    "gpu_name": info.gpu_name,
                    "validation_batch_limit": validation_batches,
                    "project_version": "0.7.0",
                },
                overwrite=overwrite_export,
            )
            summary.update(
                optimizer_steps=state["step"],
                best_validation_nll=state["best_nll"],
                valid_tokens=state["valid_tokens"],
                export_directory=str(export_dir),
            )
    except KeyboardInterrupt:
        summary.update(
            status="INTERRUPTED", message="Interrupted; resume the latest consistent checkpoint."
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if checkpoint is not None and safe_rng is not None and not dry_run and not in_step:
            try:
                checkpoint(interrupt=True)
            except OSError as exc:
                summary["message"] += f" Interrupt checkpoint could not be written: {exc}"
    except Exception as exc:
        status = (
            "FAILED_OOM"
            if isinstance(exc, torch.cuda.OutOfMemoryError)
            else "FAILED_NONFINITE"
            if isinstance(exc, NonFiniteError)
            else "FAILED_CONFIGURATION"
            if isinstance(exc, (ValueError, ConfigurationError, DeviceUnavailableError))
            else "FAILED_OTHER"
        )
        message = str(exc)
        if status == "FAILED_OOM":
            message += (
                " Reduce batch size or context length; use accumulation for the effective batch."
            )
        summary.update(
            status=status,
            message=message,
            optimizer_steps=state["step"],
            learning_rate=optimizer.param_groups[0]["lr"] if optimizer else None,
            batch_size=cfg["batch_size"],
            context_length=values["model"]["context_length"],
        )
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        for batches in (train_loader, validation_loader):
            if batches is not None:
                close_loader(batches)
        summary.update(completed_at=datetime.now(UTC).isoformat(), resources=resources(device))
        summary["duration_seconds"] = perf_counter() - started if started is not None else 0
        atomic_json(directory / "summary.json", summary)
        if summary["status"] == "FAILED_OOM" and device.type == "cuda":
            torch.cuda.empty_cache()
    if summary["status"].startswith("FAILED"):
        raise TrainingFailure(summary)
    return summary
