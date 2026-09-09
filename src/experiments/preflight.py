"""Bounded corpus auditing, actual hashes, device checks and tiny writable probes."""

import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import psutil
import torch

from src.config.settings import validate_config
from src.evaluation.identity import read_json
from src.experiments.plan import scale_plan
from src.training.monitoring import environment
from src.training.precision import select_precision
from src.training.reproducibility import file_hash, verify_inputs
from src.transformer.factory import build
from src.transformer.inspection import inspect
from src.utils.device import detect_device


def enforce_cap(counts):
    if any(type(value) is not int or value < 1 for value in counts.values()):
        raise ValueError("Every canonical split must contain records")
    if sum(counts.values()) > 100000:
        raise ValueError("Homework limit is 100,000 TOTAL corpus documents, not training documents")


def audit_data(settings, plan, dataset_manifest=None):
    counts, hashes = {}, {}
    for name in ("train", "validation", "test"):
        path = settings.paths["DATA_DIR"] / f"splits/{name}.jsonl"
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    raise ValueError(f"Invalid {name} record")
                count += 1
                if sum(counts.values()) + count > 100000:
                    raise ValueError("Homework limit is 100,000 TOTAL corpus documents")
        counts[name], hashes[name] = count, file_hash(path)
    enforce_cap(counts)
    if dataset_manifest is None:
        candidates = settings.paths["EXPERIMENT_DIR"].joinpath("preprocessing").glob("*.json")
        matches = []
        for path in candidates:
            data = read_json(path)
            if data.get("state") == "completed" and data.get("split_fingerprints") == hashes:
                matches.append(path)
        if len(matches) != 1:
            raise ValueError(
                "Supply --dataset-manifest: exactly one matching completed Phase 3 "
                "manifest is required"
            )
        dataset_manifest = matches[0]
    dataset = read_json(dataset_manifest)
    if dataset.get("state") != "completed" or dataset.get("split_fingerprints") != hashes:
        raise ValueError("Completed Phase 3 manifest must match all canonical split fingerprints")
    if not dataset.get("dataset_fingerprint"):
        raise ValueError("Dataset fingerprint is required")
    if dataset.get("accepted_documents", sum(counts.values())) != sum(counts.values()):
        raise ValueError("Clean-corpus count differs from canonical split total")
    if dataset.get("split_counts", counts) != counts:
        raise ValueError("Recorded split counts differ from actual canonical counts")
    return {
        "counts": counts,
        "total_corpus_documents": sum(counts.values()),
        "split_fingerprints": hashes,
        "dataset_fingerprint": dataset["dataset_fingerprint"],
        "dataset_manifest": str(Path(dataset_manifest).resolve()),
        "scales": scale_plan(plan, counts["train"]),
    }


def storage_checks(paths, minimum_gb):
    checks = []
    for path in paths:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(path).free
        if free < minimum_gb * 10**9:
            raise ValueError(
                f"Insufficient disk at {path}: {free} bytes free; minimum {minimum_gb} GB"
            )
        with NamedTemporaryFile(prefix=".preflight-", dir=path):
            pass
        checks.append(
            {"path": str(path), "free_bytes": free, "minimum_gb": minimum_gb, "writable": True}
        )
    return checks


def preflight(settings, plan, audit, output):
    validate_config(settings.values)
    root = settings.paths["DATA_DIR"] / "splits"
    token_dir = settings.paths["MODEL_DIR"] / "tokenizer"
    _, fingerprints, warnings = verify_inputs(
        root / "train.jsonl",
        root / "validation.jsonl",
        token_dir,
        settings,
        audit["dataset_manifest"],
    )
    token_manifest = read_json(token_dir / "tokenizer_manifest.json")
    if token_manifest.get("training_split_fingerprint") != audit["split_fingerprints"]["train"]:
        raise ValueError("Canonical tokenizer must record matching train-only fitting provenance")
    info = detect_device(settings.values["device"] if plan["run_transformer"] else "cpu")
    precision = select_precision(
        torch.device(info.selected_device),
        settings.values["training"]["precision"],
        settings.values["training"]["mixed_precision"],
    )
    # Meta tensors verify model construction/counts without allocating full CPU/GPU weights.
    with torch.device("meta"):
        model, _ = build(settings, token_dir)
    architecture = inspect(model)
    return {
        "status": "READY",
        "fingerprints": fingerprints,
        "architecture": architecture,
        "warnings": warnings,
        "storage": storage_checks(
            [output, settings.paths["CHECKPOINT_DIR"], settings.paths["MODEL_DIR"]],
            plan["minimum_free_disk_gb"],
        ),
        "environment": {
            **environment(info, precision),
            "system_ram_bytes": psutil.virtual_memory().total,
            "container_detected": Path("/.dockerenv").exists(),
        },
    }
