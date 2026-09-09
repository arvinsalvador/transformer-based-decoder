"""Strict predeclared plans and stable fingerprints; no training side effects."""

import hashlib
import json
import math
import re

import yaml


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_plan(path):
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or set(data) != {"experiment"}:
        raise ValueError("Expected experiment mapping")
    plan = data["experiment"]
    fields = {
        "name",
        "subset_seed",
        "scales",
        "run_trigram",
        "run_transformer",
        "evaluate_test",
        "require_transformer_dry_run",
        "stop_on_failure",
        "sequential_models",
        "large_run_threshold_documents",
        "require_large_run_confirmation",
        "minimum_free_disk_gb",
    }
    if not isinstance(plan, dict) or set(plan) != fields:
        raise ValueError(f"Experiment fields must be exactly {sorted(fields)}")
    if not isinstance(plan["name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", plan["name"]):
        raise ValueError("Experiment name must contain safe filesystem characters")
    for key in ("subset_seed", "large_run_threshold_documents"):
        if (
            type(plan[key]) is not int
            or not (0 if key == "subset_seed" else 1) <= plan[key] <= 2**32 - 1
        ):
            raise ValueError(f"Invalid {key}")
    for key in (
        "run_trigram",
        "run_transformer",
        "evaluate_test",
        "require_transformer_dry_run",
        "stop_on_failure",
        "sequential_models",
        "require_large_run_confirmation",
    ):
        if type(plan[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    # Safety policies are intentionally non-disableable in the initial implementation.
    for key in (
        "sequential_models",
        "stop_on_failure",
        "require_large_run_confirmation",
        "require_transformer_dry_run",
    ):
        if not plan[key]:
            raise ValueError(f"Phase 9 requires {key}=true")
    if not (plan["run_trigram"] or plan["run_transformer"]):
        raise ValueError("At least one model must be enabled")
    if plan["evaluate_test"] and not (plan["run_trigram"] and plan["run_transformer"]):
        raise ValueError("Shared test comparison requires both models enabled")
    value = plan["minimum_free_disk_gb"]
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("minimum_free_disk_gb must be finite and positive")
    scales = plan["scales"]
    if not isinstance(scales, list) or not scales:
        raise ValueError("scales must be nonempty")
    previous, seen_full = 0, False
    for scale in scales:
        if scale == "full" and not seen_full:
            seen_full = True
        elif type(scale) is int and 0 < scale < 100000 and scale > previous and not seen_full:
            previous = scale
        else:
            raise ValueError(
                "Scales must be ascending unique positive integers below 100000, then optional full"
            )
    return plan


def scale_plan(plan, train_count):
    return [
        {
            "scale": str(scale),
            "requested_train_documents": scale,
            "actual_train_documents": train_count
            if scale == "full"
            else scale
            if scale <= train_count
            else 0,
            "status": "PENDING" if scale == "full" or scale <= train_count else "NOT_APPLICABLE",
        }
        for scale in plan["scales"]
    ]
