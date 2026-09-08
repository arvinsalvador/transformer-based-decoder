"""Incremental training history, finite metric handling, and sampled resources."""

import csv
import math
import platform
import sys
from datetime import UTC, datetime

import psutil
import tokenizers
import torch

COLUMNS = (
    "timestamp",
    "epoch",
    "optimizer_step",
    "train_nll",
    "validation_nll",
    "validation_perplexity",
    "learning_rate",
    "valid_tokens",
    "sequences",
    "microbatches",
    "tokens_per_second",
    "sequences_per_second",
    "elapsed_seconds",
    "gradient_norm",
    "process_ram_mb",
    "system_available_mb",
    "gpu_allocated_mb",
    "gpu_reserved_mb",
    "gpu_peak_mb",
)


def perplexity(nll):
    return math.exp(nll) if nll < 709 else float("inf")


def resources(device):
    result = {
        "process_ram_mb": psutil.Process().memory_info().rss / 2**20,
        "system_available_mb": psutil.virtual_memory().available / 2**20,
    }
    for key, function in (
        ("gpu_allocated_mb", torch.cuda.memory_allocated),
        ("gpu_reserved_mb", torch.cuda.memory_reserved),
        ("gpu_peak_mb", torch.cuda.max_memory_allocated),
    ):
        result[key] = function(device) / 2**20 if device.type == "cuda" else None
    return result


def environment(device_info, precision):
    from dataclasses import asdict

    return {
        **asdict(device_info),
        "precision": precision,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "tokenizers_version": tokenizers.__version__,
    }


def history_row(path, row):
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({"timestamp": datetime.now(UTC).isoformat(), **row})
        stream.flush()
