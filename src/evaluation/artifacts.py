"""Portable strict JSON and atomic CSV publication inside unique evaluation runs."""

import csv
import math
import os
from tempfile import NamedTemporaryFile

from src.training.checkpoint import atomic_json


def portable(value):
    # JSON has no Infinity literal. Preserve overflow explicitly, not invalid JSON.
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else "nan"
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable(v) for v in value]
    return value


def write_json(path, value):
    atomic_json(path, portable(value))


TABLE = {
    "documents": "documents",
    "prediction_events": "targets",
    "average_nll": "nats/target",
    "perplexity": "ratio",
    "evaluation_seconds": "seconds",
    "tokens_per_second": "targets/second",
    "training_duration_seconds": "seconds",
    "model_size_bytes": "bytes",
    "peak_training_ram_mb": "MiB",
    "peak_training_vram_mb": "MiB",
    "training_tokens_per_second": "targets/second",
}


def table(results):
    return [
        {"metric": key, "unit": unit, **{name: result.get(key) for name, result in results.items()}}
        for key, unit in TABLE.items()
    ]


def write_csv(path, rows, evaluation_id, fingerprints):
    temporary = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as stream:
            temporary = stream.name
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "evaluation_id",
                    "test_fingerprint",
                    "tokenizer_fingerprint",
                    "metric",
                    "trigram",
                    "transformer",
                    "unit",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "evaluation_id": evaluation_id,
                        "test_fingerprint": fingerprints["test_fingerprint"],
                        "tokenizer_fingerprint": fingerprints["tokenizer_fingerprint"],
                        **row,
                    }
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
