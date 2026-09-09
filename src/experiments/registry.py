"""Atomic status and chart-ready matrices; summary is the completion authority."""

import csv
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.evaluation.artifacts import write_json
from src.evaluation.identity import read_json
from src.training.reproducibility import file_hash


def seal(paths):
    return {str(path): file_hash(path) for path in paths}


def check_seal(record):
    if not record.get("artifacts"):
        raise ValueError("Completed stage lacks artifact fingerprints")
    for path, expected in record.get("artifacts", {}).items():
        if file_hash(path) != expected:
            raise ValueError(f"Completed artifact fingerprint changed: {path}")


def save(directory, status):
    status["updated_at"] = datetime.now(UTC).isoformat()
    write_json(directory / "status.json", status)
    rows = []
    for scale in status["scales"]:
        row = {
            key: scale.get(key)
            for key in (
                "scale",
                "requested_train_documents",
                "actual_train_documents",
                "status",
                "subset_fingerprint",
            )
        }
        for name in ("trigram", "transformer", "evaluation"):
            row[name + "_status"] = scale.get("stages", {}).get(name, {}).get("status", "PENDING")
        training = scale.get("stages", {}).get("transformer", {}).get("result", {})
        row["transformer_optimizer_steps"] = training.get("optimizer_steps")
        row["transformer_valid_tokens"] = training.get("valid_tokens")
        row["transformer_peak_vram_mb"] = training.get("resources", {}).get("gpu_peak_mb")
        row["transformer_final_process_ram_mb"] = training.get("resources", {}).get(
            "process_ram_mb"
        )
        comparison_path = scale.get("comparison_path")
        if comparison_path and scale["status"] == "COMPLETED":
            comparison = read_json(comparison_path)
            for name, result in comparison["results"].items():
                for key in (
                    "average_nll",
                    "perplexity",
                    "training_duration_seconds",
                    "model_size_bytes",
                    "parameter_count",
                    "tokens_per_second",
                    "peak_training_ram_mb",
                    "peak_training_vram_mb",
                ):
                    row[f"{name}_{key}"] = result.get(key)
            row.update(comparison["relative"])
            seconds = row.get("transformer_training_duration_seconds")
            tokens = row["transformer_valid_tokens"]
            row["transformer_training_tokens_per_second"] = (
                tokens / seconds if tokens is not None and seconds else None
            )
        scale_dir = directory / "scales" / scale["scale"]
        scale_dir.mkdir(parents=True, exist_ok=True)
        write_json(scale_dir / "status.json", scale)
        write_json(scale_dir / "scale_summary.json", row)
        rows.append(row)
    matrix = {
        "experiment_id": status["experiment_id"],
        "plan_fingerprint": status["plan_fingerprint"],
        "audit": status["audit"],
        "rows": rows,
        "status": status["status"],
        "resolved_plan": read_json(directory / "experiment_plan.json"),
        "environment": read_json(directory / "environment.json")
        if (directory / "environment.json").exists()
        else None,
    }
    write_json(directory / "experiment_matrix.json", matrix)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=directory, delete=False
        ) as stream:
            temporary = stream.name
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "experiment_matrix.csv")
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    full = next(
        (
            s
            for s in status["scales"]
            if s["scale"] == "full" and s["status"] == "COMPLETED" and s.get("comparison_path")
        ),
        None,
    )
    write_json(
        directory / "summary.json",
        {
            **matrix,
            "primary_final_result": bool(full),
            "full_result": read_json(full["comparison_path"]) if full else None,
            "full_generation_path": str(
                Path(full["comparison_path"]).parent / "generation_results.json"
            )
            if full
            else None,
        },
    )


def log(directory, scale, stage, status):
    with (directory / "experiment.log").open("a", encoding="utf-8") as stream:
        stream.write(
            f"{datetime.now(UTC).isoformat()} scale={scale} stage={stage} status={status}\n"
        )
