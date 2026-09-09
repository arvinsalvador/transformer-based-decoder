"""Evidence-based tables and atomic Markdown/JSON/CSV/HTML export."""

import csv
import html
import io
import os
from datetime import UTC, datetime
from tempfile import NamedTemporaryFile

from src.evaluation.artifacts import write_json
from src.reporting.audit import runtime_audit, source_audit
from src.reporting.results import load_results

LIMITATIONS = [
    "One primary seed; no multi-seed uncertainty estimate or distributed training.",
    "The WordPiece vocabulary is fixed across model-training scales.",
    "Trigram sees two history tokens; Transformer context is capped and resets at scoring windows.",
    "Generation recomputes context without a KV cache; timings depend on hardware/software.",
    "Artifact hashes prove consistency, not unseen manual decisions or domain generalization.",
    "Missing historical peak RAM/VRAM is unavailable, not estimated; final RSS is not peak RAM.",
]
TALKING_POINTS = {
    "Why WordPiece?": (
        "A shared fixed subword vocabulary makes token-level comparisons more defensible."
    ),
    "Why trigram?": "A transparent count-based next-token baseline using two previous tokens.",
    "Why a causal mask?": "The next-token model must not attend to future targets.",
    "What is perplexity?": "Exponential of average negative log-likelihood on shared targets.",
    "What is the cap?": "100,000 total clean documents, not 100,000 training documents.",
    "Why GPU?": (
        "Parallel matrix operations can accelerate neural training; speed must be measured."
    ),
}


def table(result):
    if result["status"] != "PASS":
        return []
    models = result["comparison"]["results"]
    rows = []
    for split in ("train", "validation", "test"):
        rows.append(
            {
                "metric": split + " documents",
                "trigram": result["counts"][split],
                "transformer": result["counts"][split],
                "unit": "documents",
                "note": "Same canonical documents; FULL primary",
            }
        )
    rows.append(
        {
            "metric": "WordPiece vocabulary",
            "trigram": result["tokenizer_vocabulary"],
            "transformer": result["tokenizer_vocabulary"],
            "unit": "IDs",
            "note": "Same train-fitted tokenizer",
        }
    )
    for key, unit, note in (
        ("prediction_events", "targets", "WordPiece tokens plus EOS; no extra BOS target"),
        ("average_nll", "nats/target", "Natural logarithm"),
        ("perplexity", "ratio", "exp(NLL)"),
        (
            "training_duration_seconds",
            "seconds",
            "Existing model-training manifest; not end-to-end pipeline time",
        ),
        ("model_size_bytes", "bytes", "Deployed model only; no optimizer/checkpoints"),
        (
            "evaluation_seconds",
            "seconds",
            "Includes stream/tokenization/scoring; excludes model load",
        ),
        ("tokens_per_second", "targets/second", "Hardware-dependent"),
        ("peak_training_ram_mb", "MiB", "N/A if not recorded; not final RSS"),
        ("peak_training_vram_mb", "MiB", "N/A if not recorded"),
        ("parameter_count", "parameters", "Transformer only; not equivalent to n-gram counts"),
        ("unique_trigrams", "n-grams", "Trigram only"),
    ):
        rows.append(
            {
                "metric": key,
                "trigram": models["trigram"].get(key),
                "transformer": models["transformer"].get(key),
                "unit": unit,
                "note": note,
            }
        )
    # Phase 9 retained allocator peak in the training summary, even if export omitted it.
    for row in rows:
        if row["metric"] == "peak_training_vram_mb" and row["transformer"] is None:
            row["transformer"] = result["training_resources"].get("gpu_peak_mb")
            row["note"] = "Transformer allocator peak from saved training summary when available"
    return rows


def compile_report(settings, *, experiment=None, latest_complete=False, readiness_only=False):
    source = source_audit()
    runtime = runtime_audit(settings)
    final = (
        {"status": "NOT_RUN", "message": "Readiness-only mode; final artifacts were not inspected"}
        if readiness_only
        else load_results(settings, experiment, latest_complete)
    )
    compliance = []
    statuses = {
        "decoder_only_transformer": final["status"],
        "total_corpus_cap": runtime["data"]["status"],
        "wordpiece": runtime["tokenizer"]["status"],
        "trigram_comparison": final["status"],
        "performance_training_time": final["status"],
    }
    for name, implementation in source["implementation"].items():
        compliance.append(
            {
                "requirement": name,
                "implementation": implementation["status"],
                "execution": statuses[name],
                "evidence": implementation["evidence"],
            }
        )
    failures = (
        source["status"] == "FAIL"
        or any(value["status"] == "FAIL" for value in runtime.values())
        or final["status"] == "FAIL"
    )
    complete = final["status"] == "PASS" and not failures
    state = "FULLY_COMPLETE" if complete else "IMPLEMENTATION_COMPLETE_EXPERIMENT_PENDING"
    return {
        "project": "Transformer-Based Decoder-Only Language Model",
        "generated_at": datetime.now(UTC).isoformat(),
        "project_status": state,
        "audit_status": "FAIL" if failures else "PASS",
        "readiness_only": readiness_only,
        "system_audit": source,
        "runtime": runtime,
        "compliance": compliance,
        "final": final,
        "comparison_table": table(final),
        "limitations": LIMITATIONS,
        "talking_points": TALKING_POINTS,
        "recommendation": "NOT READY — audit failures require correction"
        if failures
        else "READY FOR SUBMISSION"
        if complete
        else "IMPLEMENTATION READY; prepare/verify runtime inputs and GPU before execution",
    }


def display(value):
    if value is None:
        return "N/A"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def safe_text(value):
    return html.escape(str(value)).replace("|", "\\|").replace("\n", " ")


def markdown(report):
    final = report["final"]
    lines = [
        f"# {report['project']}",
        "",
        f"Status: **{report['project_status']}**",
        "",
        f"Audit: {report['audit_status']}",
        "",
    ]
    sections = [
        (
            "Homework Requirements",
            "Implementation evidence and final-experiment evidence are separate; "
            "see the compliance checklist below.",
        ),
        (
            "Objective",
            "Train a custom causal decoder-only Transformer and compare it with a trigram "
            "model using common WordPiece targets.",
        ),
        (
            "Dataset",
            "Maximum: 100,000 TOTAL clean documents. Actual counts: "
            + safe_text(
                final.get("counts") or report["runtime"]["data"].get("counts") or "NOT AVAILABLE"
            ),
        ),
        (
            "Data Pipeline",
            "Ingestion → normalization/filtering → exact deduplication → canonical "
            "train/validation/test split. WordPiece fits train only.",
        ),
        ("WordPiece Tokenizer", safe_text(report["runtime"]["tokenizer"])),
        (
            "Trigram Baseline",
            "Lidstone/add-k P(token | previous two tokens), with BOS/BOS history and one "
            "EOS target; SQLite read-only evaluation.",
        ),
        (
            "Decoder-Only Transformer Architecture",
            "Token + learned positional embeddings → pre-norm causal multi-head "
            "attention/FFN residual blocks → LayerNorm → tied vocabulary head. "
            "Architecture fingerprint is retained in final metrics.",
        ),
        (
            "Training Configuration",
            "AdamW, warmup/cosine, token-weighted accumulation and clipping; "
            "validation selects best. Final resolved settings: "
            + safe_text(final.get("resolved_plan", {}).get("settings", "NOT AVAILABLE")),
        ),
        (
            "Evaluation Methodology",
            "Same tokenizer/test documents; score every WordPiece token plus EOS once. "
            "Non-overlapping Transformer windows reset context. Natural-log NLL; "
            "perplexity = exp(NLL). Hardware-dependent timing.",
        ),
        (
            "Final Results",
            "Validated FULL result: " + safe_text(final.get("experiment_id"))
            if final["status"] == "PASS"
            else "FINAL EXPERIMENT NOT YET EXECUTED or not validated. "
            + safe_text(final.get("message", "")),
        ),
    ]
    for number, (title, body) in enumerate(sections, 1):
        lines.extend([f"## {number}. {title}", "", body, ""])
    lines.extend(["## 11. Trigram vs Transformer Comparison", ""])
    if report["comparison_table"]:
        lines += ["| Metric | Trigram | Transformer | Unit | Note |", "|---|---:|---:|---|---|"]
        for row in report["comparison_table"]:
            lines.append(
                "| "
                + " | ".join(
                    safe_text(display(row[key]))
                    for key in ("metric", "trigram", "transformer", "unit", "note")
                )
                + " |"
            )
    else:
        lines.append("No validated final metrics available; no numeric result table is invented.")
    lines += [
        "",
        "## 12. Scaling Experiment",
        "",
        "FULL is primary, never the best-looking scale. Other available/failed scales:",
        "",
    ]
    for scale in final.get("learning_curve", []):
        lines.append(
            f"- {safe_text(scale['scale'])}: {safe_text(scale['status'])}; "
            f"train documents {scale['train_documents']}"
        )
    lines += [
        "",
        "## 13. Generated Text Examples",
        "",
        "Saved outputs only; no regeneration or cherry-picking. "
        "Human ratings remain unfilled when absent.",
        "",
    ]
    for item in final.get("generation", []):
        lines += [
            f"### {safe_text(item['model_type'])}: {safe_text(item['prompt'])}",
            "",
            safe_text(item["generated_text"]),
            "",
            "Human review: " + safe_text(item.get("human_review", "Not rated")),
            "",
        ]
    lines += [
        "## 14. Computational Cost",
        "",
        "Use saved hardware and timing definitions; missing memory is N/A. Relative metrics: "
        + safe_text(final.get("relative", {})),
        "",
        "## 15. Limitations",
        "",
    ]
    lines += ["- " + text for text in [*LIMITATIONS, *final.get("warnings", [])]]
    lines += ["", "## 16. Conclusion", ""]
    lines += final.get(
        "interpretation",
        [
            "No final measured conclusion is available. "
            "Final GPU experiment has not yet been executed or validated."
        ],
    )
    lines += [
        "",
        "## 17. Reproducibility Information",
        "",
        "Experiment: " + safe_text(final.get("experiment_id", "NOT RUN")),
        "",
        "Fingerprints: " + safe_text(final.get("fingerprints", {})),
        "",
        "Use docs/GPU_RUNBOOK.md and docs/REPRODUCIBILITY.md for execution/recovery instructions.",
        "",
        "## 18. Homework Compliance Checklist",
        "",
    ]
    for row in report["compliance"]:
        lines.append(
            f"- {row['requirement']}: implementation {row['implementation']}; "
            f"execution {row['execution']}"
        )
    lines += ["", "## Presentation Talking Points", ""]
    lines += [f"- {key} {value}" for key, value in TALKING_POINTS.items()]
    return "\n".join(lines) + "\n"


def atomic_text(path, text):
    temporary = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def export_report(settings, output, **options):
    output = output.resolve()
    root = settings.paths["REPORT_DIR"].resolve()
    if output != root and root not in output.parents:
        raise ValueError("Reports must be beneath REPORT_DIR")
    report = compile_report(settings, **options)
    output.mkdir(parents=True, exist_ok=True)
    text = markdown(report)
    atomic_text(output / "final_report.md", text)
    atomic_text(
        output / "final_report.html",
        "<!doctype html><meta charset='utf-8'><title>Decoder LM report</title>"
        "<style>body{max-width:1100px;margin:2rem auto;padding:1rem}"
        "pre{white-space:pre-wrap;font:16px/1.5 system-ui}</style><pre>"
        + html.escape(text)
        + "</pre>",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["metric", "trigram", "transformer", "unit", "note"])
    writer.writeheader()
    writer.writerows(report["comparison_table"])
    atomic_text(output / "final_comparison.csv", stream.getvalue())
    write_json(
        output / "system_audit.json",
        {
            "source": report["system_audit"],
            "runtime": report["runtime"],
            "compliance": report["compliance"],
            "final_integrity": report["final"]["status"],
        },
    )
    report["report_paths"] = {
        name: str(output / name)
        for name in (
            "final_report.md",
            "final_report.html",
            "final_comparison.csv",
            "system_audit.json",
            "final_summary.json",
        )
    }
    # Completion authority published last; no model/data artifacts are modified.
    write_json(output / "final_summary.json", report)
    return report
