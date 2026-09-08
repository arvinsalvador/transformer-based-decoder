"""Atomic, streaming Phase 3 pipeline: validate → normalize → deduplicate → split."""

import hashlib
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from src.config.settings import Settings
from src.data.models import utc_now
from src.data.preprocessing.deduplication import ExactDeduplicator, normalized_hash
from src.data.preprocessing.filters import quality_check
from src.data.preprocessing.models import CleanDocument, PreprocessingStatus, ProcessingOutcome
from src.data.preprocessing.normalizer import normalize
from src.data.preprocessing.splitter import assign_split
from src.data.preprocessing.statistics import PreparationStatistics

logger = logging.getLogger(__name__)


@dataclass
class PreparationResult:
    """Small result object safe for the CLI/UI session state."""

    clean_output: Path
    rejection_output: Path
    split_paths: dict[str, Path]
    manifest_path: Path
    summary: dict
    preview: list[dict]


def _iter_raw(path: Path) -> Iterator[dict | ProcessingOutcome]:
    """Read one JSONL line at a time and never expose raw text in parser errors/logs."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                yield ProcessingOutcome(PreprocessingStatus.INVALID_RECORD, "Invalid JSONL record")
                continue
            if not isinstance(item, dict):
                yield ProcessingOutcome(
                    PreprocessingStatus.INVALID_RECORD, "JSONL item must be an object"
                )
                continue
            yield item


def _raw_fields(raw: dict) -> tuple[str, str, str, str, str | None, dict] | ProcessingOutcome:
    """Validate only fields Phase 3 needs; Phase 2 schema remains the source of truth."""
    text = raw.get("extracted_text")
    document_id = raw.get("document_id")
    source_name = raw.get("source_name")
    source_type = raw.get("source_type")
    metadata = raw.get("metadata", {})
    if not all(isinstance(value, str) for value in (text, document_id, source_name, source_type)):
        return ProcessingOutcome(
            PreprocessingStatus.INVALID_RECORD, "Missing required Phase 2 text metadata"
        )
    if not isinstance(metadata, dict):
        return ProcessingOutcome(
            PreprocessingStatus.INVALID_RECORD, "Phase 2 metadata must be an object"
        )
    if raw.get("extraction_status") != "SUCCESS":
        return ProcessingOutcome(
            PreprocessingStatus.EXTRACTION_NOT_ACCEPTED, "Phase 2 extraction was not successful"
        )
    raw_hash = raw.get("sha256")
    if raw_hash is not None and not isinstance(raw_hash, str):
        return ProcessingOutcome(
            PreprocessingStatus.INVALID_RECORD, "Phase 2 hash must be a string or null"
        )
    return text, document_id, source_name, source_type, raw_hash, metadata


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def prepare_dataset(
    input_path: str | Path,
    settings: Settings,
    *,
    clean_output: str | Path | None = None,
    split_dir: str | Path | None = None,
    limit: int | None = None,
    seed: int | None = None,
    make_splits: bool = True,
    overwrite: bool = False,
    progress: Callable[[dict], None] | None = None,
) -> PreparationResult:
    """Create canonical clean/split JSONL without mutating Phase 2 input.

    Rejections are written to a sidecar with no text body. Hashes and numeric lengths are
    the only per-corpus in-memory state; accepted text exists only for the current record.
    """
    cfg = settings.values["preprocessing"]
    input_file = Path(input_path).absolute().resolve()
    if not input_file.is_file() or input_file.suffix != ".jsonl":
        raise ValueError("input must be an existing JSONL file")
    cap = min(
        settings.values["dataset"]["working_document_limit"],
        settings.values["dataset"]["max_documents"],
        100000,
    )
    effective_limit = cap if limit is None else limit
    if type(effective_limit) is not int or not 0 < effective_limit <= cap:
        raise ValueError(f"limit must be an integer between 1 and {cap}")
    chosen_seed = settings.values["random_seed"] if seed is None else seed
    if type(chosen_seed) is not int or not 0 <= chosen_seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in the configured seed range")
    clean = Path(
        clean_output or settings.paths["DATA_DIR"] / "processed/clean_documents.jsonl"
    ).absolute()
    clean = clean.resolve()
    rejected = clean.with_name(f"{clean.stem}.rejections.jsonl")
    root = Path(split_dir or settings.paths["DATA_DIR"] / "splits").absolute().resolve()
    targets = {name: root / f"{name}.jsonl" for name in ("train", "validation", "test")}
    all_outputs = [clean, rejected, *targets.values()] if make_splits else [clean, rejected]
    if input_file in all_outputs:
        raise ValueError("clean output must differ from the Phase 2 input")
    if not overwrite and any(path.exists() for path in all_outputs):
        raise FileExistsError("Output already exists; choose another path or use --overwrite")
    clean.parent.mkdir(parents=True, exist_ok=True)
    if make_splits:
        root.mkdir(parents=True, exist_ok=True)
    manifest_dir = settings.paths["EXPERIMENT_DIR"] / "preprocessing"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    manifest = manifest_dir / f"{run_id}.json"
    stats, deduplicator, preview = PreparationStatistics(), ExactDeduplicator(), []
    fingerprint = hashlib.sha256()
    temporary = {path: _temporary(path) for path in all_outputs}
    summary = {
        "preprocessing_id": run_id,
        "started_at": utc_now(),
        "completed_at": None,
        "state": "running",
        "input_path": str(input_file),
        "clean_output": str(clean),
        "split_paths": {name: str(path) for name, path in targets.items()} if make_splits else {},
        "limit": effective_limit,
        "seed": chosen_seed,
        "split_enabled": make_splits,
        "configuration": settings.values,
        "deduplication": "exact normalized SHA-256 before split",
    }

    def save_manifest() -> None:
        payload = {**summary, **stats.snapshot(), "dataset_fingerprint": fingerprint.hexdigest()}
        interim = manifest.with_suffix(".tmp")
        interim.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        interim.replace(manifest)

    def reject(stream, raw: dict | None, outcome: ProcessingOutcome) -> None:
        stats.reject(outcome.status)
        stream.write(
            json.dumps(
                {
                    "document_id": raw.get("document_id") if raw else None,
                    "source_name": raw.get("source_name") if raw else None,
                    "preprocessing_status": outcome.status,
                    "reason": outcome.reason,
                    "duplicate_of": outcome.duplicate_of,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    save_manifest()
    try:
        with (
            temporary[clean].open("x", encoding="utf-8", newline="\n") as clean_sink,
            temporary[rejected].open("x", encoding="utf-8", newline="\n") as rejection_sink,
        ):
            split_sinks = (
                {
                    name: temporary[path].open("x", encoding="utf-8", newline="\n")
                    for name, path in targets.items()
                }
                if make_splits
                else {}
            )
            try:
                for item in _iter_raw(input_file):
                    if stats.input_documents >= effective_limit:
                        break
                    if isinstance(item, ProcessingOutcome):
                        stats.input_documents += 1
                        reject(rejection_sink, None, item)
                        continue
                    fields = _raw_fields(item)
                    if isinstance(fields, ProcessingOutcome):
                        stats.input_documents += 1
                        reject(rejection_sink, item, fields)
                        continue
                    text, doc_id, source_name, source_type, raw_hash, metadata = fields
                    stats.input(len(text), source_type)
                    try:
                        normalized = normalize(text, cfg)
                        was_truncated = False
                        if len(normalized) > cfg["max_characters_per_document"]:
                            if cfg["long_document_policy"] == "skip":
                                reject(
                                    rejection_sink,
                                    item,
                                    ProcessingOutcome(
                                        PreprocessingStatus.TOO_LONG_SKIPPED,
                                        "Text exceeds maximum character count",
                                    ),
                                )
                                continue
                            normalized = normalized[: cfg["max_characters_per_document"]]
                            was_truncated = True
                        outcome = quality_check(text, normalized, cfg)
                        if outcome:
                            reject(rejection_sink, item, outcome)
                            continue
                        digest = normalized_hash(normalized)
                        prior = deduplicator.duplicate_of(digest, doc_id)
                        if prior:
                            reject(
                                rejection_sink,
                                item,
                                ProcessingOutcome(
                                    PreprocessingStatus.DUPLICATE,
                                    "Normalized text matches an accepted document",
                                    prior,
                                ),
                            )
                            continue
                        split = assign_split(digest, chosen_seed, settings.values["split"])
                        status = (
                            PreprocessingStatus.TRUNCATED
                            if was_truncated
                            else PreprocessingStatus.ACCEPTED
                        )
                        record = CleanDocument(
                            doc_id,
                            source_name,
                            source_type,
                            raw_hash,
                            digest,
                            normalized,
                            len(normalized),
                            len(text),
                            status,
                            was_truncated,
                            {"phase2_metadata": metadata},
                        )
                        payload = json.dumps(asdict(record), ensure_ascii=False) + "\n"
                        clean_sink.write(payload)
                        if make_splits:
                            split_sinks[split].write(payload)
                        stats.accept(len(normalized), split, was_truncated)
                        fingerprint.update(digest.encode("ascii"))
                        if len(preview) < cfg["preview_documents"]:
                            preview.append(
                                {
                                    "document_id": doc_id,
                                    "source_name": source_name,
                                    "source_type": source_type,
                                    "character_count": len(normalized),
                                    "preprocessing_status": status,
                                    "truncated": was_truncated,
                                    "text": normalized[: cfg["preview_characters"]],
                                }
                            )
                    except Exception as exc:
                        reject(
                            rejection_sink,
                            item,
                            ProcessingOutcome(
                                PreprocessingStatus.PROCESSING_ERROR,
                                f"Processing failed ({type(exc).__name__})",
                            ),
                        )
                    if stats.input_documents % cfg["progress_interval"] == 0:
                        clean_sink.flush()
                        rejection_sink.flush()
                        for sink in split_sinks.values():
                            sink.flush()
                        if progress:
                            progress(stats.snapshot())
            finally:
                for sink in split_sinks.values():
                    sink.close()
        for target, interim in temporary.items():
            interim.replace(target)
        summary["split_fingerprints"] = {}
        if make_splits:
            for name, path in targets.items():
                with path.open("rb") as stream:
                    summary["split_fingerprints"][name] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
        summary["state"] = "completed"
    except BaseException:
        summary["state"] = "interrupted"
        logger.error(
            "Preparation interrupted; temporary artifacts and manifest identify the partial run"
        )
        raise
    finally:
        summary["completed_at"] = utc_now()
        save_manifest()
    final = {**summary, **stats.snapshot(), "dataset_fingerprint": fingerprint.hexdigest()}
    if progress:
        progress(stats.snapshot())
    logger.info(
        "Preparation complete: %d accepted, %d rejected",
        stats.accepted_documents,
        stats.input_documents - stats.accepted_documents,
    )
    return PreparationResult(
        clean, rejected, targets if make_splits else {}, manifest, final, preview
    )
