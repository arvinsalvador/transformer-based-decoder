"""Streaming ingestion orchestration shared by the CLI and UI."""

import hashlib
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from src.data.discovery import discover_files
from src.data.models import (
    DocumentRecord,
    ExtractionError,
    IngestionOptions,
    ParsedText,
    Status,
    utc_now,
)
from src.data.registry import get_parser
from src.data.statistics import IngestionStatistics
from src.data.validators import validate_source

logger = logging.getLogger(__name__)
MIME = {
    "txt": "text/plain",
    "csv": "text/csv",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@dataclass
class IngestionResult:
    """Artifact locations, actual summary, and a strictly bounded UI preview."""

    output_path: Path
    manifest_path: Path
    summary: dict
    preview: list[dict]


def _parse_records(path: Path, root: Path, options: IngestionOptions) -> Iterator[ParsedText]:
    try:
        validate_source(path, root, options)
        parser = get_parser(path.suffix)
        records = parser(path, options)
        try:
            yield from records
        finally:
            records.close()
    except ExtractionError as exc:
        yield ParsedText(status=exc.status, error_message=str(exc))
    except OSError:
        yield ParsedText(status=Status.READ_ERROR, error_message="Cannot read source file")
    except Exception as exc:
        # Library errors can contain document text. Report only the exception class.
        yield ParsedText(
            status=Status.PARSE_ERROR,
            error_message=f"Parser rejected source ({type(exc).__name__})",
        )


def _record(path: Path, size: int, parsed: ParsedText, options: IngestionOptions) -> DocumentRecord:
    status = parsed.status
    reason = parsed.error_message
    if status == Status.SUCCESS:
        if not parsed.text.strip():
            status, reason = Status.NO_EXTRACTABLE_TEXT, "No extractable non-whitespace text"
        elif len(parsed.text) < options.settings.values["dataset"]["min_extracted_characters"]:
            status, reason = Status.TOO_SHORT, "Text is below minimum character count"
    source_type = path.suffix.lower().lstrip(".")
    source_type = source_type if source_type in MIME else "unsupported"
    return DocumentRecord(
        document_id=uuid4().hex,
        source_name=path.name,
        source_path=str(path),
        source_type=source_type,
        file_size_bytes=size,
        extracted_text=parsed.text,
        character_count=len(parsed.text),
        extraction_status=status,
        error_message=reason,
        ingested_at=utc_now(),
        sha256=hashlib.sha256(parsed.text.encode("utf-8")).hexdigest() if parsed.text else None,
        metadata={
            "extension": path.suffix.lower(),
            "mime_type": MIME.get(source_type),
            "hash_scope": "extracted_text_utf8",
            **parsed.metadata,
        },
    )


def ingest_directory(
    source_dir: str | Path,
    options: IngestionOptions,
    *,
    output: str | Path | None = None,
    progress: Callable[[dict], None] | None = None,
) -> IngestionResult:
    """Ingest at most the effective limit of candidate records, including rejections.

    Existing output is never overwritten. All outcomes are written to JSONL; consumers
    must select SUCCESS records. On interruption, the manifest marks partial output.
    No dataset is parsed automatically: the caller must explicitly invoke this function.
    """
    source = Path(source_dir).absolute()
    if source.is_symlink() or not source.is_dir() or source != source.resolve():
        raise ValueError("source must be an existing directory without symlink components")
    output_path = Path(output or options.output_path).absolute()
    if output_path.is_symlink():
        raise ValueError("output must not be a symlink")
    output_path = output_path.resolve()
    manifest_dir = options.manifest_dir.resolve()
    if output_path.is_relative_to(source) or manifest_dir.is_relative_to(source):
        raise ValueError("Output and manifest directories must be outside the source corpus")
    if output_path.exists():
        raise FileExistsError(f"Output already exists; choose a new filename: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    manifest_path = manifest_dir / f"{run_id}.json"
    stats = IngestionStatistics()
    preview = []
    config = options.settings.values["ingestion"]
    summary = {
        "ingestion_id": run_id,
        "started_at": utc_now(),
        "completed_at": None,
        "source": str(source),
        "output_path": str(output_path),
        "state": "running",
        "configured_limit": options.limit,
        "limit_reached": False,
        "configuration": options.settings.values,
        "effective_csv": {"mode": options.csv_mode, "text_columns": options.text_columns},
        "recursive": options.recursive,
        "count_policy": "Every emitted candidate or rejection consumes one limit slot",
    }

    def save_manifest() -> None:
        # A UUID filename belongs only to this run. Atomic replacement avoids half a manifest.
        temporary = manifest_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump({**summary, **stats.snapshot()}, stream, ensure_ascii=False, indent=2)
        temporary.replace(manifest_path)

    save_manifest()
    paths = discover_files(source, options.recursive)
    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as sink:
            while stats.records_written < options.limit:
                path = next(paths, None)
                if path is None:
                    break
                try:
                    size = path.lstat().st_size
                except OSError:
                    size = 0
                source_type = path.suffix.lower().lstrip(".")
                stats.examine_file(source_type if source_type in MIME else "unsupported", size)
                candidates = _parse_records(path, source, options)
                try:
                    while stats.records_written < options.limit:
                        parsed = next(candidates, None)
                        if parsed is None:
                            break
                        record = _record(path, size, parsed, options)
                        payload = asdict(record)
                        sink.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        stats.add(record)
                        if len(preview) < config["preview_documents"]:
                            preview.append(
                                {
                                    **payload,
                                    "extracted_text": record.extracted_text[
                                        : config["preview_characters"]
                                    ],
                                }
                            )
                        if (
                            record.extraction_status != Status.SUCCESS
                            and stats.failed + stats.skipped <= 20
                        ):
                            logger.warning(
                                "Source %r: %s (%s)",
                                path.name,
                                record.extraction_status,
                                record.error_message,
                            )
                        if stats.records_written % config["progress_interval"] == 0:
                            sink.flush()
                            logger.info(
                                "Processed %d / %d candidates", stats.records_written, options.limit
                            )
                            if progress:
                                progress(stats.snapshot())
                        # Only the current document and capped previews survive.
                finally:
                    candidates.close()
            sink.flush()
        summary["limit_reached"] = stats.records_written == options.limit
        summary["state"] = "completed"
    except BaseException:
        summary["state"] = "interrupted"
        logger.error("Ingestion interrupted; partial output is recorded in the manifest")
        raise
    finally:
        paths.close()
        summary["completed_at"] = utc_now()
        save_manifest()
    final = {**summary, **stats.snapshot()}
    if progress:
        progress(stats.snapshot())
    logger.info(
        "Ingestion complete: %d successful, %d skipped, %d failed",
        stats.successful,
        stats.skipped,
        stats.failed,
    )
    return IngestionResult(output_path, manifest_path, final, preview)
