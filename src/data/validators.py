"""Filesystem and bounded extraction guards shared by every ingestion entry point."""

import stat
from pathlib import Path

from src.data.models import ExtractionError, IngestionOptions, Status

SUPPORTED = {".txt", ".csv", ".pdf", ".docx"}


def validate_source(path: Path, root: Path, options: IngestionOptions) -> int:
    """Reject symlinks, paths outside root, non-regular files, and oversized sources."""
    try:
        relative = path.absolute().relative_to(root.resolve())
        current = root.resolve()
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ExtractionError(Status.UNSAFE_PATH, "Symlinks are not ingested")
        if not path.resolve().is_relative_to(root.resolve()):
            raise ExtractionError(Status.UNSAFE_PATH, "Source escapes the selected root")
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ExtractionError(Status.UNSAFE_PATH, "Source is not a regular file")
        if path.suffix.lower() not in SUPPORTED:
            raise ExtractionError(Status.UNSUPPORTED_TYPE, "Unsupported file extension")
        if info.st_size > options.max_bytes:
            raise ExtractionError(Status.FILE_TOO_LARGE, "File exceeds configured byte limit")
        if info.st_size == 0:
            raise ExtractionError(Status.EMPTY_FILE, "Source file is empty")
        return info.st_size
    except ValueError as exc:
        raise ExtractionError(Status.UNSAFE_PATH, "Source escapes the selected root") from exc
    except OSError as exc:
        raise ExtractionError(Status.READ_ERROR, "Cannot read source metadata") from exc


def check_text(text: str) -> None:
    """Reject binary control characters without altering or discarding text."""
    if any(ord(char) < 32 and char not in "\t\r\n\f" for char in text):
        raise ExtractionError(Status.ENCODING_ERROR, "Binary-like control characters in text")


def bounded_join(parts, limit: int, separator: str = "\n") -> str:
    """Join one document's pieces with a configured character ceiling."""
    chunks = []
    count = 0
    for piece in parts:
        count += len(piece) + (len(separator) if chunks else 0)
        if count > limit:
            raise ExtractionError(Status.EXTRACTION_TOO_LARGE, "Extracted text exceeds limit")
        chunks.append(piece)
    return separator.join(chunks)
