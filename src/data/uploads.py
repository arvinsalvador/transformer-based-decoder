"""Bounded upload staging, with generated filenames and no trust in browser paths."""

from pathlib import Path
from typing import Protocol
from uuid import uuid4

from src.data.models import IngestionOptions
from src.data.validators import SUPPORTED


class Upload(Protocol):
    """Minimal interface also implemented by Streamlit UploadedFile."""

    name: str
    size: int

    def read(self, size: int = -1) -> bytes: ...
    def seek(self, offset: int) -> int: ...


def stage_uploads(uploads: list[Upload], options: IngestionOptions) -> Path:
    """Validate the whole small batch before copying originals under configured raw storage."""
    cfg = options.settings.values["ingestion"]
    if not uploads or len(uploads) > cfg["max_upload_files"]:
        raise ValueError(f"Choose 1–{cfg['max_upload_files']} files; use batch ingestion for more")
    if sum(upload.size for upload in uploads) > cfg["max_upload_total_mb"] * 1024**2:
        raise ValueError("Upload batch exceeds the configured total size limit")
    for upload in uploads:
        if Path(upload.name).suffix.lower() not in SUPPORTED:
            raise ValueError("Upload has an unsupported extension")
        if upload.size < 0 or upload.size > options.max_bytes:
            raise ValueError("Upload exceeds the configured file size limit")
    raw_root = options.settings.paths["DATA_DIR"] / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    if raw_root.is_symlink():
        raise ValueError("Raw upload directory must not be a symlink")
    destination = raw_root / "uploads" / uuid4().hex
    if not destination.resolve().is_relative_to(raw_root.resolve()):
        raise ValueError("Upload directory escapes raw storage")
    destination.mkdir(parents=True, exist_ok=False)
    for upload in uploads:
        # Keep a sanitized basename as provenance, but never use it as an identifier.
        name = upload.name.replace("\\", "/").split("/")[-1]
        name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in name)[-120:]
        path = destination / f"{uuid4().hex}__{name}"
        upload.seek(0)
        total = 0
        with path.open("xb") as sink:
            while chunk := upload.read(64 * 1024):
                total += len(chunk)
                if total > options.max_bytes or total > upload.size:
                    raise ValueError("Upload length exceeded declared or configured size")
                sink.write(chunk)
        upload.seek(0)
    return destination
