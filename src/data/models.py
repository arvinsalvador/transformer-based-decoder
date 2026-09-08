"""Typed ingestion records and bounded execution options."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.config.settings import Settings, validate_config


def utc_now() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


class Status(StrEnum):
    """Machine-readable extraction outcomes; never infer these from exception text."""

    SUCCESS = "SUCCESS"
    UNSUPPORTED_TYPE = "UNSUPPORTED_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    EMPTY_FILE = "EMPTY_FILE"
    NO_EXTRACTABLE_TEXT = "NO_EXTRACTABLE_TEXT"
    TOO_SHORT = "TOO_SHORT"
    PARSE_ERROR = "PARSE_ERROR"
    ENCODING_ERROR = "ENCODING_ERROR"
    PASSWORD_PROTECTED = "PASSWORD_PROTECTED"
    UNSAFE_PATH = "UNSAFE_PATH"
    READ_ERROR = "READ_ERROR"
    EXTRACTION_TOO_LARGE = "EXTRACTION_TOO_LARGE"


class ExtractionError(Exception):
    """Controlled parser failure with a safe reason that contains no document text."""

    def __init__(self, status: Status, reason: str):
        super().__init__(reason)
        self.status = status


@dataclass
class ParsedText:
    """One candidate logical document, or a controlled parser event."""

    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    status: Status = Status.SUCCESS
    error_message: str | None = None


@dataclass
class DocumentRecord:
    """One JSONL record; source size is repeated per row, never summed per row."""

    document_id: str
    source_name: str
    source_path: str
    source_type: str
    file_size_bytes: int
    extracted_text: str
    character_count: int
    extraction_status: Status
    error_message: str | None
    ingested_at: str
    sha256: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class IngestionOptions:
    """Options are constructed from fully validated project configuration."""

    settings: Settings
    limit: int
    recursive: bool
    csv_mode: str
    text_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_config(self.settings.values)
        cap = min(
            self.settings.values["dataset"]["max_documents"],
            self.settings.values["dataset"]["working_document_limit"],
            100000,
        )
        if type(self.limit) is not int or not 0 < self.limit <= cap:
            raise ValueError(f"limit must be an integer between 1 and the effective cap {cap}")
        if type(self.recursive) is not bool or self.csv_mode not in ("rows", "file"):
            raise ValueError("recursive must be boolean and csv_mode must be rows or file")
        if any(not isinstance(c, str) or not c.strip() for c in self.text_columns) or len(
            self.text_columns
        ) != len(set(self.text_columns)):
            raise ValueError("text_columns must contain unique nonempty names")

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        limit: int | None = None,
        recursive: bool | None = None,
        csv_mode: str | None = None,
        text_columns: list[str] | None = None,
    ) -> "IngestionOptions":
        """CLI/UI overrides can lower the working cap but never increase it."""
        cfg = settings.values
        return cls(
            settings,
            cfg["dataset"]["working_document_limit"] if limit is None else limit,
            cfg["dataset"]["recursive"] if recursive is None else recursive,
            csv_mode or cfg["csv"]["mode"],
            tuple(cfg["csv"]["text_columns"] if text_columns is None else text_columns),
        )

    @property
    def max_bytes(self) -> int:
        return self.settings.values["dataset"]["max_file_size_mb"] * 1024**2

    @property
    def max_characters(self) -> int:
        return self.settings.values["ingestion"]["max_extracted_characters"]

    @property
    def output_path(self) -> Path:
        return self.settings.paths["DATA_DIR"] / self.settings.values["ingestion"]["output_path"]

    @property
    def manifest_dir(self) -> Path:
        return (
            self.settings.paths["EXPERIMENT_DIR"]
            / self.settings.values["ingestion"]["manifest_dir"]
        )
