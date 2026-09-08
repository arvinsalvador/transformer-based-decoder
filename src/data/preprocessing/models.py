"""Typed preprocessing outcomes and clean-document representation."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PreprocessingStatus(StrEnum):
    """Controlled Phase 3 outcomes, separate from Phase 2 extraction statuses."""

    ACCEPTED = "ACCEPTED"
    EMPTY_TEXT = "EMPTY_TEXT"
    TOO_SHORT = "TOO_SHORT"
    TOO_LONG_SKIPPED = "TOO_LONG_SKIPPED"
    TRUNCATED = "TRUNCATED"
    LOW_ALPHA_RATIO = "LOW_ALPHA_RATIO"
    EXCESSIVE_CONTROL_CHARACTERS = "EXCESSIVE_CONTROL_CHARACTERS"
    EXCESSIVE_REPETITION = "EXCESSIVE_REPETITION"
    DUPLICATE = "DUPLICATE"
    INVALID_RECORD = "INVALID_RECORD"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    EXTRACTION_NOT_ACCEPTED = "EXTRACTION_NOT_ACCEPTED"


@dataclass(frozen=True)
class CleanDocument:
    """Accepted document persisted identically to clean JSONL and exactly one split."""

    document_id: str
    source_name: str
    source_type: str
    raw_sha256: str | None
    normalized_sha256: str
    text: str
    character_count: int
    original_character_count: int
    preprocessing_status: PreprocessingStatus
    truncated: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessingOutcome:
    """A non-persisted rejection/acceptance event with text deliberately omitted."""

    status: PreprocessingStatus
    reason: str | None = None
    duplicate_of: str | None = None
