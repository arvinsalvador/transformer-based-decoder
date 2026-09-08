"""Constant-size running counters; no retained corpus or per-file collection."""

from dataclasses import dataclass, field

from src.data.models import DocumentRecord, Status

FAILED = {Status.PARSE_ERROR, Status.ENCODING_ERROR, Status.READ_ERROR, Status.PASSWORD_PROTECTED}


@dataclass
class IngestionStatistics:
    """Character statistics cover all emitted records, including rejected candidates."""

    files_examined: int = 0
    records_written: int = 0
    successful: int = 0
    skipped: int = 0
    failed: int = 0
    characters_extracted: int = 0
    source_bytes_examined: int = 0
    min_characters: int | None = None
    max_characters: int = 0
    file_types: dict[str, int] = field(default_factory=dict)
    document_types: dict[str, int] = field(default_factory=dict)
    statuses: dict[str, int] = field(default_factory=dict)

    def examine_file(self, source_type: str, size: int) -> None:
        """Count each source's disk size once, even for a many-row CSV."""
        self.files_examined += 1
        self.source_bytes_examined += size
        self.file_types[source_type] = self.file_types.get(source_type, 0) + 1

    def add(self, record: DocumentRecord) -> None:
        """Update aggregates without retaining the record or its text."""
        self.records_written += 1
        self.successful += record.extraction_status == Status.SUCCESS
        self.failed += record.extraction_status in FAILED
        self.skipped += (
            record.extraction_status != Status.SUCCESS and record.extraction_status not in FAILED
        )
        self.characters_extracted += record.character_count
        self.min_characters = (
            record.character_count
            if self.min_characters is None
            else min(self.min_characters, record.character_count)
        )
        self.max_characters = max(self.max_characters, record.character_count)
        self.document_types[record.source_type] = self.document_types.get(record.source_type, 0) + 1
        status = record.extraction_status.value
        self.statuses[status] = self.statuses.get(status, 0) + 1

    def snapshot(self) -> dict:
        """Return text-free progress/manifest data with explicit count semantics."""
        return {
            "files_examined": self.files_examined,
            "total_documents": self.records_written,
            "records_written": self.records_written,
            "documents_created": self.successful,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "no_extractable_text": self.statuses.get(Status.NO_EXTRACTABLE_TEXT.value, 0),
            "unsupported": self.statuses.get(Status.UNSUPPORTED_TYPE.value, 0),
            "characters_extracted": self.characters_extracted,
            "source_bytes_examined": self.source_bytes_examined,
            "min_characters": self.min_characters or 0,
            "max_characters": self.max_characters,
            "average_characters": (
                self.characters_extracted / self.records_written if self.records_written else 0
            ),
            "file_types": dict(self.file_types),
            "document_types": dict(self.document_types),
            "statuses": dict(self.statuses),
        }
