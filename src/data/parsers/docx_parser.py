"""Paragraph/table extraction with a ZIP expansion guard before XML parsing."""

from collections.abc import Iterator
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from src.data.models import ExtractionError, IngestionOptions, ParsedText, Status
from src.data.validators import bounded_join


def parse(path: Path, options: IngestionOptions) -> Iterator[ParsedText]:
    """Extract body paragraphs and tables in document order; no styling conversion."""
    with ZipFile(path) as archive:
        expanded = sum(item.file_size for item in archive.infolist())
        maximum = options.settings.values["ingestion"]["max_docx_uncompressed_mb"] * 1024**2
        if expanded > maximum:
            raise ExtractionError(Status.EXTRACTION_TOO_LARGE, "DOCX expanded size exceeds limit")
        if any(item.flag_bits & 1 for item in archive.infolist()):
            raise ExtractionError(Status.PASSWORD_PROTECTED, "Encrypted DOCX archive")
    document = Document(path)
    metadata = {"paragraph_count": 0, "table_count": 0}

    def blocks(container):
        for block in container.iter_inner_content():
            if isinstance(block, Paragraph):
                metadata["paragraph_count"] += 1
                yield block.text
            elif isinstance(block, Table):
                metadata["table_count"] += 1
                for row in block.rows:
                    # Skip repeated references from horizontally merged cells.
                    seen = set()
                    for cell in row.cells:
                        if cell._tc not in seen:
                            seen.add(cell._tc)
                            yield from blocks(cell)

    text = bounded_join(blocks(document), options.max_characters)
    yield ParsedText(text, metadata)
