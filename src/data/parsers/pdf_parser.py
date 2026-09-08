"""PDF text extraction, page by page, with no OCR or password guessing."""

from collections.abc import Iterator
from pathlib import Path

import pymupdf

from src.data.models import ExtractionError, IngestionOptions, ParsedText, Status
from src.data.validators import bounded_join


def parse(path: Path, options: IngestionOptions) -> Iterator[ParsedText]:
    """Extract one bounded text document from a PDF and record page count."""
    with pymupdf.open(path) as document:
        if not document.is_pdf:
            raise ExtractionError(Status.PARSE_ERROR, "Source is not a PDF")
        if document.needs_pass:
            raise ExtractionError(Status.PASSWORD_PROTECTED, "Password-protected PDF")
        text = bounded_join((page.get_text("text") for page in document), options.max_characters)
        yield ParsedText(text, {"page_count": document.page_count})
