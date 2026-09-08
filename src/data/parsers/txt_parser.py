"""Strict UTF-8/BOM decoding with an explicit Windows-1252 fallback."""

from collections.abc import Iterator
from pathlib import Path

from src.data.models import ExtractionError, IngestionOptions, ParsedText, Status
from src.data.validators import check_text


def decode_text(raw: bytes) -> tuple[str, str, bool]:
    """Decode without replacement or ignored bytes; record any fallback."""
    try:
        return (
            raw.decode("utf-8-sig"),
            "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8",
            False,
        )
    except UnicodeDecodeError:
        try:
            return raw.decode("cp1252"), "cp1252", True
        except UnicodeDecodeError as exc:
            raise ExtractionError(
                Status.ENCODING_ERROR, "Cannot decode as UTF-8 or Windows-1252"
            ) from exc


def parse(path: Path, options: IngestionOptions) -> Iterator[ParsedText]:
    """Read only one size-checked TXT; preserve whitespace and capitalization."""
    with path.open("rb") as source:
        raw = source.read(options.max_bytes + 1)
    if len(raw) > options.max_bytes:
        raise ExtractionError(Status.FILE_TOO_LARGE, "File grew beyond configured size limit")
    text, encoding, fallback = decode_text(raw)
    check_text(text)
    if len(text) > options.max_characters:
        raise ExtractionError(Status.EXTRACTION_TOO_LARGE, "Extracted text exceeds limit")
    yield ParsedText(text, {"encoding": encoding, "encoding_fallback": fallback})
