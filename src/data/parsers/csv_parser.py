"""Streaming RFC-style comma-separated data. Selected-column order is preserved."""

import csv
from collections.abc import Iterator
from pathlib import Path

from src.data.models import ExtractionError, IngestionOptions, ParsedText, Status
from src.data.validators import bounded_join, check_text


def read_header(path: Path) -> list[str]:
    """Read only the header for UI selection; errors are controlled and text is not echoed."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream, strict=True), [])
        if not header or any(not c.strip() for c in header) or len(header) != len(set(header)):
            raise ExtractionError(Status.PARSE_ERROR, "CSV requires nonempty unique column names")
        for column in header:
            check_text(column)
        return header
    except UnicodeError as exc:
        raise ExtractionError(Status.ENCODING_ERROR, "CSV must use UTF-8 (BOM accepted)") from exc
    except (OSError, csv.Error) as exc:
        raise ExtractionError(Status.PARSE_ERROR, "Cannot read CSV header") from exc


def _rows(path: Path, options: IngestionOptions) -> Iterator[ParsedText]:
    # csv.reader streams physical lines and handles quoted commas/multiline fields.
    # The standard 128 KiB per-field ceiling remains in place (no global state changes).
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if (
            not header
            or any(not name.strip() for name in header)
            or len(set(header)) != len(header)
        ):
            raise ExtractionError(Status.PARSE_ERROR, "CSV requires nonempty unique column names")
        if not options.text_columns:
            raise ExtractionError(Status.PARSE_ERROR, "Select CSV text_columns explicitly")
        if any(column not in header for column in options.text_columns):
            raise ExtractionError(Status.PARSE_ERROR, "Selected CSV text column is missing")
        indices = [header.index(column) for column in options.text_columns]
        for row_index, row in enumerate(reader, start=1):
            metadata = {
                "row_index": row_index,
                "physical_line": reader.line_num,
                "text_columns": list(options.text_columns),
                "encoding": "utf-8-sig",
            }
            if len(row) != len(header):
                yield ParsedText(
                    metadata=metadata,
                    status=Status.PARSE_ERROR,
                    error_message="CSV row width differs from header",
                )
                continue
            text = bounded_join((row[index] for index in indices), options.max_characters)
            check_text(text)
            yield ParsedText(text, metadata)


def parse(path: Path, options: IngestionOptions) -> Iterator[ParsedText]:
    """Yield rows lazily or a single bounded file document; never materialize a row list."""
    try:
        rows = _rows(path, options)
        try:
            if options.csv_mode == "rows":
                found = False
                for row in rows:
                    found = True
                    yield row
                if not found:
                    yield ParsedText(metadata={"row_count": 0})
            else:
                count = 0

                def pieces():
                    nonlocal count
                    for row in rows:
                        count += 1
                        if row.status != Status.SUCCESS:
                            raise ExtractionError(row.status, row.error_message)
                        yield row.text

                text = bounded_join(pieces(), options.max_characters)
                yield ParsedText(
                    text,
                    {
                        "row_count": count,
                        "text_columns": list(options.text_columns),
                        "encoding": "utf-8-sig",
                    },
                )
        finally:
            rows.close()
    except UnicodeError as exc:
        raise ExtractionError(Status.ENCODING_ERROR, "CSV must use UTF-8 (BOM accepted)") from exc
    except csv.Error as exc:
        raise ExtractionError(Status.PARSE_ERROR, "Malformed CSV or field exceeds 128 KiB") from exc
