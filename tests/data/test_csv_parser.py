"""CSV logical rows, selection order, malformed rows and early stop."""

from dataclasses import replace

import pytest

from src.data.ingestion import _parse_records, ingest_directory
from src.data.models import Status


def test_rows_and_selection(corpus, options):
    (corpus / "rows.csv").write_text(
        '\ufefftitle,body,n\n"A, B","Line 1\nLine 2",42\nC,D,7\n', encoding="utf-8"
    )
    options = replace(options, text_columns=("n", "title", "body"))
    results = list(_parse_records(corpus / "rows.csv", corpus, options))
    assert [r.text for r in results] == ["42\nA, B\nLine 1\nLine 2", "7\nC\nD"]
    assert results[0].metadata["row_index"] == 1


def test_file_mode(corpus, options):
    path = corpus / "rows.csv"
    path.write_text("body\nfirst\nsecond\n")
    result = list(
        _parse_records(path, corpus, replace(options, csv_mode="file", text_columns=("body",)))
    )
    assert len(result) == 1
    assert result[0].text == "first\nsecond"
    assert result[0].metadata["row_count"] == 2


@pytest.mark.parametrize(
    ("text", "columns", "status"),
    [
        ("body\nhi\n", ("missing",), Status.PARSE_ERROR),
        ("body\nhi\n", (), Status.PARSE_ERROR),
        ("body,body\na,b\n", ("body",), Status.PARSE_ERROR),
        ('body\n"unfinished', ("body",), Status.PARSE_ERROR),
        ("body\n" + "x" * 140000, ("body",), Status.PARSE_ERROR),
        ("", ("body",), Status.EMPTY_FILE),
    ],
)
def test_csv_failures(corpus, options, text, columns, status):
    path = corpus / "bad.csv"
    path.write_text(text)
    records = list(_parse_records(path, corpus, replace(options, text_columns=columns)))
    assert records[-1].status == status


def test_empty_numeric_and_malformed_rows(corpus, options):
    (corpus / "rows.csv").write_text("id,body\n1,\n2,123\n3,x,extra\n4,good\n")
    result = ingest_directory(corpus, replace(options, text_columns=("body",)))
    assert result.summary["records_written"] == 4
    assert result.summary["successful"] == 2
    assert result.summary["failed"] == 1
    assert result.summary["no_extractable_text"] == 1


def test_stop_before_malformed_tail(corpus, options):
    (corpus / "rows.csv").write_text('body\none\ntwo\n"unterminated')
    result = ingest_directory(corpus, replace(options, limit=2, text_columns=("body",)))
    assert result.summary["successful"] == 2
    assert result.summary["failed"] == 0
    assert result.summary["limit_reached"]


def test_header_only_and_bad_encoding(corpus, options):
    options = replace(options, text_columns=("body",))
    path = corpus / "empty.csv"
    path.write_text("body\n")
    result = ingest_directory(corpus, options)
    assert result.summary["no_extractable_text"] == 1
    path.write_bytes(b"body\n\x81\n")
    assert list(_parse_records(path, corpus, options))[-1].status == Status.ENCODING_ERROR
