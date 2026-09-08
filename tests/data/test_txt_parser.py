"""Encoding preservation and explicit rejection of undecodable/binary content."""

import pytest

from src.data.ingestion import _parse_records
from src.data.models import Status


@pytest.mark.parametrize(
    ("raw", "text", "encoding"),
    [
        ("Hello 世界\n Keep  Spaces".encode(), "Hello 世界\n Keep  Spaces", "utf-8"),
        (b"\xef\xbb\xbfHello", "Hello", "utf-8-sig"),
        (b"caf\xe9", "café", "cp1252"),
    ],
)
def test_decoding(corpus, options, raw, text, encoding):
    path = corpus / "sample.txt"
    path.write_bytes(raw)
    result = list(_parse_records(path, corpus, options))[0]
    assert result.text == text
    assert result.metadata["encoding"] == encoding
    assert result.metadata["encoding_fallback"] == (encoding == "cp1252")


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        (b"", Status.EMPTY_FILE),
        (b"abc\x00def", Status.ENCODING_ERROR),
        (b"\x81\x8d", Status.ENCODING_ERROR),
    ],
)
def test_bad_text(corpus, options, raw, status):
    path = corpus / "bad.txt"
    path.write_bytes(raw)
    assert list(_parse_records(path, corpus, options))[0].status == status
