"""Browser filenames cannot control the destination; bytes are preserved."""

import io

import pytest

from src.data.ingestion import ingest_directory
from src.data.uploads import stage_uploads


class FakeUpload(io.BytesIO):
    def __init__(self, name, content):
        super().__init__(content)
        self.name = name
        self.size = len(content)


def test_upload_roundtrip(options):
    uploads = [
        FakeUpload("../../outside.txt", b"Original text"),
        FakeUpload("C:\\folder\\outside.txt", b"Second text"),
    ]
    staged = stage_uploads(uploads, options)
    paths = list(staged.iterdir())
    assert len(paths) == 2
    assert all(path.is_relative_to(options.settings.paths["DATA_DIR"] / "raw") for path in paths)
    assert {p.read_bytes() for p in paths} == {b"Original text", b"Second text"}
    assert ingest_directory(staged, options).summary["successful"] == 2


def test_upload_limit(options):
    with pytest.raises(ValueError):
        stage_uploads([FakeUpload("a.txt", b"tiny")] * 21, options)
    upload = FakeUpload("a.txt", b"tiny")
    upload.size = 100 * 1024**2
    with pytest.raises(ValueError):
        stage_uploads([upload], options)
