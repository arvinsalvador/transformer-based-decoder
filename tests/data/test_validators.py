"""Safety checks exercised through the same API as CLI/uploads."""

from dataclasses import replace

import pytest

from src.config.settings import ConfigurationError, validate_config
from src.data.ingestion import _parse_records, ingest_directory
from src.data.models import Status


@pytest.mark.parametrize("extension", ["txt", "csv", "pdf", "docx"])
def test_oversize(corpus, options, extension):
    options.settings.values["dataset"]["max_file_size_mb"] = 1
    path = corpus / f"large.{extension}"
    with path.open("wb") as stream:
        stream.truncate(1024**2 + 1)
    assert list(_parse_records(path, corpus, options))[0].status == Status.FILE_TOO_LARGE


def test_unsupported(corpus, options):
    path = corpus / "ignore.exe"
    path.write_text("not executable")
    assert list(_parse_records(path, corpus, options))[0].status == Status.UNSUPPORTED_TYPE


def test_symlink_and_traversal(corpus, tmp_path, options):
    outside = tmp_path / "outside.txt"
    outside.write_text("Private")
    link = corpus / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks not permitted on this platform")
    assert list(_parse_records(link, corpus, options))[0].status == Status.UNSAFE_PATH
    assert list(_parse_records(outside, corpus, options))[0].status == Status.UNSAFE_PATH
    result = ingest_directory(corpus, options)
    assert result.summary["successful"] == 0


@pytest.mark.parametrize("limit", [0, -1, 100001, 1001, True])
def test_limit_cannot_bypass_profile(options, limit):
    with pytest.raises(ValueError, match="limit"):
        replace(options, limit=limit)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("dataset", "max_documents", 100001),
        ("dataset", "max_file_size_mb", 0),
        ("dataset", "recursive", "yes"),
        ("csv", "mode", "other"),
        ("csv", "text_columns", ["body", "body"]),
        ("csv", "text_columns", "text"),
        ("ingestion", "preview_characters", 2001),
        ("ingestion", "preview_documents", 101),
        ("ingestion", "output_path", "../outside"),
        ("ingestion", "output_format", "json"),
    ],
)
def test_config_extensions(options, section, field, value):
    options.settings.values[section][field] = value
    with pytest.raises(ConfigurationError):
        validate_config(options.settings.values)
