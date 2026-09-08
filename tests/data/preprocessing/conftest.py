"""Prepared-dataset fixtures with safely isolated runtime paths."""

from copy import deepcopy
from dataclasses import replace

import pytest

from src.config.settings import load_settings


@pytest.fixture
def prep_settings(tmp_path):
    settings = load_settings("config/local.yaml")
    values = deepcopy(settings.values)
    values["preprocessing"]["min_characters"] = 1
    values["preprocessing"]["progress_interval"] = 1
    paths = {key: tmp_path / key.lower() for key in settings.paths}
    return replace(settings, values=values, paths=paths)


def raw_record(document_id: str, text: str, *, status="SUCCESS", source_type="txt") -> dict:
    """Minimal valid Phase 2 record shape."""
    return {
        "document_id": document_id,
        "source_name": f"{document_id}.txt",
        "source_type": source_type,
        "extracted_text": text,
        "extraction_status": status,
        "sha256": "raw-hash",
        "metadata": {"fixture": True},
    }
