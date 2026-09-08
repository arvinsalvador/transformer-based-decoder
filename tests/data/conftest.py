"""Isolated runtime roots keep all generated documents outside the repository."""

from copy import deepcopy
from dataclasses import replace

import pytest

from src.config.settings import load_settings
from src.data.models import IngestionOptions


@pytest.fixture
def options(tmp_path, monkeypatch):
    for key in ("DEVICE", "WORKING_DOCUMENT_LIMIT", "BATCH_SIZE"):
        monkeypatch.delenv(key, raising=False)
    settings = load_settings("config/local.yaml")
    values = deepcopy(settings.values)
    values["dataset"]["min_extracted_characters"] = 1
    values["ingestion"]["progress_interval"] = 1
    paths = {key: tmp_path / key.lower() for key in settings.paths}
    return IngestionOptions.from_settings(replace(settings, values=values, paths=paths))


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    return root
