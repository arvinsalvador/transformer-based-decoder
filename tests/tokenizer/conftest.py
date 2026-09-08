"""Small tokenizer settings bypass production vocabulary floor only in isolated tests."""

from copy import deepcopy
from dataclasses import replace

import pytest

from src.config.settings import load_settings


@pytest.fixture
def tokenizer_settings(tmp_path):
    settings = load_settings("config/local.yaml")
    values = deepcopy(settings.values)
    values["tokenizer"]["vocab_size"] = 80
    values["tokenizer"]["min_frequency"] = 1
    paths = {key: tmp_path / key.lower() for key in settings.paths}
    return replace(settings, values=values, paths=paths)


@pytest.fixture
def splits(tmp_path):
    train, validation, test = (
        tmp_path / "train.jsonl",
        tmp_path / "validation.jsonl",
        tmp_path / "test.jsonl",
    )
    rows = [
        "Artificial intelligence improves software systems.",
        "Machine learning identifies patterns in data.",
        "Cybersecurity protects computer networks.",
        "Transformer models process token sequences.",
        "Decoder models predict the next token.",
        "Software vulnerabilities can affect systems.",
    ]
    import json

    for path, texts in (
        (train, rows),
        (validation, ["Cybersecurity CVE-2026-1234."]),
        (test, ["NovelHyperTerm protects APIs."]),
    ):
        path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts))
    return train, validation, test
