import json
from copy import deepcopy
from dataclasses import replace

import pytest

from src.config.settings import load_settings
from src.tokenizer.service import train_wordpiece


@pytest.fixture
def training_case(tmp_path):
    settings = load_settings("config/local.yaml")
    values = deepcopy(settings.values)
    values["device"] = "cpu"
    values["tokenizer"].update(vocab_size=1000, min_frequency=1)
    values["model"].update(
        context_length=8,
        embedding_dim=16,
        num_heads=2,
        num_layers=1,
        feedforward_dim=32,
        dropout=0.0,
    )
    values["training"].update(
        batch_size=2,
        epochs=4,
        num_workers=0,
        persistent_workers=False,
        sequence_stride=8,
        shuffle_buffer_size=2,
        learning_rate=0.01,
        warmup_steps=0,
        logging_steps=1,
        deterministic=True,
    )
    values["training"]["checkpoint"].update(save_every_steps=1, keep_last_n=2)
    values["training"]["early_stopping"]["enabled"] = False
    paths = {key: tmp_path / key.lower() for key in settings.paths}
    settings = replace(settings, values=values, paths=paths)
    root = paths["DATA_DIR"] / "splits"
    root.mkdir(parents=True)
    train, validation = root / "train.jsonl", root / "validation.jsonl"
    train.write_text(
        "".join(
            json.dumps({"text": text}) + "\n"
            for text in ["alpha beta gamma", "beta gamma", "gamma alpha"] * 4
        )
    )
    validation.write_text(json.dumps({"text": "alpha beta gamma"}) + "\n")
    # Deliberately invalid: any attempt to parse test data should fail.
    (root / "test.jsonl").write_text("{DO NOT READ TEST}")
    token_dir = train_wordpiece(train, settings).output_dir
    return settings, train, validation, token_dir
