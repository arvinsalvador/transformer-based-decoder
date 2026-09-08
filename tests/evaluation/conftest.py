import json

import pytest

from src.evaluation.identity import read_json
from src.training.reproducibility import file_hash
from src.training.trainer import run_training
from src.trigram.trainer import train
from tests.training import conftest as training_fixtures

training_case = training_fixtures.training_case


@pytest.fixture
def evaluation_case(training_case):
    settings, source, validation, token_dir = training_case
    settings.values["evaluation"]["generation"]["max_new_tokens"] = 3
    settings.values["evaluation"]["document_limit"] = None
    # Temporary synthetic fixtures only; the evaluator itself never imports training.
    train(source, token_dir, settings)
    run_training(settings, source, validation, token_dir, max_steps=2)
    test = source.parent / "test.jsonl"
    test.write_text(
        "\n".join(json.dumps({"text": text}) for text in ("alpha beta gamma", "alpha " * 20, ""))
        + "\n"
    )
    manifest = settings.paths["EXPERIMENT_DIR"] / "dataset.json"
    manifest.write_text(
        json.dumps(
            {
                "state": "completed",
                "dataset_fingerprint": "synthetic",
                "split_fingerprints": {
                    name: file_hash(source.parent / f"{name}.jsonl")
                    for name in ("train", "validation", "test")
                },
            }
        )
    )
    # Use declared provenance consistently on synthetic manifests.
    for path in (
        token_dir / "tokenizer_manifest.json",
        settings.paths["MODEL_DIR"] / "trigram/trigram_manifest.json",
        settings.paths["MODEL_DIR"] / "transformer/model_manifest.json",
    ):
        data = read_json(path)
        data.update(dataset_fingerprint="synthetic", dataset_fingerprint_source="declared_corpus")
        path.write_text(json.dumps(data))
    return settings, test, token_dir, manifest
