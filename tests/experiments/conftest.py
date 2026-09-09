import json

import pytest

from src.config.settings import PROJECT_ROOT
from src.experiments.plan import load_plan
from src.training.reproducibility import file_hash
from tests.training import conftest as training_fixtures

training_case = training_fixtures.training_case


@pytest.fixture
def experiment_case(training_case):
    settings, train, validation, tokenizer = training_case
    settings.values["training"]["max_steps"] = 2
    settings.values["evaluation"]["generation"]["max_new_tokens"] = 2
    test = train.parent / "test.jsonl"
    test.write_text('{"text":"alpha beta"}\n')
    manifest = settings.paths["EXPERIMENT_DIR"] / "preprocessing/test.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "state": "completed",
                "dataset_fingerprint": "fixture",
                "split_fingerprints": {
                    name: file_hash(train.parent / f"{name}.jsonl")
                    for name in ("train", "validation", "test")
                },
            }
        )
    )
    plan = load_plan(PROJECT_ROOT / "config/experiments.yaml")
    plan.update(scales=[3, 5, 20, "full"], minimum_free_disk_gb=0.000001)
    return settings, plan, manifest
