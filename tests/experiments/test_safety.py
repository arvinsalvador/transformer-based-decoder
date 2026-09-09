import copy
from pathlib import Path

import pytest

from src.evaluation.identity import read_json
from src.experiments import preflight as checks
from src.experiments import runner
from src.experiments.registry import save
from src.training import checkpoint
from tests.experiments.test_experiments import run


def test_unwritable_path(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("not writable")

    monkeypatch.setattr(checks, "NamedTemporaryFile", denied)
    with pytest.raises(PermissionError):
        checks.storage_checks([tmp_path], 0.000001)


def test_missing_tokenizer_rejected(experiment_case):
    token = experiment_case[0].paths["MODEL_DIR"] / "tokenizer/tokenizer.json"
    token.rename(token.with_name("fixture_missing.json"))
    with pytest.raises(OSError):
        run(experiment_case, mode="preflight")


def test_wrong_split_hash_rejected(experiment_case):
    path = experiment_case[0].paths["DATA_DIR"] / "splits/validation.jsonl"
    path.write_text('{"text":"changed fixture"}\n')
    with pytest.raises(ValueError, match="fingerprints"):
        run(experiment_case, mode="preflight")


def test_atomic_status_old_preserved(experiment_case, monkeypatch):
    status = run(experiment_case, mode="preflight")
    path = Path(status["run_directory"]) / "status.json"
    before = path.read_bytes()
    real = checkpoint.os.replace

    def fail(source, destination):
        if Path(destination) == path:
            raise OSError("injected publication failure")
        real(source, destination)

    monkeypatch.setattr(checkpoint.os, "replace", fail)
    candidate = copy.deepcopy(status)
    candidate["status"] = "RUNNING"
    with pytest.raises(OSError):
        save(path.parent, candidate)
    assert path.read_bytes() == before
    assert read_json(path)["status"] == "READY"


def test_no_quality_gate(experiment_case, monkeypatch):
    original = runner.evaluate_models

    def worse(*args, **kwargs):
        result = original(*args, **kwargs)
        data = read_json(result["comparison_path"])
        # This synthetic test deliberately makes the Transformer worse.
        data["results"]["transformer"]["perplexity"] = data["results"]["trigram"]["perplexity"] * 2
        from src.evaluation.artifacts import write_json

        write_json(Path(result["comparison_path"]), data)
        return result

    monkeypatch.setattr(runner, "evaluate_models", worse)
    result = run(experiment_case, mode="execute", selected=[3])
    assert result["scales"][0]["status"] == "COMPLETED"
