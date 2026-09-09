import copy
from pathlib import Path

import pytest
import torch

from src.evaluation.identity import read_json
from src.experiments import runner
from src.experiments.plan import load_plan
from src.experiments.preflight import audit_data, enforce_cap, storage_checks
from src.experiments.runner import run_experiment
from src.experiments.subsets import prepare_subset, ranked_locators, selected_lines, verify_subset
from src.training import trainer
from src.training.reproducibility import file_hash


def run(case, **kwargs):
    settings, plan, manifest = case
    return run_experiment(settings, plan, dataset_manifest=manifest, **kwargs)


def test_plan_preflight_no_training(experiment_case, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Training forbidden")

    monkeypatch.setattr(runner, "train", forbidden)
    monkeypatch.setattr(runner, "run_training", forbidden)
    result = run(experiment_case)
    assert result["status"] == "PLANNED" and not result["training_launched"]
    assert result["audit"]["total_corpus_documents"] == 14
    assert result["audit"]["scales"][2]["status"] == "NOT_APPLICABLE"
    result = run(experiment_case, mode="preflight")
    assert result["status"] == "READY", result


def test_nested_exact_subsets(experiment_case):
    settings, plan, manifest = experiment_case
    audit = audit_data(settings, plan, manifest)
    train = settings.paths["DATA_DIR"] / "splits/train.jsonl"
    first = list(selected_lines(train, 42, 3))
    assert first == list(selected_lines(train, 42, 5))[:3]
    assert first == list(selected_lines(train, 42, 3))
    assert ranked_locators(train, 42) != ranked_locators(train, 43)
    for scale in (audit["scales"][0], audit["scales"][-1]):
        path, provenance, result = prepare_subset(
            settings,
            settings.paths["EXPERIMENT_DIR"] / scale["scale"],
            scale,
            42,
            audit["split_fingerprints"]["train"],
        )
        assert len(path.read_text().splitlines()) == scale["actual_train_documents"]
        assert verify_subset(settings, path, provenance) == result


def test_cap():
    enforce_cap({"train": 80000, "validation": 10000, "test": 10000})
    with pytest.raises(ValueError, match="TOTAL"):
        enforce_cap({"train": 80001, "validation": 10000, "test": 10000})


def test_confirmation_no_training(experiment_case, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No training without confirmation")

    monkeypatch.setattr(runner, "run_scale", forbidden)
    experiment_case[1]["large_run_threshold_documents"] = 3
    result = run(experiment_case, mode="execute", selected=[3])
    assert result["status"] == "READY" and "confirm" in result["message"]
    result = run(experiment_case, mode="execute", selected=["full"])
    assert "confirm" in result["message"]


def test_tiny_full_pipeline_and_idempotence(experiment_case, monkeypatch):
    settings = experiment_case[0]
    protected = [p for p in settings.paths["MODEL_DIR"].rglob("*") if p.is_file()]
    hashes = {path: file_hash(path) for path in protected}
    result = run(experiment_case, mode="execute", confirm_large_run=True)
    assert result["status"] == "COMPLETED", result
    directory = Path(result["run_directory"])
    summary = read_json(directory / "summary.json")
    assert summary["primary_final_result"]
    assert summary["full_result"]["scope"] == "FULL TEST SET"
    assert (directory / "experiment_matrix.csv").is_file()
    for scale in result["scales"]:
        if scale["status"] == "NOT_APPLICABLE":
            continue
        tri = scale["stages"]["trigram"]["result"]
        comparison = read_json(scale["comparison_path"])
        tf = comparison["results"]["transformer"]
        assert tri["subset_fingerprint"] == tf["subset_fingerprint"] == scale["subset_fingerprint"]
        assert tf["test_fingerprint"] == result["audit"]["split_fingerprints"]["test"]
        assert tf["training_run_id"]
    ids = [
        s["stages"]["transformer"]["result"]["run_id"]
        for s in result["scales"]
        if s["status"] == "COMPLETED"
    ]
    assert len(set(ids)) == 3

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed scale was rerun")

    monkeypatch.setattr(runner, "run_scale", forbidden)
    again = run(experiment_case, mode="execute", resume=result["experiment_id"])
    assert again["status"] == "COMPLETED"
    assert hashes == {path: file_hash(path) for path in protected}


@pytest.mark.parametrize("failure", ["FAILED_OOM", "FAILED_NONFINITE"])
def test_failure_stops_larger_scales(experiment_case, monkeypatch, failure):
    calls = []
    original = runner.run_training

    def fail(*args, **kwargs):
        calls.append(kwargs.get("dry_run"))
        if kwargs.get("dry_run"):
            return original(*args, **kwargs)
        raise trainer.TrainingFailure({"status": failure, "message": "synthetic failure"})

    monkeypatch.setattr(runner, "run_training", fail)
    result = run(experiment_case, mode="execute", confirm_large_run=True)
    assert result["status"] == "FAILED"
    assert result["scales"][0]["status"] == failure
    assert result["scales"][1]["status"] == "PENDING"
    assert calls == [True, False]


def test_evaluation_only_resume(experiment_case, monkeypatch):
    original = runner.evaluate_models
    monkeypatch.setattr(
        runner,
        "evaluate_models",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("evaluation failure")),
    )
    first = run(experiment_case, mode="execute", selected=[3])
    assert first["status"] == "FAILED"
    monkeypatch.setattr(runner, "evaluate_models", original)

    def forbidden(*args, **kwargs):
        raise AssertionError("Training should be skipped")

    monkeypatch.setattr(runner, "train", forbidden)
    monkeypatch.setattr(runner, "run_training", forbidden)
    result = run(experiment_case, mode="execute", selected=[3], resume=first["experiment_id"])
    assert result["scales"][0]["status"] == "COMPLETED", result


def test_real_interruption_resume(experiment_case, monkeypatch):
    original = runner.run_training
    calls = []

    def interrupt(*args, **kwargs):
        calls.append(kwargs.get("resume"))
        if kwargs.get("dry_run"):
            return original(*args, **kwargs)
        count = 0
        real_loss = trainer.loss_sum

        def loss(*a, **k):
            nonlocal count
            count += 1
            if count == 2:
                raise KeyboardInterrupt()
            return real_loss(*a, **k)

        with monkeypatch.context() as patch:
            patch.setattr(trainer, "loss_sum", loss)
            return original(*args, **kwargs)

    monkeypatch.setattr(runner, "run_training", interrupt)
    first = run(experiment_case, mode="execute", selected=[5])
    assert first["status"] == "INTERRUPTED", first

    def resume(*args, **kwargs):
        assert kwargs["resume"].name == "latest.pt"
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "run_training", resume)
    second = run(experiment_case, mode="execute", selected=[5], resume=first["experiment_id"])
    assert second["scales"][1]["status"] == "COMPLETED", second


def test_changed_plan_rejected(experiment_case):
    first = run(experiment_case, mode="preflight")
    experiment_case[0].values["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="fingerprint"):
        run(experiment_case, mode="execute", selected=[3], resume=first["experiment_id"])


def test_cuda_missing_preflight(experiment_case, monkeypatch):
    experiment_case[0].values["device"] = "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = run(experiment_case, mode="preflight")
    assert result["status"] == "PREFLIGHT_FAILED" and "CUDA" in result["message"]


def test_disk_failure(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda path: type("Disk", (), {"free": 0})())
    with pytest.raises(ValueError, match="Insufficient disk"):
        storage_checks([tmp_path], 1)


def test_plan_validation(tmp_path):
    import yaml

    from src.config.settings import PROJECT_ROOT

    plan = load_plan(PROJECT_ROOT / "config/experiments.yaml")
    for scales in ([100000], [5, 3], [3, 3], ["full", 3], [True]):
        candidate = copy.deepcopy(plan)
        candidate["scales"] = scales
        path = tmp_path / "plan.yaml"
        path.write_text(yaml.safe_dump({"experiment": candidate}))
        with pytest.raises(ValueError):
            load_plan(path)
