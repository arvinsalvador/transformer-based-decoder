from pathlib import Path

import pytest
import torch

from src.training import trainer
from src.training.checkpoint import load_checkpoint
from src.training.trainer import TrainingFailure, run_training


def execute(case, **kwargs):
    return run_training(*case, **kwargs)


def test_dry_run_and_optimizer_run(training_case):
    settings, _, _, _ = training_case
    dry = execute(training_case, dry_run=True)
    assert dry["optimizer_steps"] == 0 and dry["forward"] == dry["backward"] == "PASS"
    assert not settings.paths["CHECKPOINT_DIR"].exists()
    assert not (settings.paths["MODEL_DIR"] / "transformer").exists()
    result = execute(training_case, max_steps=2)
    checkpoint = load_checkpoint(settings.paths["CHECKPOINT_DIR"] / result["run_id"] / "latest.pt")
    assert result["status"] == "COMPLETED" and result["optimizer_steps"] == 2
    assert checkpoint["scheduler"]["steps"] == 2
    assert checkpoint["optimizer"]["state"]
    assert result["validation"]["valid_tokens"] > 0
    assert (settings.paths["MODEL_DIR"] / "transformer/best_model.pt").is_file()
    with pytest.raises(TrainingFailure, match="Export exists"):
        execute(training_case, max_steps=1)


def test_resume_two_plus_two_and_fingerprint_guard(training_case):
    settings, train, validation, token_dir = training_case
    first = execute(training_case, max_steps=2)
    path = settings.paths["CHECKPOINT_DIR"] / first["run_id"] / "latest.pt"
    before = load_checkpoint(path)
    result = execute(training_case, max_steps=4, resume=path, overwrite_export=True)
    after = load_checkpoint(path)
    assert result["optimizer_steps"] == after["state"]["step"] == 4
    assert after["scheduler"]["steps"] == 4
    assert after["scheduler"]["total_steps"] == before["scheduler"]["total_steps"]
    original = validation.read_text()
    validation.write_text(original + original)
    with pytest.raises(TrainingFailure, match="fingerprint mismatch"):
        execute(training_case, max_steps=5, resume=path, overwrite_export=True)
    assert load_checkpoint(path)["state"]["step"] == 4


def test_accumulation_and_partial_epoch(training_case):
    settings, _, _, _ = training_case
    cfg = settings.values["training"]
    cfg.update(batch_size=2, gradient_accumulation_steps=4, epochs=1)
    result = execute(training_case)
    state = load_checkpoint(settings.paths["CHECKPOINT_DIR"] / result["run_id"] / "latest.pt")[
        "state"
    ]
    # 12 short documents / batch 2 = 6 microbatches -> groups of 4 and 2.
    assert state["microbatches"] == 6 and state["step"] == 2
    assert state["valid_tokens"] > 0


@pytest.mark.parametrize(
    ("exception", "status"),
    [
        (torch.cuda.OutOfMemoryError("synthetic"), "FAILED_OOM"),
        (trainer.NonFiniteError("synthetic nonfinite"), "FAILED_NONFINITE"),
    ],
)
def test_failure_preserves_previous_checkpoint(training_case, monkeypatch, exception, status):
    settings, _, _, _ = training_case
    first = execute(training_case, max_steps=1)
    directory = settings.paths["CHECKPOINT_DIR"] / first["run_id"]
    old = (directory / "latest.pt").read_bytes()
    export = settings.paths["MODEL_DIR"] / "transformer/best_model.pt"
    old_export = export.read_bytes()

    def fail(*args, **kwargs):
        raise exception

    monkeypatch.setattr(trainer, "loss_sum", fail)
    with pytest.raises(TrainingFailure) as caught:
        execute(training_case, max_steps=2, resume=directory / "latest.pt", overwrite_export=True)
    assert caught.value.summary["status"] == status
    assert (directory / "latest.pt").read_bytes() == old
    assert export.read_bytes() == old_export


def test_early_stopping_and_best_distinct(training_case, monkeypatch):
    settings, _, _, _ = training_case
    settings.values["training"]["early_stopping"].update(enabled=True, patience=1)
    sequence = iter([1.0, 2.0])

    def validation(*args, **kwargs):
        nll = next(sequence)
        return {"nll": nll, "perplexity": 2.0, "valid_tokens": 3}

    monkeypatch.setattr(trainer, "validate", validation)
    result = execute(training_case)
    assert result["status"] == "EARLY_STOPPED"
    directory = settings.paths["CHECKPOINT_DIR"] / result["run_id"]
    assert (
        load_checkpoint(directory / "best.pt")["state"]["step"]
        < load_checkpoint(directory / "latest.pt")["state"]["step"]
    )
    assert len(list(directory.glob("checkpoint_step_*.pt"))) <= 2


def test_keyboard_interrupt_and_tiny_loss_decrease(training_case, monkeypatch):
    settings, _, _, token_dir = training_case
    original = trainer.loss_sum
    calls = 0

    def interrupt(logits, labels):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return original(logits, labels)

    with monkeypatch.context() as patch:
        patch.setattr(trainer, "loss_sum", interrupt)
        result = execute(training_case, max_steps=3)
    assert result["status"] == "INTERRUPTED"
    directory = settings.paths["CHECKPOINT_DIR"] / result["run_id"]
    assert load_checkpoint(directory / "interrupt.pt")["state"]["step"] == 1
    result = execute(training_case, resume=directory / "latest.pt", max_steps=5)
    assert result["optimizer_steps"] == 5
    history = Path(result["run_directory"]) / "history.csv"
    import csv

    rows = list(csv.DictReader(history.open()))
    assert float(rows[-1]["train_nll"]) < float(rows[0]["train_nll"])


def test_min_delta_stops_but_retains_actual_best(training_case, monkeypatch):
    settings, _, _, _ = training_case
    settings.values["training"]["early_stopping"].update(enabled=True, patience=1, min_delta=0.1)
    sequence = iter([1.0, 0.95])
    monkeypatch.setattr(
        trainer,
        "validate",
        lambda *args, **kwargs: {"nll": next(sequence), "perplexity": 2.0, "valid_tokens": 3},
    )
    result = execute(training_case)
    assert result["status"] == "EARLY_STOPPED"
    best = load_checkpoint(settings.paths["CHECKPOINT_DIR"] / result["run_id"] / "best.pt")
    assert best["state"]["best_nll"] == 0.95
    assert best["state"]["early_reference"] == 1.0


def test_explicit_cuda_fails_and_test_split_rejected(training_case, monkeypatch):
    settings, train, validation, token_dir = training_case
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    settings.values["device"] = "cuda"
    with pytest.raises(TrainingFailure, match="CUDA was requested"):
        execute(training_case, dry_run=True)
    settings.values["device"] = "cpu"
    with pytest.raises(TrainingFailure, match="canonical"):
        run_training(settings, train.parent / "test.jsonl", validation, token_dir, dry_run=True)
