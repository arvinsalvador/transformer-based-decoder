import random

import numpy as np
import pytest
import torch

from src.training import checkpoint
from src.training.checkpoint import atomic_torch, export_best, load_checkpoint, save_checkpoint
from src.training.reproducibility import restore_rng, rng_state, seed_all


def payload(step):
    return {
        "format_version": 1,
        "state": {"step": step},
        "model": {"weight": torch.ones(2)},
        "rng": rng_state(),
    }


def test_rng_roundtrip_atomic_failure_retention(tmp_path, monkeypatch):
    seed_all(123, False)
    state = payload(1)
    expected = (random.random(), np.random.rand(), torch.rand(2))
    save_checkpoint(tmp_path, state, archive=True, best=True, keep=2)
    old = (tmp_path / "latest.pt").read_bytes()
    restore_rng(load_checkpoint(tmp_path / "latest.pt")["rng"])
    assert random.random() == expected[0] and np.random.rand() == expected[1]
    assert torch.equal(torch.rand(2), expected[2])
    real_save = torch.save

    def broken(value, stream):
        stream.write(b"partial")
        raise OSError("disk write failure")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", broken)
        with pytest.raises(OSError):
            save_checkpoint(tmp_path, payload(2), archive=True)
    assert (tmp_path / "latest.pt").read_bytes() == old
    assert torch.save is real_save
    for step in range(2, 6):
        save_checkpoint(tmp_path, payload(step), archive=True, keep=2)
    assert len(list(tmp_path.glob("checkpoint_step_*.pt"))) == 2
    assert load_checkpoint(tmp_path / "best.pt")["state"]["step"] == 1
    assert load_checkpoint(tmp_path / "latest.pt")["state"]["step"] == 5
    assert not list(tmp_path.glob(".*"))


def test_export_pair_preservation(tmp_path, monkeypatch):
    export_best(tmp_path, payload(1), {"run_id": "first"})
    old = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        export_best(tmp_path, payload(2), {"run_id": "second"})
    original = checkpoint.os.replace

    def broken(source, target):
        if source.name == "model_manifest.json" and source.parent != tmp_path:
            raise OSError("publication failure")
        return original(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint.os, "replace", broken)
        with pytest.raises(OSError):
            export_best(tmp_path, payload(2), {"run_id": "second"}, overwrite=True)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == old


@pytest.mark.parametrize(
    "key",
    [
        "dataset_fingerprint",
        "tokenizer_fingerprint",
        "architecture_fingerprint",
        "vocabulary_size",
    ],
)
def test_resume_identity_guards(training_case, key):
    from src.training.trainer import TrainingFailure, run_training

    settings = training_case[0]
    result = run_training(*training_case, max_steps=1)
    path = settings.paths["CHECKPOINT_DIR"] / result["run_id"] / "latest.pt"
    modified = load_checkpoint(path)
    modified["fingerprints"][key] = "mismatch"
    atomic_torch(path, modified)
    with pytest.raises(TrainingFailure, match="fingerprint mismatch"):
        run_training(*training_case, resume=path, max_steps=2, overwrite_export=True)


def test_exact_resume_matches_uninterrupted(training_case, monkeypatch):
    """Same scheduler horizon, dropout RNG, data cursor and optimizer moments."""
    from src.training import trainer

    settings = training_case[0]
    settings.values["model"]["dropout"] = 0.1
    full = trainer.run_training(*training_case, max_steps=4)
    full_state = load_checkpoint(settings.paths["CHECKPOINT_DIR"] / full["run_id"] / "latest.pt")
    original = trainer.finish_accumulation
    calls = 0

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        return original(*args)

    with monkeypatch.context() as patch:
        patch.setattr(trainer, "finish_accumulation", interrupted)
        partial = trainer.run_training(*training_case, max_steps=4, overwrite_export=True)
    path = settings.paths["CHECKPOINT_DIR"] / partial["run_id"] / "latest.pt"
    assert load_checkpoint(path)["state"]["step"] == 2
    resumed = trainer.run_training(*training_case, max_steps=4, resume=path, overwrite_export=True)
    resumed_state = load_checkpoint(
        settings.paths["CHECKPOINT_DIR"] / resumed["run_id"] / "latest.pt"
    )
    for name, tensor in full_state["model"].items():
        assert torch.equal(tensor, resumed_state["model"][name]), name
    assert full_state["scheduler"] == resumed_state["scheduler"]
    for key in full_state["optimizer"]["state"]:
        for name in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(
                full_state["optimizer"]["state"][key][name],
                resumed_state["optimizer"]["state"][key][name],
            )
