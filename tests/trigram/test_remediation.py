"""Disk counting, bounded writes, real tokenizer integration and failure recovery."""

import json
import sqlite3
from collections import Counter
from copy import deepcopy
from dataclasses import replace

import pytest

from src.config.settings import load_settings
from src.tokenizer.service import load_tokenizer, train_wordpiece
from src.trigram import trainer
from src.trigram.model import TrigramModel
from src.trigram.serialization import load_model, save_model
from src.trigram.sqlite_model import CountTable, SQLiteTrigramModel


def test_disk_matches_memory_and_bounds_writes(tmp_path, monkeypatch):
    metadata = dict(vocabulary_size=1000, bos_id=1, eos_id=2, add_k=0.1)
    memory = TrigramModel(**metadata)
    sequences = [[3, 4, 5], [3, 4, 6], list(range(10, 510))]
    for ids in sequences:
        memory.update(ids)

    def forbidden(*args, **kwargs):
        raise AssertionError("Disk operations must not construct a memory model")

    monkeypatch.setattr(TrigramModel, "__init__", forbidden)
    path = tmp_path / "counts.sqlite"
    with SQLiteTrigramModel(path, metadata, batch_size=32) as disk:
        for ids in sequences:
            disk.update(ids)
        assert disk.flushes > 10
        assert disk.max_buffered_updates <= 32
        assert not any(disk.buffers)
        assert disk.statistics() == memory.statistics()
        for name in ("unigrams", "bigrams", "trigrams"):
            assert isinstance(getattr(disk, name), CountTable)
            assert dict(getattr(disk, name).items()) == getattr(memory, name)
        assert disk.probability(3, 4, 5) == pytest.approx(1.1 / 102)
        assert disk.probability(900, 900, 900) == memory.probability(900, 900, 900)
    with load_model(path) as disk:
        before = path.read_bytes()
        for ids in ([], [3], [3, 4], [900, 901]):
            a, b = disk.score_sequences([ids]), memory.score_sequences([ids])
            assert a.log_likelihood == pytest.approx(b.log_likelihood)
            assert a.average_nll == pytest.approx(b.average_nll)
            assert a.perplexity == pytest.approx(b.perplexity)
            for strategy in ("greedy", "sample"):
                options = dict(max_new_tokens=8, strategy=strategy, seed=42)
                assert disk.generate(ids, **options)[0] == memory.generate(ids, **options)[0]
        assert path.read_bytes() == before
        with pytest.raises(ValueError, match="read-only"):
            disk.update([3])
        with pytest.raises(sqlite3.OperationalError):
            disk.connection.execute("DELETE FROM trigram")
    with pytest.raises(sqlite3.ProgrammingError):
        disk.statistics()


@pytest.fixture
def corpus(tmp_path):
    settings = load_settings("config/local.yaml")
    values = deepcopy(settings.values)
    values["trigram"]["storage_backend"] = "sqlite"
    settings = replace(
        settings, values=values, paths={key: tmp_path / key.lower() for key in settings.paths}
    )
    source, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    source.write_text(json.dumps({"text": "Artificial intelligence improves software."}) + "\n")
    validation.write_text(json.dumps({"text": "Software improves intelligence."}) + "\n")
    token_dir = settings.paths["MODEL_DIR"] / "tokenizer"
    train_wordpiece(source, settings, output_dir=token_dir)
    return settings, source, validation, token_dir


def test_integration_overwrite_and_failures(corpus, monkeypatch):
    settings, source, validation, token_dir = corpus
    target = settings.paths["MODEL_DIR"] / "trigram"

    def no_memory(*args, **kwargs):
        raise AssertionError("SQLite trainer constructed memory counts")

    monkeypatch.setattr(trainer, "TrigramModel", no_memory)

    def run(**kwargs):
        return trainer.train(source, token_dir, settings, validation_path=validation, **kwargs)

    first = run()
    assert first["storage_backend"] == "sqlite"
    assert first["training_documents"] == 1
    assert first["source_tokens"] > 0
    assert first["model_artifact_size_bytes"] > 0
    assert first["training_duration_seconds"] >= 0
    tokenizer = load_tokenizer(token_dir)
    with load_model(target / "trigram_counts.sqlite") as model:
        result = trainer.score(model, tokenizer, validation, settings)
        assert result["perplexity"] == first["validation"]["perplexity"]
        assert model.generate([], max_new_tokens=5)[1]["generated_tokens"] > 0

    def snapshot():
        return {name: (target / name).read_bytes() for name in trainer.ARTIFACTS}

    old = snapshot()
    with pytest.raises(FileExistsError):
        run()
    assert snapshot() == old
    source.write_text(source.read_text() * 2)
    second = run(overwrite=True)
    assert second["training_documents"] == 2
    assert second["trigram_events"] == first["trigram_events"] * 2
    old = snapshot()
    with load_model(target / "trigram_counts.sqlite") as model:
        assert (
            model.trigrams[
                model.bos_id,
                model.bos_id,
                tokenizer.encode("Artificial", add_special_tokens=False).ids[0],
            ]
            == 2
        )
    source.write_text(source.read_text() + "{broken}\n")
    with pytest.raises(ValueError):
        run(overwrite=True)
    assert snapshot() == old
    source.write_text(json.dumps({"text": "Artificial intelligence improves software."}) + "\n")
    # Fail after the DB has been replaced but before metadata publication completes.
    real_replace = trainer.os.replace

    def fail_metadata(src, dst):
        if src.name == "training_statistics.json" and src.parent != target:
            raise OSError("injected publish failure")
        return real_replace(src, dst)

    with monkeypatch.context() as patch:
        patch.setattr(trainer.os, "replace", fail_metadata)
        with pytest.raises(OSError, match="injected"):
            run(overwrite=True)
    assert snapshot() == old
    assert set(p.name for p in target.iterdir()) == set(trainer.ARTIFACTS)


@pytest.mark.parametrize(("profile", "expected"), [("local", "memory"), ("gpu", "sqlite")])
def test_auto_backend_records_actual_choice(corpus, profile, expected):
    settings, source, validation, token_dir = corpus
    settings.values["environment"] = profile
    settings.values["trigram"]["storage_backend"] = "auto"
    result = trainer.train(source, token_dir, settings, validation_path=validation)
    assert result["storage_backend"] == expected


def test_save_existing_database_is_safe(tmp_path, monkeypatch):
    model = TrigramModel(10, 1, 2, 0.1)
    model.update([3])
    path = save_model(model, tmp_path / "counts.sqlite")
    old = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_model(model, path)
    model.update([4])
    save_model(model, path, overwrite=True)
    with load_model(path) as loaded:
        assert loaded.unigrams[4] == 1
    old = path.read_bytes()
    model.trigrams = Counter({(object(), 3, 4): 1})
    with pytest.raises(sqlite3.ProgrammingError):
        save_model(model, path, overwrite=True)
    assert path.read_bytes() == old
    assert list(tmp_path.iterdir()) == [path]
