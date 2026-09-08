import json
import random
from collections import Counter

import pytest
from torch.utils.data import DataLoader

from src.training.data import Collator, SequenceDataset, bounded_shuffle, sequence_windows


def test_shift_and_chunk_boundaries():
    result = list(sequence_windows([10, 11, 12], 2, 3, 4, 4))
    assert result == [{"input_ids": [2, 10, 11, 12], "labels": [10, 11, 12, 3]}]
    chunks = list(sequence_windows(list(range(10, 21)), 2, 3, 4, 4))
    assert all(len(s["input_ids"]) <= 4 for s in chunks)
    assert [v for chunk in chunks for v in chunk["labels"]] == [*range(10, 21), 3]
    assert list(sequence_windows([], 2, 3, 4, 4))[0]["labels"] == [3]
    assert list(sequence_windows([10], 2, 3, 4, 4))[0]["input_ids"] == [2, 10]
    overlap = list(sequence_windows(list(range(10)), 20, 21, 4, 2))
    assert overlap[1]["input_ids"] == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        list(sequence_windows([1], 2, 3, 4, 5))


def test_collator():
    samples = [{"input_ids": [2, 10, 11], "labels": [10, 11, 3]}, {"input_ids": [2], "labels": [3]}]
    batch = Collator(7)(samples)
    assert batch["input_ids"].tolist() == [[2, 10, 11], [2, 7, 7]]
    assert batch["attention_mask"].tolist() == [[True, True, True], [True, False, False]]
    assert batch["labels"].tolist() == [[10, 11, 3], [3, -100, -100]]
    assert Collator(7)(samples[:1])["labels"].shape == (1, 3)


def test_bounded_shuffle():
    consumed = []

    def source():
        for i in range(100):
            consumed.append(i)
            yield i

    shuffled = bounded_shuffle(source(), 4, random.Random(42))
    first = next(shuffled)
    assert len(consumed) == 5
    assert sorted([first, *shuffled]) == list(range(100))
    assert list(bounded_shuffle(range(20), 4, random.Random(42))) == list(
        bounded_shuffle(range(20), 4, random.Random(42))
    )


def make_dataset(case, shuffle=0):
    _, path, _, token_dir = case
    manifest = json.loads((token_dir / "tokenizer_manifest.json").read_text())
    return SequenceDataset(path, token_dir, manifest["special_tokens"], 8, 8, shuffle=shuffle)


def test_worker_partition_and_persistent_epoch(training_case):
    dataset = make_dataset(training_case, shuffle=3)
    expected = Counter(item["document_index"] for item in dataset.records(0, 1))
    partitioned = [
        item["document_index"] for worker in range(2) for item in dataset.records(worker, 2)
    ]
    assert Counter(partitioned) == expected
    from src.training.data import close_loader

    batches = DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        multiprocessing_context="spawn",
        persistent_workers=True,
    )
    try:
        a = [item["document_index"] for item in batches]
        assert Counter(a) == expected
        assert [item["document_index"] for item in batches] == a
        dataset.set_epoch(1)
        b = [item["document_index"] for item in batches]
        assert Counter(b) == expected and b != a
    finally:
        close_loader(batches)


def test_incremental_reading_and_validation_order(training_case):
    dataset = make_dataset(training_case)
    first = next(iter(dataset))
    assert first["document_index"] == 0
    original = dataset.path.read_text()
    dataset.path.write_text(original + "{bad}")
    assert next(iter(dataset))["document_index"] == 0
    with pytest.raises(ValueError):
        list(dataset)
    dataset.path.write_text(original)
    assert list(dataset) == list(dataset)
    assert all(item["input_ids"][0] == dataset.special["bos_token_id"] for item in dataset)
