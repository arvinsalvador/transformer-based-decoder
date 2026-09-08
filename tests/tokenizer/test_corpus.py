"""Corpus reader fails loudly on damaged canonical JSONL and streams batches."""

import json

import pytest

from src.tokenizer.corpus import CorpusError, text_batches, texts


def test_streaming_batches(splits):
    train, _, _ = splits
    assert list(texts(train, 2))[0] == "Artificial intelligence improves software systems."
    assert [len(batch) for batch in text_batches(train, 2)] == [2, 2, 2]


@pytest.mark.parametrize(
    "line", ["{bad}\n", json.dumps({"other": "x"}) + "\n", json.dumps({"text": None}) + "\n"]
)
def test_corrupt_input(tmp_path, line):
    source = tmp_path / "bad.jsonl"
    source.write_text(line)
    with pytest.raises(CorpusError):
        list(texts(source, 1))
