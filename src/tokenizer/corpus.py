"""Strict, streaming readers for canonical Phase 3 split JSONL."""

import json
from collections.abc import Iterator
from pathlib import Path


class CorpusError(ValueError):
    """A canonical split contains malformed or unsafe training data."""


def text_batches(path: str | Path, batch_size: int) -> Iterator[list[str]]:
    """Yield stable JSONL-order text batches without retaining the corpus."""
    source = Path(path)
    if not source.is_file() or source.suffix != ".jsonl":
        raise CorpusError("Corpus must be an existing JSONL file")
    batch: list[str] = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"Malformed JSONL at line {line_number}") from exc
            text = record.get("text") if isinstance(record, dict) else None
            if not isinstance(text, str) or not text:
                raise CorpusError(f"Missing nonempty text at line {line_number}")
            batch.append(text)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def texts(path: str | Path, batch_size: int) -> Iterator[str]:
    """Flatten the streaming batch reader for analysis."""
    for batch in text_batches(path, batch_size):
        yield from batch
