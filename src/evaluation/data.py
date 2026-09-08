"""One incremental document stream and one exact target-event policy."""

import json
from itertools import islice
from pathlib import Path

from src.training.data import Collator, sequence_windows


def documents(path, tokenizer, limit=None, pad_id=None):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(islice(stream, limit), 1):
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"Invalid test text at line {number}")
            ids = tokenizer.encode(record["text"], add_special_tokens=False).ids
            if pad_id in ids:
                raise ValueError("Literal PAD in evaluation document; do not silently omit targets")
            yield ids


def batches(source, special, context, size, counters):
    pending = []
    collate = Collator(special["pad_token_id"])
    for ids in source:
        counters["documents"] += 1
        for window in sequence_windows(
            ids, special["bos_token_id"], special["eos_token_id"], context, context
        ):
            pending.append(window)
            if len(pending) == size:
                yield collate(pending)
                pending.clear()
    if pending:
        yield collate(pending)
