"""Per-document causal windows, deterministic worker sharding and padded batches."""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from src.tokenizer.service import load_tokenizer


def sequence_windows(ids, bos, eos, context, stride):
    """Context is INPUT length: each full window consumes context+1 source IDs."""
    if not 1 <= stride <= context:
        raise ValueError("stride must be between 1 and context")
    full = [bos, *ids, eos]
    for start in range(0, len(full) - 1, stride):
        window = full[start : start + context + 1]
        yield {"input_ids": window[:-1], "labels": window[1:]}
        if start + context >= len(full) - 1:
            break


def bounded_shuffle(source, capacity, rng):
    buffer = []
    for item in source:
        if len(buffer) < capacity:
            buffer.append(item)
        else:
            index = rng.randrange(capacity)
            yield buffer[index]
            buffer[index] = item
    rng.shuffle(buffer)
    yield from buffer


class SequenceDataset(IterableDataset):
    def __init__(self, path, tokenizer_path, special, context, stride, *, shuffle=0, seed=42):
        self.path, self.tokenizer_path = Path(path), Path(tokenizer_path)
        self.special, self.context, self.stride = special, context, stride
        self.shuffle, self.seed = shuffle, seed
        # Shared epoch is visible to persistent DataLoader workers on fork/spawn.
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def set_epoch(self, epoch):
        self.epoch.fill_(epoch)

    def records(self, worker_id, workers):
        tokenizer = load_tokenizer(self.tokenizer_path)
        with self.path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if index % workers != worker_id:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    raise ValueError(f"Invalid split text at line {index + 1}")
                ids = tokenizer.encode(record["text"], add_special_tokens=False).ids
                if self.special["pad_token_id"] in ids:
                    raise ValueError("Corpus contains literal PAD token; clean it before training")
                for sample in sequence_windows(
                    ids,
                    self.special["bos_token_id"],
                    self.special["eos_token_id"],
                    self.context,
                    self.stride,
                ):
                    yield {**sample, "document_index": index}

    def __iter__(self):
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        source = self.records(worker_id, workers)
        if self.shuffle:
            rng = random.Random(self.seed + int(self.epoch) * 100003 + worker_id)
            yield from bounded_shuffle(source, self.shuffle, rng)
        else:
            yield from source


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, samples):
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        length = max(len(s["input_ids"]) for s in samples)
        inputs = torch.full((len(samples), length), self.pad_id, dtype=torch.long)
        labels = torch.full_like(inputs, -100)
        mask = torch.zeros_like(inputs, dtype=torch.bool)
        for row, sample in enumerate(samples):
            size = len(sample["input_ids"])
            inputs[row, :size] = torch.tensor(sample["input_ids"])
            labels[row, :size] = torch.tensor(sample["labels"])
            mask[row, :size] = True
        return {"input_ids": inputs, "attention_mask": mask, "labels": labels}


def loader(dataset, cfg, pad_id, *, validation=False):
    # Validation stays ordered and avoids independent worker tails.
    workers = 0 if validation else cfg["num_workers"]
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=workers,
        collate_fn=Collator(pad_id),
        pin_memory=cfg["pin_memory"],
        persistent_workers=bool(workers and cfg["persistent_workers"]),
        generator=torch.Generator().manual_seed(dataset.seed),
        **({"prefetch_factor": 1, "multiprocessing_context": "spawn"} if workers else {}),
    )


def close_loader(value):
    # PyTorch has no public close method for persistent-worker DataLoaders.
    iterator = getattr(value, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
