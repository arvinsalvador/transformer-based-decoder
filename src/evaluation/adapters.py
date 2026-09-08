"""Read-only likelihood adapters. Loading/tokenizer fitting is outside timed scoring."""

import math
from time import perf_counter

import torch

from src.evaluation.data import batches, documents
from src.evaluation.metrics import ModelEvaluationResult
from src.training.optimization import loss_sum
from src.training.precision import autocast


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def score_trigram(model, tokenizer, path, limit, special, metadata):
    start, count, events, total = perf_counter(), 0, 0, 0.0
    for ids in documents(path, tokenizer, limit, special["pad_token_id"]):
        count += 1
        first = second = special["bos_token_id"]
        for target in [*ids, special["eos_token_id"]]:
            total -= math.log(model.probability(first, second, target))
            events += 1
            first, second = second, target
    return ModelEvaluationResult(
        "trigram", count, events, total, perf_counter() - start, "cpu", "float64", metadata
    ).to_dict()


def score_transformer(model, tokenizer, path, limit, special, metadata, cfg, device, precision):
    model.eval()
    counters = {"documents": 0}

    def stream():
        return batches(
            documents(path, tokenizer, limit, special["pad_token_id"]),
            special,
            model.config.context_length,
            cfg["transformer_batch_size"],
            counters,
        )

    def forward(batch):
        batch = {key: value.to(device) for key, value in batch.items()}
        with autocast(device, precision):
            return loss_sum(model(batch["input_ids"], batch["attention_mask"]), batch["labels"])

    with torch.inference_mode():
        warmups = cfg["gpu_warmup_batches"] if torch.device(device).type == "cuda" else 0
        if warmups:
            # Warmup replays a bounded first batch; the timed pass still scores ALL targets.
            iterator = stream()
            first = next(iterator, None)
            if first is not None:
                for _ in range(warmups):
                    forward(first)
            iterator.close()
            counters["documents"] = 0
        synchronize(device)
        start, total, events = perf_counter(), 0.0, 0
        for batch in stream():
            loss, count = forward(batch)
            total += float(loss)
            events += count
        synchronize(device)
    return ModelEvaluationResult(
        "transformer",
        counters["documents"],
        events,
        total,
        perf_counter() - start,
        str(device),
        precision,
        {**metadata, "warmup_batches": warmups},
    ).to_dict()
