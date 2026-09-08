"""Training-only validation monitoring; test-set evaluation belongs to Phase 8."""

from time import perf_counter

import torch

from src.training.monitoring import perplexity
from src.training.optimization import loss_sum
from src.training.precision import autocast


def validate(model, batches, device, precision):
    started = perf_counter()
    nll_sum = tokens = sequences = 0
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
                with autocast(device, precision):
                    logits = model(batch["input_ids"], batch["attention_mask"])
                    loss, count = loss_sum(logits, batch["labels"])
                nll_sum += float(loss)
                tokens += count
                sequences += batch["input_ids"].shape[0]
        if not tokens:
            raise ValueError("Validation split has no prediction targets")
        nll = nll_sum / tokens
        duration = perf_counter() - started
        return {
            "nll": nll,
            "perplexity": perplexity(nll),
            "valid_tokens": tokens,
            "sequences": sequences,
            "duration_seconds": duration,
            "tokens_per_second": tokens / max(duration, 1e-12),
        }
    finally:
        model.train(was_training)
