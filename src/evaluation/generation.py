"""Identical canonical prompts, existing trigram generation and context-cropped decoder."""

from time import perf_counter

import torch
import yaml

from src.evaluation.adapters import synchronize
from src.training.precision import autocast

RUBRIC = {
    name: {"rating": None, "notes": None, "scale": "1 (poor) to 5 (strong); human-entered only"}
    for name in (
        "grammatical_coherence",
        "topic_relevance",
        "non_repetition",
        "context_continuity",
        "completeness",
        "technical_plausibility",
    )
}


def load_prompts(path):
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    prompts = data.get("prompts") if isinstance(data, dict) else None
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(p, str) or not p.strip() for p in prompts)
    ):
        raise ValueError("Prompt file requires a nonempty list of nonempty strings")
    return prompts


def transformer_generate(model, ids, special, cfg, device, precision, seed):
    model.eval()
    history, generated = [special["bos_token_id"], *ids], []
    rng = torch.Generator(device=device).manual_seed(seed)
    stopped = False
    synchronize(device)
    start = perf_counter()
    with torch.inference_mode():
        for _ in range(cfg["max_new_tokens"]):
            inputs = torch.tensor([history[-model.config.context_length :]], device=device)
            with autocast(device, precision):
                logits = model(inputs)[0, -1].float().clone()
            logits[[special["pad_token_id"], special["bos_token_id"]]] = -torch.inf
            if (
                not torch.isfinite(logits).any()
                or torch.isnan(logits).any()
                or torch.isposinf(logits).any()
            ):
                raise ValueError("Invalid generation logits")
            if cfg["strategy"] == "greedy":
                token = int(logits.argmax())
            else:
                values, indices = torch.topk(
                    logits / cfg["temperature"], min(cfg["top_k"], logits.numel())
                )
                token = int(indices[torch.multinomial(values.softmax(-1), 1, generator=rng)])
            if token == special["eos_token_id"]:
                stopped = True
                break
            generated.append(token)
            history.append(token)
    synchronize(device)
    duration = perf_counter() - start
    return generated, {
        "duration_seconds": duration,
        "tokens_per_second": len(generated) / duration if duration else None,
        "stopped_on_eos": stopped,
    }


def generate_comparison(models, tokenizer, prompts, special, cfg, device, precision, seed):
    results = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        for name, model in models.items():
            if name == "trigram":
                generated, timing = model.generate(ids, **cfg, seed=seed)
            else:
                generated, timing = transformer_generate(
                    model, ids, special, cfg, device, precision, seed
                )
            results.append(
                {
                    "prompt": prompt,
                    "prompt_token_count": len(ids),
                    "model_type": name,
                    "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
                    "generated_token_ids": generated,
                    "generated_token_count": len(generated),
                    "strategy": cfg["strategy"],
                    "seed": seed,
                    **timing,
                    "human_review": RUBRIC,
                }
            )
    return results
