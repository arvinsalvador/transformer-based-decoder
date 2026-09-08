"""Strict, independently validated evaluation settings."""

import math

FIELDS = {
    "document_limit",
    "transformer_batch_size",
    "precision",
    "generation",
    "gpu_warmup_batches",
}


def validate_evaluation(cfg):
    if not isinstance(cfg, dict) or set(cfg) != FIELDS:
        raise ValueError(f"evaluation must contain exactly {sorted(FIELDS)}")
    for name, minimum in (
        ("document_limit", 1),
        ("transformer_batch_size", 1),
        ("gpu_warmup_batches", 0),
    ):
        value = cfg[name]
        if name == "document_limit" and value is None:
            continue
        if type(value) is not int or value < minimum:
            raise ValueError(f"evaluation.{name} must be an integer >= {minimum}")
    if cfg["precision"] not in ("auto", "fp32", "fp16", "bf16"):
        raise ValueError("Invalid evaluation precision")
    generation = cfg["generation"]
    if not isinstance(generation, dict) or set(generation) != {
        "strategy",
        "max_new_tokens",
        "temperature",
        "top_k",
    }:
        raise ValueError("Invalid evaluation.generation fields")
    if generation["strategy"] not in ("greedy", "sample"):
        raise ValueError("Invalid evaluation generation strategy")
    for name in ("max_new_tokens", "top_k"):
        if type(generation[name]) is not int or generation[name] < 1:
            raise ValueError(f"evaluation.generation.{name} must be positive integer")
    value = generation["temperature"]
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("evaluation generation temperature must be finite and positive")
