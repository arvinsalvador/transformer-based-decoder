"""Stable hash buckets avoid retaining document texts for random shuffling."""

import hashlib


def assign_split(normalized_sha256: str, seed: int, ratios: dict[str, float]) -> str:
    """Map hash+global seed to a deterministic [0,1) bucket."""
    raw = hashlib.sha256(f"{seed}:{normalized_sha256}".encode("ascii")).digest()
    bucket = int.from_bytes(raw[:8], "big") / 2**64
    if bucket < ratios["train_ratio"]:
        return "train"
    if bucket < ratios["train_ratio"] + ratios["validation_ratio"]:
        return "validation"
    return "test"
