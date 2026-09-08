"""Architecture fingerprints, counts, and educational causal masks."""

import hashlib
import json

import torch


def inspect(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
        "fp32_weight_bytes": total * 4,
        "fp16_bf16_weight_bytes": total * 2,
        "architecture_fingerprint": hashlib.sha256(
            json.dumps(model.config.to_dict(), sort_keys=True).encode()
        ).hexdigest(),
    }


def causal_mask(length):
    return torch.tril(torch.ones(length, length, dtype=torch.int64)).tolist()
