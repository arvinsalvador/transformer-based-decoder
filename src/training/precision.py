"""Explicit CPU/CUDA precision policy using the installed PyTorch AMP API."""

from contextlib import nullcontext

import torch


def select_precision(device, requested, mixed):
    if device.type == "cpu":
        if requested in ("fp16", "bf16"):
            raise ValueError("CPU training requires fp32 or auto precision")
        return "fp32"
    if not mixed or requested == "fp32":
        return "fp32"
    supported = torch.cuda.is_bf16_supported()
    if requested == "bf16" and not supported:
        raise ValueError("Explicit BF16 requested but GPU does not support BF16")
    return ("bf16" if supported else "fp16") if requested == "auto" else requested


def autocast(device, precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16
    )


def make_scaler(precision):
    return torch.amp.GradScaler("cuda") if precision == "fp16" else None
