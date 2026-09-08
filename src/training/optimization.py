"""Token-weighted objective, AdamW groups and optimizer-step LR scheduling."""

import math

import torch
from torch.nn import functional as F


class NonFiniteError(RuntimeError):
    pass


def loss_sum(logits, labels):
    count = int((labels != -100).sum())
    if not count:
        raise ValueError("Batch has no prediction targets")
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="sum"
    )
    if not torch.isfinite(loss):
        raise NonFiniteError("Nonfinite loss")
    return loss, count


def optimizer_for(model, cfg):
    decay, no_decay = [], []
    for parameter in model.parameters():  # shared/tied parameters occur once
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0},
        ],
        lr=cfg["learning_rate"],
        betas=(cfg["beta1"], cfg["beta2"]),
        eps=cfg["epsilon"],
    )


class WarmupCosine:
    """Set LR for the next optimizer update. Horizon survives resume unchanged."""

    def __init__(self, optimizer, total_steps, warmup_steps, minimum):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = min(warmup_steps, self.total_steps)
        self.minimum, self.steps = minimum, 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._apply()

    def ratio(self, update):
        if self.warmup_steps and update <= self.warmup_steps:
            return update / self.warmup_steps
        span = max(1, self.total_steps - self.warmup_steps)
        fraction = min(1, max(0, (update - self.warmup_steps) / span))
        return self.minimum + (1 - self.minimum) * (1 + math.cos(math.pi * fraction)) / 2

    def _apply(self):
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * self.ratio(self.steps + 1)

    def step(self):
        self.steps += 1
        self._apply()

    def state_dict(self):
        return {
            name: getattr(self, name)
            for name in ("total_steps", "warmup_steps", "minimum", "steps", "base_lrs")
        }

    def load_state_dict(self, state):
        for name, value in state.items():
            setattr(self, name, value)
        self._apply()


def finish_accumulation(model, optimizer, scaler, tokens, max_grad_norm):
    """Backward used SUM losses; divide by actual targets, including partial groups."""
    if scaler is not None:
        scaler.unscale_(optimizer)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(tokens)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if not torch.isfinite(norm):
        raise NonFiniteError("Nonfinite gradient norm")
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)
