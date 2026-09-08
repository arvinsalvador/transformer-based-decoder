import math

import pytest
import torch

from src.training.optimization import (
    NonFiniteError,
    WarmupCosine,
    finish_accumulation,
    loss_sum,
    optimizer_for,
)
from src.training.precision import make_scaler, select_precision
from src.transformer.config import TransformerConfig
from src.transformer.model import DecoderOnlyTransformer


def test_loss_sum_hand_computable():
    logits = torch.zeros(1, 3, 4)
    loss, count = loss_sum(logits, torch.tensor([[0, 2, -100]]))
    assert count == 2 and float(loss) == pytest.approx(2 * math.log(4))
    with pytest.raises(ValueError):
        loss_sum(logits, torch.full((1, 3), -100))


def test_scheduler_and_optimizer(training_case):
    cfg = training_case[0].values["training"]
    model = DecoderOnlyTransformer(TransformerConfig(10, 0, 8, 8, 1, 2, 16, 0))
    optimizer = optimizer_for(model, cfg)
    assert isinstance(optimizer, torch.optim.AdamW)
    params = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(params) == len({id(p) for p in model.parameters()}) == len(set(map(id, params)))
    assert all(p.ndim < 2 for p in optimizer.param_groups[1]["params"])
    schedule = WarmupCosine(optimizer, 10, 2, 0.1)
    assert schedule.ratio(1) == 0.5
    assert schedule.ratio(2) == 1
    assert schedule.ratio(6) == pytest.approx(0.55)
    assert schedule.ratio(10) == pytest.approx(0.1)
    assert WarmupCosine(optimizer, 2, 500, 0.1).ratio(2) == 1


def test_partial_accumulation_matches_token_weighted_full_batch():
    torch.manual_seed(42)
    a, b = torch.nn.Linear(3, 4), torch.nn.Linear(3, 4)
    b.load_state_dict(a.state_dict())
    inputs = [torch.randn(1, 3), torch.randn(3, 3)]
    targets = [torch.tensor([1]), torch.tensor([0, 2, 3])]
    opt_a = torch.optim.SGD(a.parameters(), lr=0.1)
    opt_b = torch.optim.SGD(b.parameters(), lr=0.1)
    before = a.weight.detach().clone()
    for x, y in zip(inputs, targets, strict=True):
        loss, _ = loss_sum(a(x), y)
        loss.backward()
        assert torch.equal(before, a.weight)  # No update before the boundary.
    finish_accumulation(a, opt_a, None, 4, 1000)
    loss, _ = loss_sum(b(torch.cat(inputs)), torch.cat(targets))
    (loss / 4).backward()
    opt_b.step()
    assert torch.allclose(a.weight, b.weight)


@pytest.mark.parametrize(
    ("supported", "requested", "expected"),
    [
        (True, "auto", "bf16"),
        (False, "auto", "fp16"),
        (True, "fp32", "fp32"),
        (True, "bf16", "bf16"),
        (False, "fp16", "fp16"),
    ],
)
def test_precision_mocked(monkeypatch, supported, requested, expected):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: supported)
    assert select_precision(torch.device("cuda"), requested, True) == expected
    assert select_precision(torch.device("cpu"), "auto", True) == "fp32"
    assert make_scaler("bf16") is None and make_scaler("fp32") is None
    calls = []
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *args: calls.append(args) or "scaler")
    assert make_scaler("fp16") == "scaler" and calls == [("cuda",)]
    if not supported:
        with pytest.raises(ValueError):
            select_precision(torch.device("cuda"), "bf16", True)


def test_scaler_unscales_before_normalization_and_clipping(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.weight.grad = torch.tensor([[80.0]])
    events = []

    class Scaler:
        def unscale_(self, opt):
            events.append("unscale")
            model.weight.grad.div_(10)

        def step(self, opt):
            events.append("step")
            assert model.weight.grad.item() == pytest.approx(1)
            opt.step()

        def update(self):
            events.append("update")

    original = torch.nn.utils.clip_grad_norm_

    def clip(parameters, maximum):
        events.append("clip")
        assert model.weight.grad.item() == 4  # Unscaled, then divided by two targets.
        return original(parameters, maximum)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    assert finish_accumulation(model, optimizer, Scaler(), 2, 1) == 4
    assert events == ["unscale", "clip", "step", "update"]
    assert model.weight.grad is None


def test_nonfinite_gradient_never_steps():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    model.weight.grad = torch.tensor([[float("nan")]])
    with pytest.raises(NonFiniteError, match="gradient"):
        finish_accumulation(model, optimizer, None, 1, 1)
    assert torch.equal(model.weight, before)
