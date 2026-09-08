import math

import pytest
import torch

from src.training.data import Collator
from src.training.validation import validate
from src.transformer.config import TransformerConfig
from src.transformer.model import DecoderOnlyTransformer


def test_validation_weighted_read_only_and_causal_padding():
    model = DecoderOnlyTransformer(TransformerConfig(5, 0, 8, 8, 1, 2, 16, 0))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    batches = [
        Collator(0)([{"input_ids": [1, 2, 3], "labels": [2, 3, 4]}]),
        Collator(0)([{"input_ids": [1], "labels": [4]}]),
    ]
    result = validate(model, batches, torch.device("cpu"), "fp32")
    assert result["nll"] == pytest.approx(math.log(5))
    assert result["valid_tokens"] == 4 and result["sequences"] == 2
    assert result["perplexity"] == pytest.approx(5)
    assert model.training
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    # Interior masked token must not affect later valid outputs; future changes
    # must not affect earlier outputs even when a padding mask is provided.
    torch.manual_seed(1)
    model = DecoderOnlyTransformer(TransformerConfig(9, 0, 8, 8, 1, 2, 16, 0)).eval()
    mask = torch.tensor([[1, 0, 1, 1]], dtype=torch.bool)
    a, b = torch.tensor([[1, 2, 3, 4]]), torch.tensor([[1, 8, 3, 4]])
    assert torch.allclose(model(a, mask)[:, 2:], model(b, mask)[:, 2:], atol=1e-6)
    b = torch.tensor([[1, 2, 3, 8]])
    assert torch.allclose(model(a, mask)[:, :3], model(b, mask)[:, :3], atol=1e-6)
