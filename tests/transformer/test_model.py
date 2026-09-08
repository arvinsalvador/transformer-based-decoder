import pytest
import torch

from src.transformer.config import TransformerConfig
from src.transformer.inspection import causal_mask, inspect
from src.transformer.model import DecoderOnlyTransformer
from src.transformer.serialization import load_config, save_config


def tiny(tied=True):
    return DecoderOnlyTransformer(
        TransformerConfig(17, 0, 8, 16, 2, 4, 32, 0.0, tie_embeddings=tied)
    )


def test_shape_tying_and_inspection():
    model = tiny()
    logits = model(torch.tensor([[1, 2, 3], [4, 5, 0]]))
    assert logits.shape == (2, 3, 17)
    assert model.lm_head.weight is model.token_embedding.weight
    assert inspect(model)["total_parameters"] == sum(p.numel() for p in model.parameters())
    assert causal_mask(3) == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]


def test_causality_validation_and_backward():
    torch.manual_seed(1)
    model = tiny().eval()
    a = torch.tensor([[1, 2, 3, 4, 5]])
    b = torch.tensor([[1, 2, 3, 6, 7]])
    assert torch.allclose(model(a)[:, :3], model(b)[:, :3], atol=1e-6)
    model.train()
    model(torch.tensor([[1, 2, 3]])).sum().backward()
    assert model.token_embedding.weight.grad is not None
    with pytest.raises(ValueError):
        model(torch.zeros((1, 9), dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.zeros((1, 0), dtype=torch.long))


def test_untied_and_bad_configuration():
    model = tiny(False)
    assert model.lm_head.weight is not model.token_embedding.weight
    with pytest.raises(ValueError):
        TransformerConfig(10, 0, 8, 15, 1, 4, 32, 0)


def test_config_round_trip_and_state_dict_output_equivalence(tmp_path):
    model = tiny().eval()
    assert load_config(save_config(model.config, tmp_path / "architecture.json")) == model.config
    restored = DecoderOnlyTransformer(model.config).eval()
    restored.load_state_dict(model.state_dict())
    tokens = torch.tensor([[1, 2, 3]])
    assert torch.allclose(model(tokens), restored(tokens))
