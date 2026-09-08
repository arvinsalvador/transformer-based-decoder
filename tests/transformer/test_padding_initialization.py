"""Padding initialization and factory/state persistence regressions."""

from copy import deepcopy
from dataclasses import replace
from io import BytesIO

import pytest
import torch

from src.config.settings import load_settings
from src.tokenizer.service import train_wordpiece
from src.transformer.config import TransformerConfig
from src.transformer.factory import build
from src.transformer.model import DecoderOnlyTransformer


@pytest.mark.parametrize("tied", [True, False])
def test_pad_zero_and_state_roundtrip(tied):
    config = TransformerConfig(17, 0, 8, 16, 1, 4, 32, 0, tie_embeddings=tied)
    model = DecoderOnlyTransformer(config)
    assert torch.count_nonzero(model.token_embedding.weight[0]) == 0
    assert torch.count_nonzero(model.token_embedding.weight[1]) > 0
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = DecoderOnlyTransformer(config)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.count_nonzero(restored.token_embedding.weight[0]) == 0
    # Embedding lookup itself suppresses pad gradients. Tied LM-head gradients are separate.
    model.token_embedding(torch.tensor([0, 1])).sum().backward()
    assert torch.count_nonzero(model.token_embedding.weight.grad[0]) == 0
    DecoderOnlyTransformer(replace(config, pad_token_id=None))


def test_factory_padding_row(tmp_path):
    settings = load_settings("config/local.yaml")
    values = deepcopy(settings.values)
    values["model"].update(embedding_dim=16, num_layers=1, feedforward_dim=32)
    settings = replace(settings, values=values, paths={k: tmp_path / k for k in settings.paths})
    source = tmp_path / "train.jsonl"
    source.write_text('{"text":"Tiny canonical tokenizer fixture."}\n')
    token_dir = train_wordpiece(source, settings).output_dir
    model, _ = build(settings, token_dir)
    assert torch.count_nonzero(model.token_embedding.weight[model.config.pad_token_id]) == 0
