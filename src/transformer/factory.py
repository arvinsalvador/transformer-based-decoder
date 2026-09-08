"""Build architecture from validated settings and canonical tokenizer metadata."""

import json
from pathlib import Path

from src.tokenizer.service import load_tokenizer
from src.transformer.config import TransformerConfig
from src.transformer.model import DecoderOnlyTransformer


def build(settings, tokenizer_path=None, *, synthetic_vocab_size=None):
    if tokenizer_path is None and synthetic_vocab_size is None:
        raise ValueError("Canonical tokenizer is required outside synthetic development mode")
    if tokenizer_path:
        directory = Path(tokenizer_path)
        tokenizer = load_tokenizer(directory)
        manifest = json.loads((directory / "tokenizer_manifest.json").read_text())
        vocab_size, pad_id, fingerprint = (
            tokenizer.get_vocab_size(),
            manifest["special_tokens"]["pad_token_id"],
            manifest["tokenizer_fingerprint"],
        )
    else:
        vocab_size, pad_id, fingerprint = synthetic_vocab_size, None, None
    values = settings.values["model"]
    config = TransformerConfig(
        vocab_size=vocab_size,
        pad_token_id=pad_id,
        context_length=values["context_length"],
        embedding_dim=values["embedding_dim"],
        num_layers=values["num_layers"],
        num_heads=values["num_heads"],
        feedforward_dim=values["feedforward_dim"],
        dropout=values["dropout"],
        layer_norm_eps=values["layer_norm_eps"],
        bias=values["bias"],
        tie_embeddings=values["tie_embeddings"],
        initialization_std=values["initialization_std"],
    )
    return DecoderOnlyTransformer(config), fingerprint
