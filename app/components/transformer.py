"""Phase 6 architecture inspection only; training is deliberately absent."""

import streamlit as st

from src.transformer.factory import build
from src.transformer.inspection import causal_mask, inspect


def render_transformer(settings):
    st.title("Decoder-only Transformer")
    st.info("Architecture ready. Final weights: Not trained. Training begins in Phase 7.")
    token_dir = settings.paths["MODEL_DIR"] / "tokenizer"
    if not (token_dir / "tokenizer.json").is_file():
        st.warning("Canonical tokenizer artifact is required for production inspection.")
        return
    if st.button("Build / Inspect Architecture"):
        try:
            model, fingerprint = build(settings, token_dir)
            st.json(
                {
                    **inspect(model),
                    "vocabulary_size": model.config.vocab_size,
                    "tokenizer_fingerprint": fingerprint,
                    "head_dim": model.config.head_dim,
                }
            )
            st.write(causal_mask(8))
        except ValueError as exc:
            st.error(str(exc))
