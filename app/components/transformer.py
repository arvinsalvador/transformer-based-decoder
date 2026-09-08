"""Phase 6 architecture inspection only; training is deliberately absent."""

import streamlit as st

from src.transformer.factory import build
from src.transformer.inspection import causal_mask, inspect


def render_transformer(settings):
    st.title("Decoder-only Transformer")
    available = (settings.paths["MODEL_DIR"] / "transformer/best_model.pt").is_file()
    st.info(
        "Architecture ready. "
        + ("A model export is available. " if available else "No model export is available. ")
        + "Use Training for bounded runs; final experiments remain pending."
    )
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
