"""Future workflow descriptions, without simulated functionality or automatic page discovery."""

import streamlit as st

SECTIONS = {
    "Documents": "Raw TXT, CSV, PDF, and DOCX ingestion.",
    "Preprocessing": "Normalize, filter, deduplicate, and create canonical splits.",
    "Tokenizer": "Train and inspect a WordPiece tokenizer from the canonical training split.",
    "Trigram Model": (
        "Trigram training, scoring, and generation are implemented; final experiments are pending."
    ),
    "Transformer": (
        "Decoder-only architecture implemented; final weights are not trained. "
        "Training begins in Phase 7."
    ),
    "Training": "Resource controls and checkpointing will be added in Phase 7.",
    "Evaluation": "Evaluation will be added in Phase 8.",
    "Comparison": "Controlled performance and training-time comparisons belong to Phase 9.",
    "Generate Text": "Generation will be added after model implementation and UI integration.",
    "Experiments": "Controlled experiments and reporting belong to Phases 9–10.",
}


def render_placeholder(section: str) -> None:
    """Display the boundary of the current phase."""
    st.title(section)
    st.info("Available in a later phase")
    st.write(SECTIONS[section])
