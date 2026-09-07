"""Future workflow descriptions, without simulated functionality or automatic page discovery."""

import streamlit as st

SECTIONS = {
    "Documents": "TXT, CSV, DOCX, and PDF ingestion will be added in Phase 2.",
    "Preprocessing": "Cleaning and dataset splits will be added in Phase 3.",
    "Tokenizer": "WordPiece tokenizer training will be added in Phase 4.",
    "Trigram Model": "The trigram baseline will be added in Phase 5.",
    "Transformer": "The decoder-only architecture will be added in Phase 6.",
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
