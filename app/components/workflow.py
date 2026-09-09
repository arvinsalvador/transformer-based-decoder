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
        "Use the Training page for bounded development runs."
    ),
    "Training": "Training, validation monitoring, checkpointing, and resume are implemented.",
    "Evaluation": "Shared test evaluation is implemented; final experiments remain pending.",
    "Comparison": "Measured comparisons and generation examples are implemented.",
    "Generate Text": "View paired continuations generated during shared evaluation.",
    "Experiments": "Controlled orchestration is implemented; inspect here and execute via CLI.",
}


def render_placeholder(section: str) -> None:
    """Display the boundary of the current phase."""
    st.title(section)
    st.info("Available in a later phase")
    st.write(SECTIONS[section])
