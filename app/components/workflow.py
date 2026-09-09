"""Implemented workflow descriptions, without automatic page discovery."""

SECTIONS = {
    "Final Results": "Audited FULL result, readiness, saved examples and presentation view.",
    "Documents": "Raw TXT, CSV, PDF, and DOCX ingestion.",
    "Preprocessing": "Normalize, filter, deduplicate, and create canonical splits.",
    "Tokenizer": "Train and inspect a WordPiece tokenizer from the canonical training split.",
    "Trigram Model": (
        "Trigram training, scoring, and generation are implemented; final experiments are pending."
    ),
    "Transformer": (
        "Decoder-only architecture implemented; consult saved final audit for training status. "
        "Use the Training page for bounded development runs."
    ),
    "Training": "Training, validation monitoring, checkpointing, and resume are implemented.",
    "Evaluation": "Shared test evaluation is implemented; final experiments remain pending.",
    "Comparison": "Measured comparisons and generation examples are implemented.",
    "Generate Text": "View paired continuations generated during shared evaluation.",
    "Experiments": "Controlled orchestration is implemented; inspect here and execute via CLI.",
}
