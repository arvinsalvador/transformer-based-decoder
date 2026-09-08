"""Bounded Streamlit controls for Phase 3; batch CLI remains recommended for large corpora."""

from pathlib import Path

import streamlit as st

from src.config.settings import Settings
from src.data.preprocessing.pipeline import prepare_dataset


def render_preprocessing(settings: Settings) -> None:
    """Show configuration, safely invoke streaming processing, retain only capped results."""
    st.title("Preprocessing")
    st.caption(
        "Phase 3 · Conservative normalization, filtering, deduplication, and canonical splits"
    )
    st.info("For large corpora, use the batch preprocessing command on the training server.")
    cfg, splits = settings.values["preprocessing"], settings.values["split"]
    default_input = settings.paths["DATA_DIR"] / "processed/documents.jsonl"
    input_text = st.text_input("Input Phase 2 JSONL", value=str(default_input))
    input_path = Path(input_text)
    if input_path.is_file():
        st.success(
            "Input dataset found. Records are streamed; the corpus is not placed in session state."
        )
    else:
        st.warning(
            "Input dataset is not present. Ingest documents first or provide a valid JSONL path."
        )
    st.json(
        {
            "normalization": {
                key: cfg[key]
                for key in (
                    "unicode_normalization",
                    "normalize_whitespace",
                    "max_consecutive_blank_lines",
                    "preserve_case",
                    "preserve_punctuation",
                    "strip_control_characters",
                    "long_document_policy",
                )
            },
            "filters": {
                key: cfg[key]
                for key in (
                    "min_characters",
                    "max_characters_per_document",
                    "min_alpha_ratio",
                    "max_control_ratio",
                )
            },
            "splits": {
                "train": splits["train_ratio"],
                "validation": splits["validation_ratio"],
                "test": splits["test_ratio"],
                "seed": settings.values["random_seed"],
            },
        }
    )
    overwrite = st.checkbox("Replace existing prepared output", value=False)
    if st.button("Run Preprocessing", disabled=not input_path.is_file(), type="primary"):
        progress = st.progress(0.0)
        message = st.empty()
        try:
            cap = min(settings.values["dataset"]["working_document_limit"], 100000)

            def callback(snapshot: dict) -> None:
                progress.progress(min(snapshot["input_documents"] / cap, 1.0))
                message.text(
                    f"Processed {snapshot['input_documents']:,} · "
                    f"Accepted {snapshot['accepted_documents']:,} · "
                    f"Duplicates {snapshot['duplicates']:,}"
                )

            result = prepare_dataset(input_path, settings, overwrite=overwrite, progress=callback)
            st.session_state["preparation_result"] = result
            progress.progress(1.0)
            st.success("Preparation completed. Deduplication occurred before split assignment.")
        except (ValueError, OSError) as exc:
            st.error(f"Cannot prepare dataset: {exc}")
    result = st.session_state.get("preparation_result")
    if result is None:
        return
    summary = result.summary
    st.subheader("Results")
    st.write(
        {
            key: summary[key]
            for key in (
                "input_documents",
                "accepted_documents",
                "rejected_documents",
                "duplicates",
                "truncated",
                "total_input_characters",
                "total_output_characters",
                "dataset_fingerprint",
            )
        }
    )
    st.json(
        {
            "rejection_reasons": summary["rejection_reason_distribution"],
            "splits": summary["split_counts"],
            "acceptance_rate": summary["acceptance_rate"],
        }
    )
    st.text(
        f"Clean JSONL: {result.clean_output}\nRejections: {result.rejection_output}\n"
        f"Manifest: {result.manifest_path}"
    )
    st.subheader("Accepted document preview")
    if result.preview:
        st.dataframe(
            [
                {
                    key: item[key]
                    for key in (
                        "document_id",
                        "source_name",
                        "source_type",
                        "character_count",
                        "preprocessing_status",
                        "truncated",
                    )
                }
                for item in result.preview
            ],
            hide_index=True,
        )
        chosen = st.selectbox(
            "Preview document",
            range(len(result.preview)),
            format_func=lambda i: result.preview[i]["source_name"] + f" ({i + 1})",
        )
        st.text(result.preview[chosen]["text"][: cfg["preview_characters"]])
