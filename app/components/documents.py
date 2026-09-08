"""Small-upload and configured raw-folder workflows backed by the same streaming API."""

import csv
import io
from uuid import uuid4

import streamlit as st

from src.config.settings import Settings
from src.data.ingestion import ingest_directory
from src.data.models import IngestionOptions
from src.data.uploads import stage_uploads


def render_documents(settings: Settings) -> None:
    """Render source selection, bounded CSV previews, progress and actual run results."""
    st.title("Documents")
    st.caption("Phase 2 · Raw document extraction · TXT | CSV | PDF | DOCX")
    st.info(
        "Large datasets should be placed in the configured data/raw directory and ingested "
        "using the batch/server ingestion workflow. Browser uploads are for small samples only."
    )
    config = settings.values
    limits = config["ingestion"]
    st.write(
        f"Configured document limit: {config['dataset']['working_document_limit']:,} · "
        "Maximum homework limit: 100,000"
    )
    st.caption("Each candidate, including a skipped or failed extraction, consumes one limit slot.")
    source_mode = st.radio(
        "Dataset source", ["Upload Files", "Existing Raw Folder"], horizontal=True
    )
    uploads = []
    eligible = True
    header_names = []
    if source_mode == "Upload Files":
        st.caption(
            f"Up to {limits['max_upload_files']} files, {limits['max_upload_total_mb']} MiB total. "
            f"Per-file limit: {config['dataset']['max_file_size_mb']} MiB."
        )
        uploads = st.file_uploader(
            "Choose small document samples",
            type=["txt", "csv", "pdf", "docx"],
            accept_multiple_files=True,
            max_upload_size=config["dataset"]["max_file_size_mb"],
        )
        eligible = (
            0 < len(uploads) <= limits["max_upload_files"]
            and sum(item.size for item in uploads) <= limits["max_upload_total_mb"] * 1024**2
        )
        if uploads and not eligible:
            st.warning("Upload count or total size exceeds the configured sample limit.")
        if eligible:
            for upload in uploads:
                if upload.name.lower().endswith(".csv"):
                    try:
                        upload.seek(0)
                        prefix = upload.read(65536).decode("utf-8-sig")
                        names = next(csv.reader(io.StringIO(prefix), strict=True), [])
                        header_names.extend(name for name in names if name not in header_names)
                        st.text(f"CSV columns in {upload.name}: {', '.join(names)[:2000]}")
                    except (UnicodeError, csv.Error):
                        st.warning("CSV header preview unavailable; batch mode requires UTF-8 CSV.")
                    finally:
                        upload.seek(0)
    else:
        st.text(f"Raw folder: {settings.paths['DATA_DIR'] / 'raw'}")
        st.caption(
            "Place original files here before ingestion. Nothing is scanned until you start."
        )
    csv_mode = st.selectbox(
        "CSV mode",
        ["rows", "file"],
        index=0 if config["csv"]["mode"] == "rows" else 1,
        format_func=lambda mode: (
            "Each row as document" if mode == "rows" else "File as one document"
        ),
    )
    if header_names:
        columns = st.multiselect(
            "CSV text columns (selection order is concatenation order)",
            header_names,
            default=[c for c in config["csv"]["text_columns"] if c in header_names],
        )
    else:
        columns = [
            c.strip()
            for c in st.text_input(
                "CSV text columns (comma-separated, in order)",
                value=", ".join(config["csv"]["text_columns"]),
            ).split(",")
            if c.strip()
        ]
    recursive = st.checkbox("Include subdirectories", value=config["dataset"]["recursive"])
    if st.button("Ingest Dataset", disabled=not eligible, type="primary"):
        bar = st.progress(0.0)
        message = st.empty()
        try:
            options = IngestionOptions.from_settings(
                settings, csv_mode=csv_mode, text_columns=columns, recursive=recursive
            )
            source = (
                stage_uploads(uploads, options)
                if source_mode == "Upload Files"
                else settings.paths["DATA_DIR"] / "raw"
            )
            output = options.output_path.with_name(
                f"{options.output_path.stem}-{uuid4().hex}.jsonl"
            )

            def progress(snapshot: dict) -> None:
                bar.progress(min(snapshot["records_written"] / options.limit, 1.0))
                message.text(
                    f"Examined {snapshot['records_written']:,} candidates · "
                    f"Successful {snapshot['successful']:,} · Failed {snapshot['failed']:,}"
                )

            result = ingest_directory(source, options, output=output, progress=progress)
            st.session_state["ingestion_result"] = result
            bar.progress(1.0)
            st.success("Ingestion completed. Raw inputs are preserved; no cleaning was performed.")
        except (ValueError, OSError) as exc:
            st.error(f"Cannot ingest dataset: {exc}")
    result = st.session_state.get("ingestion_result")
    if result is not None:
        st.subheader("Results")
        summary = result.summary
        st.write(
            {
                key: summary[key]
                for key in (
                    "files_examined",
                    "documents_created",
                    "records_written",
                    "successful",
                    "skipped",
                    "failed",
                    "no_extractable_text",
                    "unsupported",
                    "limit_reached",
                    "characters_extracted",
                    "source_bytes_examined",
                )
            }
        )
        st.text(f"JSONL: {result.output_path}\nManifest: {result.manifest_path}")
        st.caption("Raw size counts each file once. Character statistics include rejected records.")
        with st.expander("Character statistics and type distribution"):
            st.json(
                {
                    key: summary[key]
                    for key in (
                        "min_characters",
                        "max_characters",
                        "average_characters",
                        "file_types",
                        "document_types",
                        "statuses",
                    )
                }
            )
        st.subheader("Document preview")
        if result.preview:
            keys = (
                "document_id",
                "source_name",
                "source_type",
                "character_count",
                "file_size_bytes",
                "extraction_status",
            )
            st.dataframe(
                [{key: record[key] for key in keys} for record in result.preview], hide_index=True
            )
            selection = st.selectbox(
                "Preview document",
                range(len(result.preview)),
                format_func=lambda i: result.preview[i]["source_name"] + f" ({i + 1})",
            )
            st.text(result.preview[selection]["extracted_text"][: limits["preview_characters"]])
            st.caption(
                f"At most {limits['preview_documents']} records and "
                f"{limits['preview_characters']} characters per preview."
            )
