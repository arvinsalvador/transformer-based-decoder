"""Live environment diagnostics and honest project status."""

from dataclasses import asdict

import psutil
import streamlit as st

from src.config.settings import Settings
from src.utils.device import DeviceUnavailableError, detect_device


def render_dashboard(settings: Settings) -> None:
    """Render configuration and hardware without loading any dataset or model."""
    config = settings.values
    st.title("Transformer-Based Decoder-Only Language Model")
    st.caption("Phase 7 · WordPiece baseline and Transformer training engine")
    st.info(
        "Phases 1–7 are implemented. Shared evaluation and final corpus experiments remain pending."
    )
    st.markdown(
        "Build a decoder-only Transformer with a **WordPiece tokenizer**, using **up to "
        "100,000 documents**. Compare performance and training time with a trigram language model."
    )
    st.subheader("Environment")
    try:
        info = detect_device(config["device"])
        selected = info.selected_device.upper()
    except DeviceUnavailableError as exc:
        st.warning(str(exc))
        info = detect_device("cpu")
        selected = "Unavailable (CUDA required)"
        st.caption("CPU diagnostics shown below; the configured CUDA selection has failed.")
    if config["environment"] == "gpu" and not info.cuda_available:
        st.warning("GPU profile is active without CUDA. Final GPU experiments cannot run here.")
    columns = st.columns(4)
    for column, label, value in zip(
        columns,
        ["Profile", "Selected device", "Documents", "Batch size"],
        [
            config["environment"],
            selected,
            config["dataset"]["working_document_limit"],
            config["training"]["batch_size"],
        ],
        strict=True,
    ):
        column.metric(label, value)
    st.write(
        {
            "Configuration": str(settings.config_path),
            "GPU": info.gpu_name or "None detected",
            "PyTorch": info.pytorch_version,
            "CUDA available": info.cuda_available,
            "Context length": config["model"]["context_length"],
            "Epochs": config["training"]["epochs"],
            "System RAM (GiB)": round(psutil.virtual_memory().total / 2**30, 1),
        }
    )
    st.subheader("Planned workflow")
    st.code(
        "Documents → Preprocessing → WordPiece\n"
        "                               ├─ Trigram ─────┐\n"
        "                               └─ Transformer ┤\n"
        "                                  Evaluation → Comparison",
        language=None,
    )
    st.caption(
        "100,000 is a maximum. Start with small local subsets; "
        "WordPiece analysis reports token counts."
    )
    with st.expander("Hardware details and active configuration"):
        st.json(asdict(info))
        st.json(config)
        st.write({key: str(path) for key, path in settings.paths.items()})
