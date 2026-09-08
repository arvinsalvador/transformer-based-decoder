"""Synchronous bounded development runs and a lightweight run-summary viewer."""

import copy
import json
from dataclasses import asdict, replace
from heapq import nlargest

import streamlit as st

from src.training.precision import select_precision
from src.training.trainer import TrainingFailure, run_training
from src.utils.device import DeviceUnavailableError, detect_device

MAX_UI_STEPS = 100


def render_training(settings):
    st.title("Transformer Training")
    st.info("Training engine implemented. Use CLI for long GPU jobs; browser runs are bounded.")
    try:
        import torch

        info = detect_device(settings.values["device"])
        precision = select_precision(
            torch.device(info.selected_device),
            settings.values["training"]["precision"],
            settings.values["training"]["mixed_precision"],
        )
        st.json({**asdict(info), "precision": precision})
    except (DeviceUnavailableError, ValueError) as exc:
        st.warning(str(exc))
    cfg = settings.values["training"]
    train = settings.paths["DATA_DIR"] / "splits/train.jsonl"
    validation = settings.paths["DATA_DIR"] / "splits/validation.jsonl"
    token_dir = settings.paths["MODEL_DIR"] / "tokenizer"
    st.write(
        {
            "train": str(train),
            "validation": str(validation),
            "tokenizer": str(token_dir),
            "context": settings.values["model"]["context_length"],
            "layers": settings.values["model"]["num_layers"],
            "heads": settings.values["model"]["num_heads"],
            "effective_batch": cfg["batch_size"] * cfg["gradient_accumulation_steps"],
        }
    )
    st.json(cfg)
    ready = all(path.is_file() for path in (train, validation, token_dir / "tokenizer.json"))
    steps = st.number_input("Maximum optimizer steps", min_value=1, max_value=MAX_UI_STEPS, value=5)
    overwrite = st.checkbox("Allow replacement of existing best-model export", value=False)
    dry = st.button("Run Dry Run", disabled=not ready)
    small = st.button("Start Small Training", disabled=not ready)
    st.caption(
        "Browser safety: batch ≤4, context ≤256, 0 workers, shuffle ≤100, "
        "accumulation ≤4, validation limited to 2 batches. Overrides stay in memory."
    )
    if dry or small:
        values = copy.deepcopy(settings.values)
        values["training"].update(
            num_workers=0,
            persistent_workers=False,
            batch_size=min(4, cfg["batch_size"]),
            shuffle_buffer_size=100,
            gradient_accumulation_steps=min(4, cfg["gradient_accumulation_steps"]),
        )
        if (
            values["model"]["context_length"] > 256
            or values["model"]["embedding_dim"] > 256
            or values["model"]["num_layers"] > 4
            or values["model"]["feedforward_dim"] > 1024
        ):
            st.error("Architecture exceeds browser safety limits; use the CLI.")
        else:
            try:
                with st.spinner("Running bounded training check"):
                    result = run_training(
                        replace(settings, values=values),
                        train,
                        validation,
                        token_dir,
                        max_steps=int(steps),
                        dry_run=dry,
                        overwrite_export=overwrite,
                        validation_batches=2,
                    )
                st.json(result)
            except TrainingFailure as exc:
                st.error(str(exc))
                st.json(exc.summary)
    st.code(
        "python scripts/train_transformer.py --train data/splits/train.jsonl "
        "--validation data/splits/validation.jsonl --tokenizer models/tokenizer "
        "--config config/gpu.yaml --dry-run",
        language="bash",
    )
    root = settings.paths["EXPERIMENT_DIR"] / "training"
    if root.exists():
        recent = nlargest(10, root.glob("*/summary.json"), key=lambda path: path.stat().st_mtime)
        with st.expander("Recent run summaries (at most 10)"):
            for path in recent:
                try:
                    st.json(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    st.warning(f"Cannot read summary: {path.name}")
