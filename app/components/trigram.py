"""Artifact-backed Phase 5 trigram controls."""

import json

import streamlit as st

from src.tokenizer.service import load_tokenizer
from src.trigram.serialization import load_model
from src.trigram.trainer import train


def render_trigram(settings):
    st.title("Trigram Language Model")
    st.caption("Phase 5 · CPU-only WordPiece-ID baseline")
    st.info("Training reads only train.jsonl. Validation/test scoring never updates counts.")
    data, token_dir, model_dir = (
        settings.paths["DATA_DIR"],
        settings.paths["MODEL_DIR"] / "tokenizer",
        settings.paths["MODEL_DIR"] / "trigram",
    )
    train_path, validation = data / "splits/train.jsonl", data / "splits/validation.jsonl"
    st.write(
        {
            "Tokenizer": "Available" if (token_dir / "tokenizer.json").is_file() else "Missing",
            "Model artifact": "Trained"
            if (model_dir / "trigram_manifest.json").is_file()
            else "Not trained",
            "backend": settings.values["trigram"]["storage_backend"],
            "smoothing": f"add-k ({settings.values['trigram']['add_k']})",
        }
    )
    if st.button(
        "Train Trigram Model",
        disabled=not train_path.is_file() or not (token_dir / "tokenizer.json").is_file(),
    ):
        try:
            st.json(
                train(
                    train_path,
                    token_dir,
                    settings,
                    validation_path=validation if validation.is_file() else None,
                )
            )
        except (OSError, ValueError, FileExistsError) as exc:
            st.error(str(exc))
    if not (model_dir / "trigram_counts.sqlite").is_file():
        st.warning(
            "Trigram module: Ready. Model artifact: Not trained. "
            "Use the CLI for final large-corpus training."
        )
        return
    manifest = json.loads((model_dir / "trigram_manifest.json").read_text())
    st.json(manifest)
    prompt = st.text_area("Prompt", "Artificial intelligence can")
    if st.button("Generate"):
        tokenizer = load_tokenizer(token_dir)
        with load_model(model_dir / "trigram_counts.sqlite") as model:
            ids, metrics = model.generate(
                tokenizer.encode(prompt, add_special_tokens=False).ids,
                max_new_tokens=50,
                seed=settings.values["random_seed"],
            )
        st.write({"text": tokenizer.decode(ids, skip_special_tokens=True), **metrics})
