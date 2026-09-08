"""Bounded Phase 4 tokenizer controls; training remains an explicit user action."""


import streamlit as st

from src.config.settings import Settings
from src.tokenizer.service import encode, inspect_vocabulary, load_tokenizer, train_wordpiece


def render_tokenizer(settings: Settings) -> None:
    st.title("Tokenizer")
    st.caption("Phase 4 · Train and inspect a genuine WordPiece vocabulary")
    st.info(
        "Vocabulary training uses only the canonical training split. "
        "Validation and test are analysis-only."
    )
    cfg = settings.values["tokenizer"]
    train = settings.paths["DATA_DIR"] / "splits/train.jsonl"
    validation = settings.paths["DATA_DIR"] / "splits/validation.jsonl"
    test = settings.paths["DATA_DIR"] / "splits/test.jsonl"
    st.write(
        {
            "train": str(train),
            "validation": str(validation),
            "test": str(test),
            "requested vocabulary": cfg["vocab_size"],
            "minimum frequency": cfg["min_frequency"],
            "prefix": cfg["continuing_subword_prefix"],
            "special tokens": cfg["special_tokens"],
        }
    )
    st.caption(
        "For the final corpus, run the batch command on the training server. "
        "Training is CPU-oriented."
    )
    overwrite = st.checkbox("Replace existing tokenizer artifacts", value=False)
    if st.button("Train WordPiece Tokenizer", disabled=not train.is_file(), type="primary"):
        try:
            result = train_wordpiece(
                train,
                settings,
                validation_path=validation if validation.is_file() else None,
                test_path=test if test.is_file() else None,
                overwrite=overwrite,
            )
            st.session_state["tokenizer_result"] = result
            st.success("Tokenizer training completed from the training split only.")
        except (ValueError, OSError, FileExistsError) as exc:
            st.error(f"Cannot train tokenizer: {exc}")
    result = st.session_state.get("tokenizer_result")
    artifact = result.output_dir if result else settings.paths["MODEL_DIR"] / "tokenizer"
    if not (artifact / "tokenizer.json").is_file():
        st.warning(
            "WordPiece module is ready; no tokenizer artifact has been trained at this location."
        )
        return
    tokenizer = load_tokenizer(artifact)
    manifest = result.manifest if result else {}
    st.subheader("Artifact")
    st.write(
        {
            "path": str(artifact),
            "vocabulary size": tokenizer.get_vocab_size(),
            "fingerprint": manifest.get(
                "tokenizer_fingerprint", "Read tokenizer_manifest.json for metadata"
            ),
        }
    )
    st.dataframe(
        [
            {"Token": token, "ID": token_id}
            for token, token_id in inspect_vocabulary(tokenizer, 100)
        ],
        hide_index=True,
    )
    sample = st.text_area("Encode text", value="Cybersecurity protects software systems.")
    special = st.checkbox("Add BOS/EOS", value=False)
    if sample:
        encoded = encode(tokenizer, sample, add_special_tokens=special, unk_token=cfg["unk_token"])
        st.write(
            {
                "tokens": encoded.tokens,
                "ids": encoded.ids,
                "unknown count": encoded.unknown_count,
                "decoded": tokenizer.decode(encoded.ids, skip_special_tokens=not special),
            }
        )
