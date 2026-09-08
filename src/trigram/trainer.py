"""Streaming train/split scoring orchestration using the canonical tokenizer."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from src.config.settings import Settings
from src.tokenizer.corpus import texts
from src.tokenizer.service import load_tokenizer
from src.trigram.model import TrigramModel
from src.trigram.serialization import save_model


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(directory: Path) -> dict:
    return json.loads((directory / "tokenizer_manifest.json").read_text(encoding="utf-8"))


def train(
    train_path,
    tokenizer_path,
    settings: Settings,
    *,
    output_dir=None,
    validation_path=None,
    dataset_fingerprint=None,
    overwrite=False,
) -> dict:
    source, token_dir = Path(train_path).resolve(), Path(tokenizer_path).resolve()
    if source.name != "train.jsonl":
        raise ValueError("Training requires canonical train.jsonl")
    target = Path(output_dir or settings.paths["MODEL_DIR"] / "trigram").resolve()
    if (
        target != settings.paths["MODEL_DIR"].resolve()
        and settings.paths["MODEL_DIR"].resolve() not in target.parents
    ):
        raise ValueError("Artifacts must be beneath MODEL_DIR")
    if target.exists() and any(p.name != ".gitkeep" for p in target.iterdir()) and not overwrite:
        raise FileExistsError("Trigram artifacts exist; use --overwrite")
    target.mkdir(parents=True, exist_ok=True)
    tokenizer_manifest, tokenizer = _manifest(token_dir), load_tokenizer(token_dir)
    if dataset_fingerprint and tokenizer_manifest.get("dataset_fingerprint") not in (
        None,
        dataset_fingerprint,
    ):
        raise ValueError("Tokenizer dataset fingerprint does not match requested dataset")
    special = tokenizer_manifest["special_tokens"]
    model = TrigramModel(
        tokenizer.get_vocab_size(),
        special["bos_token_id"],
        special["eos_token_id"],
        settings.values["trigram"]["add_k"],
    )
    started_at, start, chars, documents = datetime.now(UTC).isoformat(), perf_counter(), 0, 0
    for text in texts(source, settings.values["tokenizer"]["training_batch_documents"]):
        documents, chars = documents + 1, chars + len(text)
        model.update(tokenizer.encode(text, add_special_tokens=False).ids)
    artifact = target / "trigram_counts.sqlite"
    save_model(model, artifact)
    stats = model.statistics()
    validation = score(model, tokenizer, validation_path, settings) if validation_path else None
    manifest = {
        "model_type": "trigram",
        "tokenizer_fingerprint": tokenizer_manifest["tokenizer_fingerprint"],
        "dataset_fingerprint": dataset_fingerprint or tokenizer_manifest.get("dataset_fingerprint"),
        "training_split_fingerprint": fingerprint(source),
        "vocabulary_size": tokenizer.get_vocab_size(),
        "smoothing": "add_k",
        "add_k": model.add_k,
        "storage_backend": settings.values["trigram"]["storage_backend"],
        "training_documents": documents,
        "source_characters": chars,
        **stats,
        "training_started_at": started_at,
        "training_finished_at": datetime.now(UTC).isoformat(),
        "training_duration_seconds": perf_counter() - start,
        "model_artifact": artifact.name,
        "model_artifact_size_bytes": artifact.stat().st_size,
        "project_version": "0.5.0",
        "validation": validation,
    }
    (target / "training_statistics.json").write_text(
        json.dumps({"training": stats, "validation": validation}, indent=2), encoding="utf-8"
    )
    (target / "trigram_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def score(model, tokenizer, split_path, settings: Settings) -> dict:
    result = model.score_sequences(
        tokenizer.encode(text, add_special_tokens=False).ids
        for text in texts(split_path, settings.values["tokenizer"]["training_batch_documents"])
    )
    return result.__dict__
