"""Streaming count training with staged, rollback-protected artifact publication."""

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from src.config.settings import Settings
from src.tokenizer.corpus import texts
from src.tokenizer.service import load_tokenizer
from src.trigram.model import TrigramModel
from src.trigram.serialization import load_model, save_model
from src.trigram.sqlite_model import SQLiteTrigramModel

ARTIFACTS = ("trigram_counts.sqlite", "training_statistics.json", "trigram_manifest.json")


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(stage, target):
    """Restore the old artifact set on a caught publication failure.

    Each rename is atomic on the same filesystem. Exclusive writer access is
    required; abrupt process/OS failure is not a multi-file transaction.
    """
    backups, published = [], []
    try:
        for name in ARTIFACTS:
            destination = target / name
            if destination.exists():
                backup = stage / (name + ".previous")
                shutil.copy2(destination, backup)
                backups.append((backup, destination))
            os.replace(stage / name, destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            destination.unlink()
        for backup, destination in reversed(backups):
            os.replace(backup, destination)
        raise


def train(
    train_path,
    tokenizer_path,
    settings: Settings,
    *,
    output_dir=None,
    validation_path=None,
    dataset_fingerprint=None,
    overwrite=False,
    subset_manifest=None,
):
    source, token_dir = Path(train_path).resolve(), Path(tokenizer_path).resolve()
    subset = None
    if subset_manifest:
        from src.experiments.subsets import verify_subset

        subset = verify_subset(settings, source, subset_manifest)
    if source.name != "train.jsonl":
        raise ValueError("Training requires canonical train.jsonl")
    target = Path(output_dir or settings.paths["MODEL_DIR"] / "trigram").resolve()
    if settings.paths["MODEL_DIR"].resolve() not in target.parents:
        raise ValueError("Artifacts must be beneath MODEL_DIR")
    if target.exists() and any(p.name != ".gitkeep" for p in target.iterdir()) and not overwrite:
        raise FileExistsError("Trigram artifacts exist; use --overwrite")
    tokenizer_manifest = json.loads(
        (token_dir / "tokenizer_manifest.json").read_text(encoding="utf-8")
    )
    tokenizer = load_tokenizer(token_dir)
    if dataset_fingerprint and tokenizer_manifest.get("dataset_fingerprint") not in (
        None,
        dataset_fingerprint,
    ):
        raise ValueError("Tokenizer dataset fingerprint does not match requested dataset")
    special = tokenizer_manifest["special_tokens"]
    metadata = dict(
        vocabulary_size=tokenizer.get_vocab_size(),
        bos_id=special["bos_token_id"],
        eos_id=special["eos_token_id"],
        add_k=settings.values["trigram"]["add_k"],
    )
    backend = settings.values["trigram"]["storage_backend"]
    if backend == "auto":
        backend = "sqlite" if settings.values["environment"] == "gpu" else "memory"
    if backend not in ("sqlite", "memory"):
        raise ValueError("Unsupported trigram backend")
    target.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".trigram-stage-", dir=target) as directory:
        stage = Path(directory)
        artifact = stage / ARTIFACTS[0]
        model = (
            SQLiteTrigramModel(artifact, metadata)
            if backend == "sqlite"
            else TrigramModel(**metadata)
        )
        started_at, start = datetime.now(UTC).isoformat(), perf_counter()
        chars = documents = source_tokens = 0
        try:
            for text in texts(source, 1):
                documents += 1
                chars += len(text)
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                source_tokens += len(ids)
                model.update(ids)
            if not documents:
                raise ValueError("Training corpus is empty")
            stats = model.statistics()
            if backend == "memory":
                save_model(model, artifact)
        finally:
            model.close()
        duration, finished_at = perf_counter() - start, datetime.now(UTC).isoformat()
        with load_model(artifact) as ready:
            if ready.connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("SQLite validation failed")
            validation = (
                score(ready, tokenizer, validation_path, settings) if validation_path else None
            )
        manifest = {
            "model_type": "trigram",
            "tokenizer_fingerprint": tokenizer_manifest["tokenizer_fingerprint"],
            "dataset_fingerprint": dataset_fingerprint
            or tokenizer_manifest.get("dataset_fingerprint"),
            "training_split_fingerprint": fingerprint(source),
            "vocabulary_size": tokenizer.get_vocab_size(),
            "smoothing": "add_k",
            "add_k": metadata["add_k"],
            "storage_backend": backend,
            "training_documents": documents,
            "source_characters": chars,
            "source_tokens": source_tokens,
            **stats,
            "training_started_at": started_at,
            "training_finished_at": finished_at,
            "training_duration_seconds": duration,
            "model_artifact": ARTIFACTS[0],
            "model_artifact_size_bytes": artifact.stat().st_size,
            "project_version": "0.5.0",
            "validation": validation,
        }
        if subset:
            manifest.update(
                subset_fingerprint=subset["subset_fingerprint"],
                parent_train_fingerprint=subset["parent_train_fingerprint"],
            )
        (stage / ARTIFACTS[1]).write_text(
            json.dumps({"training": stats, "validation": validation}, indent=2), encoding="utf-8"
        )
        (stage / ARTIFACTS[2]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _publish(stage, target)
    return manifest


def score(model, tokenizer, split_path, settings: Settings):
    return model.score_sequences(
        tokenizer.encode(text, add_special_tokens=False).ids for text in texts(split_path, 1)
    ).__dict__
