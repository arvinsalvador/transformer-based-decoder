"""Run fingerprints and safely serializable RNG state."""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from src.tokenizer.service import load_tokenizer


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed, deterministic):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def rng_state():
    name, values, position, gaussian, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [name, torch.tensor(values.astype(np.int64)), position, gaussian, cached],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    name, values, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, values.cpu().numpy().astype(np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def verify_inputs(train, validation, token_dir, settings, dataset_manifest=None):
    """Never open test.jsonl. Bind resumes to actual train and validation bytes."""
    root = settings.paths["DATA_DIR"].resolve() / "splits"
    for name, supplied in (("train", train), ("validation", validation)):
        if Path(supplied).resolve() != (root / f"{name}.jsonl").resolve():
            raise ValueError(f"Use canonical DATA_DIR/splits/{name}.jsonl")
    if Path(train).resolve() == Path(validation).resolve():
        raise ValueError("Train and validation must be distinct")
    directory = Path(token_dir).resolve()
    if directory != (settings.paths["MODEL_DIR"] / "tokenizer").resolve():
        raise ValueError("Use the canonical MODEL_DIR/tokenizer")
    manifest = json.loads((directory / "tokenizer_manifest.json").read_text(encoding="utf-8"))
    digest = file_hash(directory / "tokenizer.json")
    if manifest.get("tokenizer_fingerprint") != digest:
        raise ValueError("Tokenizer fingerprint mismatch")
    tokenizer = load_tokenizer(directory)
    if manifest.get("tokenizer_type") != "WordPiece":
        raise ValueError("Canonical tokenizer must be WordPiece")
    if manifest["actual_vocabulary_size"] != tokenizer.get_vocab_size():
        raise ValueError("Tokenizer vocabulary size mismatch")
    special = manifest["special_tokens"]
    for name in ("pad_token", "unk_token", "bos_token", "eos_token"):
        if tokenizer.token_to_id(special[name]) != special[name + "_id"]:
            raise ValueError(f"Tokenizer special ID mismatch: {name}")
    hashes = {
        "train_fingerprint": file_hash(train),
        "validation_fingerprint": file_hash(validation),
    }
    declared = manifest.get("dataset_fingerprint")
    warnings = []
    train_digest = manifest.get("training_split_fingerprint")
    if train_digest and train_digest != hashes["train_fingerprint"]:
        raise ValueError("Tokenizer training split fingerprint mismatch")
    if not train_digest:
        warnings.append(
            "Legacy tokenizer has no train-split hash; "
            "original fitting provenance cannot be verified."
        )
    if dataset_manifest:
        dataset = json.loads(Path(dataset_manifest).read_text(encoding="utf-8"))
        if dataset.get("state") != "completed":
            raise ValueError("Dataset manifest must be completed")
        if declared and declared != dataset.get("dataset_fingerprint"):
            raise ValueError("Dataset fingerprint mismatch")
        declared = dataset["dataset_fingerprint"]
        for name in ("train", "validation"):
            expected = dataset.get("split_fingerprints", {}).get(name)
            if expected and expected != hashes[name + "_fingerprint"]:
                raise ValueError(f"Dataset {name} fingerprint mismatch")
    pair = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return (
        special,
        {
            **hashes,
            "dataset_fingerprint": declared or pair,
            "dataset_fingerprint_source": "declared_corpus"
            if declared
            else "train_validation_pair",
            "tokenizer_fingerprint": digest,
            "vocabulary_size": tokenizer.get_vocab_size(),
        },
        warnings,
    )
