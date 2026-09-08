"""Canonical artifact identity and historical provenance checks without writes."""

import hashlib
import json
import math
from pathlib import Path

from src.tokenizer.service import load_tokenizer
from src.training.reproducibility import file_hash


class FingerprintError(ValueError):
    pass


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Manifest must be an object: {path}")
    return value


def verify(settings, test, token_dir, manifests, dataset_manifest=None):
    test, token_dir = Path(test).resolve(), Path(token_dir).resolve()
    if test != (settings.paths["DATA_DIR"] / "splits/test.jsonl").resolve():
        raise FingerprintError("Use canonical DATA_DIR/splits/test.jsonl")
    if test in {
        (settings.paths["DATA_DIR"] / f"splits/{name}.jsonl").resolve()
        for name in ("train", "validation")
    }:
        raise FingerprintError("Test must be distinct from train and validation")
    if token_dir != (settings.paths["MODEL_DIR"] / "tokenizer").resolve():
        raise FingerprintError("Use canonical MODEL_DIR/tokenizer")
    token_manifest = read_json(token_dir / "tokenizer_manifest.json")
    digest = file_hash(token_dir / "tokenizer.json")
    tokenizer = load_tokenizer(token_dir)
    if (
        token_manifest.get("tokenizer_type") != "WordPiece"
        or token_manifest.get("actual_vocabulary_size") != tokenizer.get_vocab_size()
    ):
        raise FingerprintError("Canonical WordPiece vocabulary mismatch")
    special = token_manifest["special_tokens"]
    for name in ("pad_token", "unk_token", "bos_token", "eos_token"):
        if tokenizer.token_to_id(special[name]) != special[name + "_id"]:
            raise FingerprintError("Tokenizer special ID mismatch")
    for manifest in [token_manifest, *manifests.values()]:
        if manifest.get("tokenizer_fingerprint") != digest:
            raise FingerprintError("Tokenizer fingerprint mismatch")
        if manifest.get("dataset_fingerprint_source") == "train_validation_pair":
            pair = {
                key: manifest.get(key) for key in ("train_fingerprint", "validation_fingerprint")
            }
            expected_pair = hashlib.sha256(json.dumps(pair, sort_keys=True).encode()).hexdigest()
            if not all(pair.values()) or manifest.get("dataset_fingerprint") != expected_pair:
                raise FingerprintError("Dataset train/validation pair fingerprint mismatch")
    corpus = {
        m["dataset_fingerprint"]
        for m in [token_manifest, *manifests.values()]
        if m.get("dataset_fingerprint")
        and m.get("dataset_fingerprint_source") != "train_validation_pair"
    }
    train_hashes = {
        m.get("training_split_fingerprint") or m.get("train_fingerprint")
        for m in [token_manifest, *manifests.values()]
    }
    train_hashes.discard(None)
    if len(corpus) > 1 or len(train_hashes) > 1:
        raise FingerprintError("Dataset/training-split fingerprint mismatch")
    test_hash = file_hash(test)
    warnings = []
    historical_test_verified = False
    if dataset_manifest:
        dataset = read_json(dataset_manifest)
        if dataset.get("state") != "completed":
            raise FingerprintError("Dataset manifest is not completed")
        if corpus and dataset.get("dataset_fingerprint") not in corpus:
            raise FingerprintError("Dataset fingerprint mismatch")
        expected = dataset.get("split_fingerprints", {}).get("test")
        if expected and expected != test_hash:
            raise FingerprintError("Test split fingerprint mismatch")
        historical_test_verified = bool(expected)
        if dataset.get("dataset_fingerprint"):
            corpus.add(dataset["dataset_fingerprint"])
        expected_train = dataset.get("split_fingerprints", {}).get("train")
        if expected_train and train_hashes and expected_train not in train_hashes:
            raise FingerprintError("Dataset train fingerprint mismatch")
        expected_validation = dataset.get("split_fingerprints", {}).get("validation")
        actual_validation = manifests.get("transformer", {}).get("validation_fingerprint")
        if expected_validation and actual_validation and expected_validation != actual_validation:
            raise FingerprintError("Dataset validation fingerprint mismatch")
    if not historical_test_verified:
        warnings.append(
            "Historical test bytes are unverified: supply a completed preprocessing manifest "
            "with split hashes. Current bytes are hashed and rechecked."
        )
    if not corpus:
        warnings.append(
            "Whole-corpus provenance unavailable; current test and available "
            "training-source hashes are recorded."
        )
    for name, manifest in manifests.items():
        if manifest.get("vocabulary_size") != tokenizer.get_vocab_size():
            raise FingerprintError(f"{name} vocabulary mismatch")
    return (
        tokenizer,
        special,
        {
            "tokenizer_fingerprint": digest,
            "test_fingerprint": test_hash,
            "dataset_fingerprint": next(iter(corpus), None),
            "test_dataset_path": str(test),
            "historical_test_verified": historical_test_verified,
            "warnings": warnings,
        },
    )


def training_metadata(manifest):
    keys = (
        "training_duration_seconds",
        "peak_training_ram_mb",
        "peak_training_vram_mb",
        "training_tokens_per_second",
        "training_valid_tokens",
        "optimizer_steps",
    )
    values, reasons = {}, {}
    for key in keys:
        value = manifest.get(key)
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
        ):
            raise ValueError(f"Malformed training metric: {key}")
        values[key] = value
        if value is None:
            reasons[key] = "Not recorded in the model manifest; not estimated after the fact"
    return {**values, "missing_metric_reasons": reasons}
