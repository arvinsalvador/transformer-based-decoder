"""Stable ranked locators, streamed subset writes, and independently verified provenance."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.evaluation.identity import read_json
from src.training.checkpoint import atomic_json
from src.training.reproducibility import file_hash


def ranked_locators(path, seed):
    entries = []
    with Path(path).open("rb") as stream:
        number = 0
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError("Invalid training record")
            identity = (
                record.get("document_id") or hashlib.sha256(record["text"].encode()).hexdigest()
            )
            score = hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()
            entries.append((score, number, offset))
            number += 1
    return sorted(entries)


def selected_lines(path, seed, count):
    entries = ranked_locators(path, seed)
    if count > len(entries) or count < 1:
        raise ValueError("Subset count exceeds canonical training count or is empty")
    with Path(path).open("rb") as stream:
        for _, _, offset in entries[:count]:
            stream.seek(offset)
            yield stream.readline().rstrip(b"\r\n") + b"\n"


def prepare_subset(settings, directory, scale, seed, parent_fingerprint):
    parent = settings.paths["DATA_DIR"] / "splits/train.jsonl"
    directory.mkdir(parents=True, exist_ok=True)
    path = parent if scale["scale"] == "full" else directory / "train.jsonl"
    manifest_path = directory / "subset_manifest.json"
    if manifest_path.exists():
        manifest = verify_subset(settings, path, manifest_path)
        if manifest["seed"] != seed or manifest["actual_count"] != scale["actual_train_documents"]:
            raise ValueError("Existing subset plan mismatch")
        return path, manifest_path, manifest
    if scale["scale"] != "full":
        temporary = None
        try:
            with NamedTemporaryFile("wb", dir=directory, delete=False) as output:
                temporary = output.name
                for line in selected_lines(parent, seed, scale["actual_train_documents"]):
                    output.write(line)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
    manifest = {
        "subset_path": str(path.resolve()),
        "parent_train_fingerprint": parent_fingerprint,
        "subset_fingerprint": file_hash(path),
        "seed": seed,
        "scale": scale["scale"],
        "requested_count": scale["requested_train_documents"],
        "actual_count": scale["actual_train_documents"],
        "ordering": "canonical bytes for FULL; SHA256(seed:document_id or text_hash), "
        "line-index tie break for subsets",
    }
    atomic_json(manifest_path, manifest)
    verify_subset(settings, path, manifest_path)
    return path, manifest_path, manifest


def verify_subset(settings, train, manifest_path):
    manifest = read_json(manifest_path)
    parent = settings.paths["DATA_DIR"] / "splits/train.jsonl"
    train = Path(train).resolve()
    if train != Path(manifest["subset_path"]).resolve():
        raise ValueError("Subset path mismatch")
    if file_hash(parent) != manifest["parent_train_fingerprint"]:
        raise ValueError("Subset parent train fingerprint mismatch")
    if file_hash(train) != manifest["subset_fingerprint"]:
        raise ValueError("Subset fingerprint mismatch")
    if manifest["scale"] == "full":
        if train != parent.resolve():
            raise ValueError("FULL must use canonical training file")
        count = len(ranked_locators(parent, manifest["seed"]))
        if count != manifest["actual_count"]:
            raise ValueError("FULL count mismatch")
    else:
        if settings.paths["EXPERIMENT_DIR"].resolve() not in train.parents:
            raise ValueError("Subset must be beneath EXPERIMENT_DIR")
        digest = hashlib.sha256()
        for line in selected_lines(parent, manifest["seed"], manifest["actual_count"]):
            digest.update(line)
        if digest.hexdigest() != manifest["subset_fingerprint"]:
            raise ValueError("Subset does not match deterministic canonical selection")
    return manifest
