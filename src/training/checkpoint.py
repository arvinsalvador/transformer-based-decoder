"""Atomic state dictionaries, bounded retention and guarded best-weight export."""

import json
import os
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import torch


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return value


def save_checkpoint(directory, payload, *, archive=False, best=False, keep=3, interrupt=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = ["latest.pt"]
    if archive:
        names.insert(0, f"checkpoint_step_{payload['state']['step']:08d}.pt")
    if best:
        names.insert(0, "best.pt")
    if interrupt:
        names.insert(0, "interrupt.pt")
    # Complete serialization and all copies before touching existing checkpoints.
    with TemporaryDirectory(prefix=".checkpoint-", dir=directory) as stage_name:
        stage = Path(stage_name)
        atomic_torch(stage / names[0], payload)
        for name in names[1:]:
            shutil.copy2(stage / names[0], stage / name)
        for name in names:
            os.replace(stage / name, directory / name)
    archives = sorted(
        path
        for path in directory.iterdir()
        if re.fullmatch(r"checkpoint_step_[0-9]+\.pt", path.name)
    )
    for path in archives[:-keep]:
        path.unlink()


def export_best(directory, payload, manifest, overwrite=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = ("best_model.pt", "model_manifest.json")
    if any((directory / name).exists() for name in names) and not overwrite:
        raise FileExistsError("Export exists; use --overwrite-export or another MODEL_DIR")
    with TemporaryDirectory(prefix=".export-", dir=directory) as stage_name:
        stage = Path(stage_name)
        atomic_torch(stage / names[0], payload["model"])
        manifest = {**manifest, "model_file_size_bytes": (stage / names[0]).stat().st_size}
        atomic_json(stage / names[1], manifest)
        backups, published = [], []
        try:
            for name in names:
                target = directory / name
                if target.exists():
                    backup = stage / (name + ".previous")
                    shutil.copy2(target, backup)
                    backups.append((backup, target))
                os.replace(stage / name, target)
                published.append(target)
        except BaseException:
            for target in published:
                target.unlink()
            for backup, target in backups:
                os.replace(backup, target)
            raise
    return manifest
