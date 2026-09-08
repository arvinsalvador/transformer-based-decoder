"""Deterministic directory traversal without following symlink directories."""

import os
from collections.abc import Iterator
from pathlib import Path


def discover_files(root: Path, recursive: bool) -> Iterator[Path]:
    """Sort one directory at a time, not all paths in the entire corpus.

    Symlinks and inaccessible subdirectories are emitted for controlled rejection.
    A concurrent hostile filesystem writer is outside this local-storage trust model.
    """
    errors = []
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=errors.append):
        for error in errors:
            yield Path(error.filename)
        errors.clear()
        dirs.sort()
        files.sort()
        for name in list(dirs):
            path = Path(directory) / name
            if path.is_symlink():
                dirs.remove(name)
                yield path
        for name in files:
            if name != ".gitkeep":
                yield Path(directory) / name
        if not recursive:
            dirs.clear()
    for error in errors:
        yield Path(error.filename)
