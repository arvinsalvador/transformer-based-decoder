"""SQLite persistence and lazy read-only model loading."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from src.trigram.sqlite_model import SQLiteTrigramModel


def save_model(model, path, *, overwrite=False):
    """Publish a fully written database, preserving an existing model on failure."""
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError("Model exists; explicit overwrite is required")
    with TemporaryDirectory(prefix=".trigram-save-", dir=target.parent) as directory:
        staged = Path(directory) / target.name
        metadata = {
            name: getattr(model, name) for name in ("vocabulary_size", "bos_id", "eos_id", "add_k")
        }
        with SQLiteTrigramModel(staged, metadata) as output:
            with output.connection:
                output.connection.executemany(
                    "INSERT INTO unigram VALUES (?,?)", model.unigrams.items()
                )
                output.connection.executemany(
                    "INSERT INTO bigram VALUES (?,?,?)",
                    ((*key, count) for key, count in model.bigrams.items()),
                )
                output.connection.executemany(
                    "INSERT INTO trigram VALUES (?,?,?,?)",
                    ((*key, count) for key, count in model.trigrams.items()),
                )
        os.replace(staged, target)
    return target


def load_model(path):
    """Open disk tables read-only; use as a context manager or call close()."""
    return SQLiteTrigramModel(path)
