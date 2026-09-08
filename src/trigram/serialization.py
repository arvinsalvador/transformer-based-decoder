"""Safe SQLite serialization for memory and SQLite counting backends."""

import json
import sqlite3
from pathlib import Path

from src.trigram.model import TrigramModel


def save_model(model: TrigramModel, path: str | Path) -> Path:
    target = Path(path)
    connection = sqlite3.connect(target)
    try:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            "CREATE TABLE unigram (token INTEGER PRIMARY KEY, count INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE bigram (a INTEGER, b INTEGER, count INTEGER NOT NULL, PRIMARY KEY(a,b))"
        )
        connection.execute(
            "CREATE TABLE trigram (a INTEGER, b INTEGER, c INTEGER, count INTEGER NOT NULL, "
            "PRIMARY KEY(a,b,c))"
        )
        metadata = {
            "vocabulary_size": model.vocabulary_size,
            "bos_id": model.bos_id,
            "eos_id": model.eos_id,
            "add_k": model.add_k,
        }
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [(key, json.dumps(value)) for key, value in metadata.items()],
        )
        connection.executemany("INSERT INTO unigram VALUES (?, ?)", model.unigrams.items())
        connection.executemany(
            "INSERT INTO bigram VALUES (?, ?, ?)",
            [(a, b, count) for (a, b), count in model.bigrams.items()],
        )
        connection.executemany(
            "INSERT INTO trigram VALUES (?, ?, ?, ?)",
            [(a, b, c, count) for (a, b, c), count in model.trigrams.items()],
        )
        connection.execute("CREATE INDEX trigram_context ON trigram(a,b)")
        connection.commit()
    finally:
        connection.close()
    return target


def load_model(path: str | Path) -> TrigramModel:
    connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    try:
        values = {
            key: json.loads(value)
            for key, value in connection.execute("SELECT key, value FROM metadata")
        }
        model = TrigramModel(
            values["vocabulary_size"], values["bos_id"], values["eos_id"], values["add_k"]
        )
        model.unigrams.update(dict(connection.execute("SELECT token, count FROM unigram")))
        model.bigrams.update(
            {(a, b): count for a, b, count in connection.execute("SELECT a,b,count FROM bigram")}
        )
        model.trigrams.update(
            {
                (a, b, c): count
                for a, b, c, count in connection.execute("SELECT a,b,c,count FROM trigram")
            }
        )
        return model
    finally:
        connection.close()
