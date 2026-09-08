"""Bounded SQLite writes and indexed inference, without full Python count tables."""

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path

from src.trigram.model import TrigramModel


class CountTable(Mapping):
    """Read-only mapping facade; SQL identifiers come only from fixed internal schemas."""

    def __init__(self, connection, table, columns):
        self.connection, self.table, self.columns = connection, table, columns

    def __getitem__(self, key):
        keys = key if isinstance(key, tuple) else (key,)
        where = " AND ".join(f"{column}=?" for column in self.columns)
        row = self.connection.execute(
            f"SELECT count FROM {self.table} WHERE {where}", keys
        ).fetchone()
        return row[0] if row else 0

    def __len__(self):
        return self.connection.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]

    def __iter__(self):
        for key, _ in self.items():
            yield key

    def items(self):
        for *keys, count in self.connection.execute(
            f"SELECT {','.join(self.columns)}, count FROM {self.table}"
        ):
            yield (keys[0] if len(keys) == 1 else tuple(keys)), count

    def total(self):
        return self.connection.execute(
            f"SELECT COALESCE(SUM(count),0) FROM {self.table}"
        ).fetchone()[0]


class SQLiteTrigramModel(TrigramModel):
    """The shared scoring/generation implementation queries this model's disk tables."""

    def __init__(self, path, metadata=None, *, batch_size=4096):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.path = Path(path).resolve()
        self.writable = metadata is not None
        self.connection = sqlite3.connect(
            self.path.as_uri() + ("?mode=rwc" if self.writable else "?mode=ro"), uri=True
        )
        self.batch_size, self.flushes, self.max_buffered_updates = batch_size, 0, 0
        self.buffers = [[], [], []]
        try:
            self.connection.execute("PRAGMA cache_size=-4096")
            if metadata is not None:
                with self.connection:
                    self.connection.execute(
                        "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)"
                    )
                    self.connection.execute(
                        "CREATE TABLE unigram(token INTEGER PRIMARY KEY,count INTEGER)"
                    )
                    self.connection.execute(
                        "CREATE TABLE bigram(a INTEGER,b INTEGER,count INTEGER,PRIMARY KEY(a,b))"
                    )
                    self.connection.execute(
                        "CREATE TABLE trigram(a INTEGER,b INTEGER,c INTEGER,count INTEGER,"
                        "PRIMARY KEY(a,b,c))"
                    )
                    self.connection.executemany(
                        "INSERT INTO metadata VALUES (?,?)",
                        ((k, json.dumps(v)) for k, v in metadata.items()),
                    )
            else:
                metadata = {
                    k: json.loads(v)
                    for k, v in self.connection.execute("SELECT key,value FROM metadata")
                }
            for name in ("vocabulary_size", "bos_id", "eos_id", "add_k"):
                setattr(self, name, metadata[name])
            self.unigrams = CountTable(self.connection, "unigram", ("token",))
            self.bigrams = CountTable(self.connection, "bigram", ("a", "b"))
            self.trigrams = CountTable(self.connection, "trigram", ("a", "b", "c"))
        except BaseException:
            self.connection.close()
            raise

    def _append(self, table, row):
        self.buffers[table].append(row)
        size = sum(map(len, self.buffers))
        self.max_buffered_updates = max(size, self.max_buffered_updates)
        if size >= self.batch_size:
            self.flush()

    def update(self, token_ids):
        if not self.writable:
            raise ValueError("Loaded SQLite model is read-only")
        previous = []
        # Iterate the document plus markers without making another full sequence.
        from itertools import chain

        for token in chain((self.bos_id, self.bos_id), token_ids, (self.eos_id,)):
            self._append(0, (token, 1))
            if previous:
                self._append(1, (previous[-1], token, 1))
            if len(previous) == 2:
                self._append(2, (*previous, token, 1))
            previous = [*previous[-1:], token]
        self.flush()
        return len(token_ids) + 1

    def flush(self):
        if not any(self.buffers):
            return
        with self.connection:
            self.connection.executemany(
                "INSERT INTO unigram VALUES (?,?) ON CONFLICT(token) "
                "DO UPDATE SET count=count+excluded.count",
                self.buffers[0],
            )
            self.connection.executemany(
                "INSERT INTO bigram VALUES (?,?,?) ON CONFLICT(a,b) "
                "DO UPDATE SET count=count+excluded.count",
                self.buffers[1],
            )
            self.connection.executemany(
                "INSERT INTO trigram VALUES (?,?,?,?) ON CONFLICT(a,b,c) "
                "DO UPDATE SET count=count+excluded.count",
                self.buffers[2],
            )
        self.buffers = [[], [], []]
        self.flushes += 1

    def candidates(self, first, second):
        rows = self.connection.execute(
            "SELECT c,count FROM trigram WHERE a=? AND b=? ORDER BY c", (first, second)
        ).fetchall()
        if rows:
            denominator = self.bigrams[first, second] + self.add_k * self.vocabulary_size
            return [(token, (count + self.add_k) / denominator) for token, count in rows]
        total = self.unigrams.total() or 1
        return [
            (token, count / total)
            for token, count in self.connection.execute(
                "SELECT token,count FROM unigram WHERE token != ? ORDER BY token", (self.bos_id,)
            )
        ]

    def statistics(self):
        return {
            "unique_unigrams": len(self.unigrams),
            "unique_bigrams": len(self.bigrams),
            "unique_trigrams": len(self.trigrams),
            "tokens_observed": self.unigrams.total(),
            "trigram_events": self.trigrams.total(),
        }

    def close(self):
        self.connection.close()
