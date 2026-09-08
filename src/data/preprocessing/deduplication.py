"""Exact normalized-text deduplication; a bounded set stores hashes, never text."""

import hashlib


def normalized_hash(text: str) -> str:
    """Return stable SHA-256 of normalized UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ExactDeduplicator:
    """First accepted document wins deterministically according to input JSONL order."""

    def __init__(self) -> None:
        self._first_ids: dict[str, str] = {}

    def duplicate_of(self, digest: str, document_id: str) -> str | None:
        """Record a new digest or return the earlier accepted document id."""
        first = self._first_ids.get(digest)
        if first is None:
            self._first_ids[digest] = document_id
        return first

    @property
    def unique_count(self) -> int:
        return len(self._first_ids)
