"""Exact deduplication stores digests and preserves the first accepted id."""

from src.data.preprocessing.deduplication import ExactDeduplicator, normalized_hash


def test_exact_hash_deduplication():
    deduplicator = ExactDeduplicator()
    digest = normalized_hash("Hello world")
    assert deduplicator.duplicate_of(digest, "first") is None
    assert deduplicator.duplicate_of(digest, "second") == "first"
    assert deduplicator.unique_count == 1
