"""Cross-check aggregate counters against actual JSONL records."""

import json

from src.data.ingestion import ingest_directory


def test_statistics(corpus, options):
    (corpus / "a.txt").write_text("12345")
    (corpus / "b.txt").write_text("1234567")
    (corpus / "c.txt").write_text("")
    result = ingest_directory(corpus, options)
    records = [json.loads(line) for line in result.output_path.read_text().splitlines()]
    summary = result.summary
    assert summary["characters_extracted"] == sum(r["character_count"] for r in records) == 12
    assert summary["min_characters"] == 0
    assert summary["max_characters"] == 7
    assert summary["average_characters"] == 4
    assert summary["successful"] == 2
    assert summary["skipped"] == 1
    assert summary["file_types"] == {"txt": 3}
    assert (
        summary["successful"] + summary["failed"] + summary["skipped"] == summary["records_written"]
    )
