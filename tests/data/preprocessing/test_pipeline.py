"""End-to-end Phase 2-style JSONL → clean corpus → canonical split files."""

import json
import tracemalloc

import pytest

from src.data.preprocessing.pipeline import prepare_dataset
from tests.data.preprocessing.conftest import raw_record


def write_raw(path, records, invalid_line=False):
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
        if invalid_line:
            stream.write("{bad json}\n")


def test_end_to_end_deduplicate_then_split(tmp_path, prep_settings):
    source = tmp_path / "documents.jsonl"
    write_raw(
        source,
        [
            raw_record("one", "Artificial intelligence is changing software development."),
            raw_record("two", "Artificial   intelligence is changing software development."),
            raw_record("three", "Cybersecurity protects systems from attacks."),
            raw_record("four", "Machine learning models learn patterns from data."),
            raw_record("five", "   "),
        ],
    )
    result = prepare_dataset(source, prep_settings)
    clean = [json.loads(line) for line in result.clean_output.read_text().splitlines()]
    rejected = [json.loads(line) for line in result.rejection_output.read_text().splitlines()]
    assert [item["document_id"] for item in clean] == ["one", "three", "four"]
    duplicate = next(item for item in rejected if item["preprocessing_status"] == "DUPLICATE")
    assert duplicate["document_id"] == "two" and duplicate["duplicate_of"] == "one"
    split_ids = []
    for path in result.split_paths.values():
        split_ids.extend(
            item["document_id"] for item in map(json.loads, path.read_text().splitlines())
        )
    assert set(split_ids) == {"one", "three", "four"}
    assert len(split_ids) == len(set(split_ids))
    assert result.summary["duplicates"] == 1
    assert result.summary["rejected_documents"] == 2


def test_invalid_records_and_extraction_failures_continue(tmp_path, prep_settings):
    source = tmp_path / "documents.jsonl"
    write_raw(
        source,
        [raw_record("bad", "text", status="PARSE_ERROR"), raw_record("good", "Useful text")],
        True,
    )
    result = prepare_dataset(source, prep_settings)
    assert result.summary["accepted_documents"] == 1
    assert result.summary["rejection_reason_distribution"]["INVALID_RECORD"] == 1
    assert result.summary["rejection_reason_distribution"]["EXTRACTION_NOT_ACCEPTED"] == 1


@pytest.mark.parametrize(
    ("policy", "accepted", "reason"), [("truncate", 1, None), ("skip", 0, "TOO_LONG_SKIPPED")]
)
def test_long_policy(tmp_path, prep_settings, policy, accepted, reason):
    prep_settings.values["preprocessing"].update(
        {"max_characters_per_document": 10, "min_characters": 1, "long_document_policy": policy}
    )
    source = tmp_path / "documents.jsonl"
    write_raw(source, [raw_record("long", "Very long useful text")])
    result = prepare_dataset(source, prep_settings)
    assert result.summary["accepted_documents"] == accepted
    if reason:
        assert result.summary["rejection_reason_distribution"][reason] == 1
    else:
        record = json.loads(result.clean_output.read_text())
        assert record["truncated"] and record["character_count"] == 10


def test_reproducible_split_and_no_overwrite(tmp_path, prep_settings):
    source = tmp_path / "documents.jsonl"
    write_raw(
        source,
        [
            raw_record(str(index), f"Useful technical document number {index}")
            for index in range(30)
        ],
    )
    first = prepare_dataset(
        source, prep_settings, clean_output=tmp_path / "one.jsonl", split_dir=tmp_path / "one"
    )
    second = prepare_dataset(
        source, prep_settings, clean_output=tmp_path / "two.jsonl", split_dir=tmp_path / "two"
    )
    assert first.summary["dataset_fingerprint"] == second.summary["dataset_fingerprint"]
    assert {name: path.read_text() for name, path in first.split_paths.items()} == {
        name: path.read_text() for name, path in second.split_paths.items()
    }
    with pytest.raises(FileExistsError):
        prepare_dataset(
            source, prep_settings, clean_output=tmp_path / "one.jsonl", split_dir=tmp_path / "one"
        )


def test_limit_and_incremental_memory(tmp_path, prep_settings):
    source = tmp_path / "documents.jsonl"
    write_raw(
        source,
        [
            raw_record(str(index), f"Technical document content {index} " * 5)
            for index in range(1500)
        ],
    )
    prep_settings.values["dataset"]["working_document_limit"] = 700
    points = []

    def progress(snapshot):
        if snapshot["input_documents"] in (100, 700):
            points.append(tracemalloc.get_traced_memory()[0])

    tracemalloc.start()
    try:
        result = prepare_dataset(source, prep_settings, progress=progress)
    finally:
        tracemalloc.stop()
    assert result.summary["input_documents"] == 700
    assert len(result.preview) == 20
    assert points[-1] - points[0] < 2 * 1024**2
