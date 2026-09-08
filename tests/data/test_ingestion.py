"""Integration, output integrity, failure continuation and bounded memory."""

import json
import subprocess
import sys
import tracemalloc
from dataclasses import replace

import pytest

from src.config.settings import PROJECT_ROOT
from src.data.ingestion import ingest_directory
from src.data.models import ParsedText


def test_mixed_limit_jsonl_manifest(corpus, options):
    (corpus / "a.txt").write_text("Alpha")
    (corpus / "b.csv").write_text("text\nOne\nTwo\nThree\n")
    (corpus / "c.txt").write_text("Never reached")
    result = ingest_directory(corpus, replace(options, limit=3, text_columns=("text",)))
    records = [json.loads(line) for line in result.output_path.read_text().splitlines()]
    assert [record["extracted_text"] for record in records] == ["Alpha", "One", "Two"]
    assert len({r["document_id"] for r in records}) == 3
    assert all(len(r["sha256"]) == 64 for r in records)
    assert result.summary["files_examined"] == 2
    assert result.summary["limit_reached"] is True
    assert json.loads(result.manifest_path.read_text())["records_written"] == 3
    assert (
        result.summary["source_bytes_examined"]
        == (corpus / "a.txt").stat().st_size + (corpus / "b.csv").stat().st_size
    )


def test_failure_continues_and_logs_no_text(corpus, options, caplog):
    (corpus / "a.pdf").write_text("SENSITIVE CONTENT not pdf")
    (corpus / "b.txt").write_text("Working text")
    result = ingest_directory(corpus, options)
    assert result.summary["failed"] == 1
    assert result.summary["successful"] == 1
    assert "SENSITIVE CONTENT" not in caplog.text


def test_recursion(corpus, options):
    (corpus / "child").mkdir()
    (corpus / "child" / "a.txt").write_text("nested")
    first = ingest_directory(corpus, replace(options, recursive=False))
    assert first.summary["records_written"] == 0
    second = ingest_directory(
        corpus, options, output=first.output_path.with_name("recursive.jsonl")
    )
    assert second.summary["successful"] == 1


def test_existing_output_and_source_protected(corpus, options):
    (corpus / "a.txt").write_text("original")
    with pytest.raises(ValueError, match="outside"):
        ingest_directory(corpus, options, output=corpus / "new.jsonl")
    result = ingest_directory(corpus, options)
    content = result.output_path.read_bytes()
    with pytest.raises(FileExistsError):
        ingest_directory(corpus, options)
    assert result.output_path.read_bytes() == content
    assert (corpus / "a.txt").read_text() == "original"


def test_parser_not_advanced_beyond_limit(corpus, options, monkeypatch):
    (corpus / "a.txt").write_text("source")
    advanced = []

    def parser(path, opts):
        for index in range(100):
            advanced.append(index)
            yield ParsedText("text")

    monkeypatch.setattr("src.data.ingestion.get_parser", lambda extension: parser)
    result = ingest_directory(corpus, replace(options, limit=5))
    assert advanced == list(range(5))
    assert result.summary["successful"] == 5


def test_bounded_incremental_run(corpus, options):
    with (corpus / "tiny.csv").open("w") as stream:
        stream.write("text\n")
        for _ in range(1000):
            stream.write("Tiny record " * 40 + "\n")
    options = replace(options, limit=600, text_columns=("text",))
    observations = []

    def progress(snapshot):
        if snapshot["records_written"] in (100, 600):
            with options.output_path.open() as stream:
                assert sum(1 for _ in stream) == snapshot["records_written"]
            observations.append(tracemalloc.get_traced_memory()[0])

    tracemalloc.start()
    try:
        result = ingest_directory(corpus, options, progress=progress)
    finally:
        tracemalloc.stop()
    assert result.summary["records_written"] == 600
    assert len(result.preview) == 20
    assert all(len(p["extracted_text"]) <= 2000 for p in result.preview)
    assert observations[-1] - observations[0] < 2 * 1024**2


def test_no_streamlit_or_torch_import():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.data.ingestion; "
            "assert 'streamlit' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_interrupted_manifest(corpus, options):
    (corpus / "a.txt").write_text("Some text")

    def stop(_snapshot):
        raise RuntimeError("simulated disconnect")

    with pytest.raises(RuntimeError):
        ingest_directory(corpus, options, progress=stop)
    manifest = json.loads(next(options.manifest_dir.glob("*.json")).read_text())
    assert manifest["state"] == "interrupted"
    assert manifest["records_written"] == 1


def test_cli_help_and_small_ingestion(corpus, options, monkeypatch):
    import scripts.ingest_dataset as cli

    monkeypatch.setattr(cli, "load_settings", lambda _: options.settings)
    (corpus / "tiny.txt").write_text("Tiny fixture")
    assert cli.main(["--source", str(corpus)]) == 0
    assert cli.main(["--source", str(corpus)]) == 2
    assert cli.main(["--source", str(corpus), "--limit", "100001"]) == 2
    help_result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/ingest_dataset.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "--text-columns" in help_result.stdout
