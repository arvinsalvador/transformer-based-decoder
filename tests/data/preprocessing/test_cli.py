"""CLI help and a small preparation invocation."""

import json
import subprocess
import sys

from scripts import prepare_dataset as cli
from src.config.settings import PROJECT_ROOT
from tests.data.preprocessing.conftest import raw_record


def test_cli_help_and_run(tmp_path, prep_settings, monkeypatch):
    source = tmp_path / "documents.jsonl"
    source.write_text(json.dumps(raw_record("one", "Useful preparation document.")) + "\n")
    monkeypatch.setattr(cli, "load_settings", lambda _: prep_settings)
    assert (
        cli.main(
            [
                "--input",
                str(source),
                "--clean-output",
                str(tmp_path / "clean.jsonl"),
                "--split-dir",
                str(tmp_path / "splits"),
            ]
        )
        == 0
    )
    help_result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/prepare_dataset.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "--no-split" in help_result.stdout
