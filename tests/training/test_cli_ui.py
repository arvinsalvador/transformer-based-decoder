import json
import os
import subprocess
import sys

import yaml
from streamlit.testing.v1 import AppTest

from src.config.settings import PROJECT_ROOT


def configure(case, tmp_path):
    settings = case[0]
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(settings.values))
    return path, {
        **os.environ,
        **{key: str(value) for key, value in settings.paths.items()},
        "CONFIG_PATH": str(path),
        "DEVICE": "cpu",
        "TOKENIZERS_PARALLELISM": "false",
    }


def test_cli_dry_five_steps_and_resume(training_case, tmp_path):
    settings, train, validation, token_dir = training_case
    config, env = configure(training_case, tmp_path)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_transformer.py"),
        "--train",
        str(train),
        "--validation",
        str(validation),
        "--tokenizer",
        str(token_dir),
        "--config",
        str(config),
    ]

    def run(*options):
        result = subprocess.run(
            [*command, *options],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    assert run("--dry-run")["backward"] == "PASS"
    first = run("--max-steps", "5")
    assert first["optimizer_steps"] == 5
    path = settings.paths["CHECKPOINT_DIR"] / first["run_id"] / "latest.pt"
    assert (
        run("--max-steps", "7", "--resume", str(path), "--overwrite-export")["optimizer_steps"] == 7
    )
    assert (train.parent / "test.jsonl").read_text() == "{DO NOT READ TEST}"


def test_streamlit_dry_run_and_small_training(training_case, tmp_path, monkeypatch):
    _, env = configure(training_case, tmp_path)
    for key in (*training_case[0].paths, "CONFIG_PATH", "DEVICE"):
        monkeypatch.setenv(key, env[key])
    app = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py")).run(timeout=30)
    app.sidebar.radio[0].set_value("Training").run(timeout=30)
    assert not app.exception
    dry = next(button for button in app.button if button.label == "Run Dry Run")
    dry.click().run(timeout=30)
    assert not app.exception and not app.error
    small = next(button for button in app.button if button.label == "Start Small Training")
    small.click().run(timeout=30)
    assert not app.exception and not app.error
    assert (training_case[0].paths["MODEL_DIR"] / "transformer/best_model.pt").is_file()
