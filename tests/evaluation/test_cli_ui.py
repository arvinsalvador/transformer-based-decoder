import json
import subprocess
import sys

from streamlit.testing.v1 import AppTest

from src.config.settings import PROJECT_ROOT
from tests.training.test_cli_ui import configure


def test_evaluation_cli_and_single_model(evaluation_case, tmp_path):
    settings, test, tokenizer, manifest = evaluation_case
    config, env = configure(evaluation_case, tmp_path)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/evaluate_models.py"),
        "--test",
        str(test),
        "--tokenizer",
        str(tokenizer),
        "--config",
        str(config),
        "--dataset-manifest",
        str(manifest),
        "--limit-documents",
        "1",
    ]
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "COMPLETED"
    # An intentionally absent Transformer is not required for trigram-only scoring.
    result = subprocess.run(
        [*command, "--model", "trigram", "--transformer-model", "missing"],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["models_evaluated"] == ["trigram"]


def test_evaluation_and_comparison_ui(evaluation_case, tmp_path, monkeypatch):
    _, env = configure(evaluation_case, tmp_path)
    for key in (*evaluation_case[0].paths, "CONFIG_PATH", "DEVICE"):
        monkeypatch.setenv(key, env[key])
    app = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py")).run(timeout=30)
    app.sidebar.radio[0].set_value("Evaluation").run(timeout=30)
    assert not app.exception
    next(button for button in app.button if button.label == "Run Evaluation").click().run(
        timeout=30
    )
    assert not app.exception and not app.error
    assert app.success
    for section in ("Comparison", "Generate Text"):
        app.sidebar.radio[0].set_value(section).run(timeout=30)
        assert not app.exception and not app.error
        assert app.dataframe
