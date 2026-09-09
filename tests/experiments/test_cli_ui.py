import json
import subprocess
import sys

import yaml
from streamlit.testing.v1 import AppTest

from src.config.settings import PROJECT_ROOT
from tests.training.test_cli_ui import configure


def test_safe_cli_plan_and_preflight(experiment_case, tmp_path):
    settings, plan, manifest = experiment_case
    config, env = configure(experiment_case, tmp_path)
    plan_path = tmp_path / "experiments.yaml"
    plan_path.write_text(yaml.safe_dump({"experiment": plan}))
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_experiments.py"),
        "--config",
        str(config),
        "--experiment-config",
        str(plan_path),
        "--dataset-manifest",
        str(manifest),
    ]
    for flags, status in (
        ([], "PLANNED"),
        (["--plan"], "PLANNED"),
        (["--preflight"], "READY"),
        (["--dry-run-only", "--scale", "3"], "READY"),
    ):
        result = subprocess.run(
            [*command, *flags], env=env, capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["status"] == status
    assert not list(settings.paths["EXPERIMENT_DIR"].rglob("best_model.pt"))


def test_experiments_ui_inspection_only(experiment_case, tmp_path, monkeypatch):
    _, env = configure(experiment_case, tmp_path)
    for key in (*experiment_case[0].paths, "CONFIG_PATH", "DEVICE"):
        monkeypatch.setenv(key, env[key])
    app = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py")).run(timeout=30)
    app.sidebar.radio[0].set_value("Experiments").run(timeout=30)
    assert not app.exception and not app.error
    assert not app.button
