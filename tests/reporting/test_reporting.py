import csv
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.config.settings import PROJECT_ROOT, load_settings
from src.evaluation.identity import read_json
from src.reporting.audit import runtime_audit, source_audit
from src.reporting.report import export_report, table
from src.reporting.results import load_results
from tests.training.test_cli_ui import configure


def empty_settings(tmp_path):
    settings = load_settings("config/local.yaml")
    return replace(settings, paths={key: tmp_path / key.lower() for key in settings.paths})


def test_no_results_readiness(tmp_path):
    settings = empty_settings(tmp_path)
    result = export_report(settings, settings.paths["REPORT_DIR"])
    assert result["project_status"] == "IMPLEMENTATION_COMPLETE_EXPERIMENT_PENDING"
    assert result["comparison_table"] == []
    assert result["final"]["status"] == "NOT_RUN"
    assert all(row["execution"] == "NOT_RUN" for row in result["compliance"])
    text = (settings.paths["REPORT_DIR"] / "final_report.md").read_text()
    assert "FINAL EXPERIMENT NOT YET EXECUTED" in text
    assert "| Metric" not in text
    assert (settings.paths["REPORT_DIR"] / "final_report.html").is_file()


def test_valid_full_report(completed_case):
    settings, experiment = completed_case
    result = export_report(
        settings, settings.paths["REPORT_DIR"], experiment=experiment["experiment_id"]
    )
    assert result["final"]["status"] == "PASS", result["final"]
    assert result["project_status"] == "FULLY_COMPLETE"
    assert all(row["execution"] == "PASS" for row in result["compliance"])
    assert result["final"]["warnings"]  # CPU and step-limited synthetic run disclosed.
    with (settings.paths["REPORT_DIR"] / "final_comparison.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(result["comparison_table"])
    assert rows[0]["trigram"] == "12"
    assert result["final"]["generation"]
    summary = read_json(settings.paths["REPORT_DIR"] / "final_summary.json")
    assert summary["final"]["experiment_id"] == experiment["experiment_id"]


def test_corrupt_fingerprint_fails(completed_case):
    settings, result = completed_case
    path = Path(result["run_directory"]) / "scales/full/models/transformer/model_manifest.json"
    data = read_json(path)
    data["tokenizer_fingerprint"] = "wrong"
    path.write_text(json.dumps(data))
    assert load_results(settings, result["experiment_id"])["status"] == "FAIL"


def test_full_limited_cannot_masquerade(completed_case):
    settings, result = completed_case
    full = result["scales"][-1]
    path = Path(full["comparison_path"])
    data = read_json(path)
    data["scope"] = "DEVELOPMENT SUBSET"
    path.write_text(json.dumps(data))
    # Even if a stale status is re-sealed, semantic FULL validation must reject it.
    from src.training.reproducibility import file_hash

    status_path = Path(result["run_directory"]) / "status.json"
    status = read_json(status_path)
    status["scales"][-1]["stages"]["evaluation"]["artifacts"][str(path)] = file_hash(path)
    status_path.write_text(json.dumps(status))
    audited = load_results(settings, result["experiment_id"])
    assert audited["status"] == "FAIL" and "limited" in audited["message"]


def test_wordpiece_type_fails(experiment_case):
    settings = experiment_case[0]
    path = settings.paths["MODEL_DIR"] / "tokenizer/tokenizer_manifest.json"
    data = read_json(path)
    data["tokenizer_type"] = "BPE"
    path.write_text(json.dumps(data))
    assert runtime_audit(settings)["tokenizer"]["status"] == "FAIL"


def test_cap_fails(tmp_path):
    settings = empty_settings(tmp_path)
    root = settings.paths["DATA_DIR"] / "splits"
    root.mkdir(parents=True)
    (root / "train.jsonl").write_text('{"text":"x"}\n' * 99999)
    for name in ("validation", "test"):
        (root / f"{name}.jsonl").write_text('{"text":"x"}\n')
    assert runtime_audit(settings)["data"]["status"] == "FAIL"


def test_static_inventory_and_ignore_policy():
    audit = source_audit()
    assert audit["status"] == "PASS", audit["findings"]
    assert audit["git_operations"] == "NONE"
    assert table({"status": "NOT_RUN"}) == []
    assert all(
        check["assurance"] == "CODE_POLICY_ONLY" for check in audit["leakage_policy"].values()
    )


def test_missing_source_is_not_pass(tmp_path):
    (tmp_path / ".gitignore").write_text("")
    audit = source_audit(tmp_path)
    assert audit["implementation"]["decoder_only_transformer"]["status"] == "FAIL"


def test_readiness_cli_and_ui(tmp_path, monkeypatch):
    settings = empty_settings(tmp_path)
    config, env = configure((settings,), tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/generate_final_report.py"),
            "--config",
            str(config),
            "--readiness-only",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (
        json.loads(result.stdout)["project_status"] == "IMPLEMENTATION_COMPLETE_EXPERIMENT_PENDING"
    )
    for key in (*settings.paths, "CONFIG_PATH", "DEVICE"):
        monkeypatch.setenv(key, env[key])
    app = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py")).run(timeout=30)
    assert not app.exception
    app.sidebar.radio[0].set_value("Final Results").run(timeout=30)
    assert not app.exception and not app.error


def test_full_ui(completed_case, tmp_path, monkeypatch):
    settings, result = completed_case
    export_report(settings, settings.paths["REPORT_DIR"], experiment=result["experiment_id"])
    _, env = configure((settings,), tmp_path)
    for key in (*settings.paths, "CONFIG_PATH", "DEVICE"):
        monkeypatch.setenv(key, env[key])
    app = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py")).run(timeout=30)
    app.sidebar.radio[0].set_value("Final Results").run(timeout=30)
    assert not app.exception and not app.error
    app.checkbox[0].check().run(timeout=30)
    assert not app.exception


def test_stale_primary_summary_fails(completed_case):
    settings, result = completed_case
    path = Path(result["run_directory"]) / "summary.json"
    summary = read_json(path)
    summary["primary_final_result"] = False
    path.write_text(json.dumps(summary))
    audited = load_results(settings, result["experiment_id"])
    assert audited["status"] == "FAIL" and "summary/matrix" in audited["message"]


def test_matrix_csv_mismatch_fails(completed_case):
    settings, result = completed_case
    path = Path(result["run_directory"]) / "experiment_matrix.csv"
    path.write_text("scale,status\nfull,COMPLETED\n")
    audited = load_results(settings, result["experiment_id"])
    assert audited["status"] == "FAIL" and "CSV" in audited["message"]


def test_primary_full_not_best_supporting_scale(completed_case):
    from src.experiments.runner import run_experiment

    settings, result = completed_case
    directory = Path(result["run_directory"])
    resolved = read_json(directory / "experiment_plan.json")
    resumed = run_experiment(
        settings,
        resolved["experiment"],
        mode="execute",
        selected=["3", "5"],
        resume=result["experiment_id"],
        dataset_manifest=result["audit"]["dataset_manifest"],
    )
    assert all(row["status"] == "COMPLETED" for row in resumed["scales"][:2]), resumed
    loaded = load_results(settings, result["experiment_id"])
    assert loaded["status"] == "PASS", loaded
    assert loaded["counts"]["train"] == 12
    assert [row["scale"] for row in loaded["learning_curve"]] == ["3", "5", "20", "full"]
    assert loaded["learning_curve"][2]["status"] != "COMPLETED"


def test_relative_metrics_and_neutral_conclusion():
    from src.evaluation.metrics import compare

    base = dict(
        documents=2,
        prediction_events=6,
        test_fingerprint="test",
        tokenizer_fingerprint="wp",
        perplexity=4.0,
        training_duration_seconds=2.0,
        model_size_bytes=100,
        tokens_per_second=20.0,
    )
    for perplexity, expected in ((2.0, "lower"), (8.0, "higher")):
        relative, statements = compare(base, {**base, "perplexity": perplexity})
        assert relative["perplexity_reduction_percent"] == (1 - perplexity / 4) * 100
        assert expected in statements[0]
    relative, statements = compare(
        {**base, "training_duration_seconds": 0, "perplexity": None}, base
    )
    assert relative["training_time_multiplier"] is None
    assert relative["perplexity_reduction_percent"] is None
    assert not any("perplexity is" in statement for statement in statements)


def test_source_path_warning_suppresses_secret_value(tmp_path):
    directory = tmp_path / "src"
    directory.mkdir()
    (directory / "example.py").write_text(
        'location = "/home/sampleuser/private"\napi_key = "fake_fixture_only_123"\n'
    )
    audit = source_audit(tmp_path)
    findings = json.dumps(audit["findings"])
    assert "workstation" in findings and "credential" in findings
    assert "fake_fixture_only_123" not in findings
