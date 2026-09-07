"""Validate profile boundaries without accessing datasets."""

from copy import deepcopy

import pytest
import yaml

from src.config.settings import ConfigurationError, load_settings, validate_config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "APP_ENV",
        "CONFIG_PATH",
        "DEVICE",
        "WORKING_DOCUMENT_LIMIT",
        "BATCH_SIZE",
        "DATA_DIR",
        "MODEL_DIR",
        "CHECKPOINT_DIR",
        "EXPERIMENT_DIR",
        "REPORT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def test_profiles():
    local = load_settings("config/local.yaml")
    gpu = load_settings("config/gpu.yaml")
    assert local.values["dataset"]["working_document_limit"] == 1000
    assert gpu.values["device"] == "cuda"
    assert gpu.values["training"]["mixed_precision"] is True
    assert local.paths["DATA_DIR"].is_absolute()


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("dataset", "max_documents", 100001),
        ("dataset", "working_document_limit", 0),
        ("dataset", "working_document_limit", 100001),
        ("training", "batch_size", -1),
        ("training", "epochs", 0),
        ("training", "num_workers", -1),
        ("training", "mixed_precision", "false"),
        ("training", "batch_size", True),
        ("training", "gradient_accumulation_steps", 0),
        ("model", "context_length", 0),
        ("model", "embedding_dim", 193),
        ("model", "num_heads", 0),
        ("model", "num_layers", 0),
        ("model", "feedforward_dim", -1),
        ("model", "dropout", float("nan")),
        ("model", "dropout", 1),
        ("tokenizer", "type", "bpe"),
        ("tokenizer", "vocab_size", 0),
    ],
)
def test_invalid_values(section, field, value):
    values = deepcopy(load_settings("config/local.yaml").values)
    values[section][field] = value
    with pytest.raises(ConfigurationError, match=field):
        validate_config(values)


def test_working_limit_exceeds_maximum():
    values = load_settings("config/local.yaml").values
    values["dataset"]["max_documents"] = 10
    with pytest.raises(ConfigurationError, match="working_document_limit"):
        validate_config(values)


def test_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "gpu")
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("BATCH_SIZE", "2")
    monkeypatch.setenv("WORKING_DOCUMENT_LIMIT", "12")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    settings = load_settings()
    assert settings.values["environment"] == "gpu"
    assert settings.values["device"] == "cpu"
    assert settings.values["training"]["batch_size"] == 2
    assert settings.values["dataset"]["working_document_limit"] == 12
    assert settings.paths["DATA_DIR"] == tmp_path


def test_precedence_and_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APP_ENV", "gpu")
    monkeypatch.setenv("CONFIG_PATH", "config/local.yaml")
    assert load_settings().values["environment"] == "local"
    assert load_settings("config/gpu.yaml").values["environment"] == "gpu"


def test_dotenv_does_not_mutate_environment(tmp_path, monkeypatch):
    values = load_settings("config/local.yaml").values
    (tmp_path / "profile.yaml").write_text(yaml.safe_dump(values))
    (tmp_path / ".env").write_text("CONFIG_PATH=profile.yaml\nBATCH_SIZE=3\nDATA_DIR=corpus\n")
    monkeypatch.setenv("BATCH_SIZE", "2")
    settings = load_settings(root=tmp_path)
    assert settings.values["training"]["batch_size"] == 2
    assert settings.paths["DATA_DIR"] == tmp_path / "corpus"


@pytest.mark.parametrize("contents", ["[]", "model: [", "{}", "environment: local"])
def test_malformed_file(tmp_path, contents):
    path = tmp_path / "bad.yaml"
    path.write_text(contents)
    with pytest.raises(ConfigurationError):
        load_settings(path)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigurationError, match="Cannot load"):
        load_settings(tmp_path / "absent.yaml")


def test_invalid_env(monkeypatch):
    monkeypatch.setenv("BATCH_SIZE", "abc")
    with pytest.raises(ConfigurationError, match="BATCH_SIZE"):
        load_settings("config/local.yaml")
