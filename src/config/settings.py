"""Load validated profiles; resolve paths against the project, never the current directory."""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigurationError(ValueError):
    """Configuration is missing, malformed, or outside the supported bounds."""


@dataclass(frozen=True)
class Settings:
    """Validated configuration and resolved runtime paths (no directories are created)."""

    config_path: Path
    values: dict[str, Any]
    paths: dict[str, Path]


def validate_config(values: dict[str, Any]) -> None:
    """Reject invalid fields before future phases allocate resources."""
    schema = {
        "dataset": {"max_documents", "working_document_limit"},
        "tokenizer": {"type", "vocab_size"},
        "model": {
            "context_length",
            "embedding_dim",
            "num_layers",
            "num_heads",
            "feedforward_dim",
            "dropout",
        },
        "training": {
            "batch_size",
            "epochs",
            "gradient_accumulation_steps",
            "mixed_precision",
            "num_workers",
        },
    }
    expected = {"environment", "device", "random_seed", *schema}
    if set(values) != expected:
        raise ConfigurationError(
            f"Expected top-level fields {sorted(expected)}; got {sorted(values, key=str)}"
        )
    for key, choices in (("environment", ("local", "gpu")), ("device", ("auto", "cpu", "cuda"))):
        if values[key] not in choices:
            raise ConfigurationError(f"{key} must be one of {choices}")

    def integer(value: Any, name: str, minimum: int = 1, maximum: int | None = None) -> None:
        if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
            raise ConfigurationError(
                f"{name} must be an integer >= {minimum}"
                + (f" and <= {maximum}" if maximum is not None else "")
            )

    integer(values["random_seed"], "random_seed", 0, 2**32 - 1)
    for section, fields in schema.items():
        block = values[section]
        if not isinstance(block, dict) or set(block) != fields:
            raise ConfigurationError(f"{section} must contain exactly {sorted(fields)}")
        for field, value in block.items():
            name = f"{section}.{field}"
            if field == "type":
                if value != "wordpiece":
                    raise ConfigurationError("tokenizer.type must be wordpiece")
            elif field == "mixed_precision":
                if type(value) is not bool:
                    raise ConfigurationError(f"{name} must be a boolean")
            elif field == "dropout":
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not 0 <= value < 1
                ):
                    raise ConfigurationError(f"{name} must be a finite number in [0, 1)")
            else:
                integer(
                    value,
                    name,
                    0 if field == "num_workers" else 1,
                    100000 if section == "dataset" else None,
                )
    if values["dataset"]["working_document_limit"] > values["dataset"]["max_documents"]:
        raise ConfigurationError("dataset.working_document_limit must be <= dataset.max_documents")
    if values["model"]["embedding_dim"] % values["model"]["num_heads"]:
        raise ConfigurationError("model.embedding_dim must be divisible by model.num_heads")


def load_settings(config_path: str | Path | None = None, *, root: Path = PROJECT_ROOT) -> Settings:
    """Load YAML; explicit path > CONFIG_PATH > APP_ENV (default local).

    Process environment overrides optional project .env values. Relative configuration
    and runtime paths are rooted at the project. No global environment is mutated.
    """
    env = {**dotenv_values(root / ".env"), **os.environ}
    profile = env.get("APP_ENV", "local")
    if config_path is None and not env.get("CONFIG_PATH") and profile not in ("local", "gpu"):
        raise ConfigurationError("APP_ENV must be local or gpu")

    def resolve(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()

    path = resolve(config_path or env.get("CONFIG_PATH") or f"config/{profile}.yaml")
    try:
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load configuration {path}: {exc}") from exc
    if not isinstance(values, dict):
        raise ConfigurationError(f"Configuration {path} must be a YAML mapping")
    if env.get("DEVICE"):
        values["device"] = env["DEVICE"]
    overrides = {
        "WORKING_DOCUMENT_LIMIT": ("dataset", "working_document_limit"),
        "BATCH_SIZE": ("training", "batch_size"),
    }
    for variable, (section, field) in overrides.items():
        if env.get(variable) is not None:
            try:
                value = int(env[variable])
            except (ValueError, TypeError) as exc:
                raise ConfigurationError(f"{variable} must be an integer") from exc
            if not isinstance(values.get(section), dict):
                raise ConfigurationError(f"{section} must be a mapping")
            values[section][field] = value
    validate_config(values)
    defaults = {
        "DATA_DIR": "data",
        "MODEL_DIR": "models",
        "CHECKPOINT_DIR": "checkpoints",
        "EXPERIMENT_DIR": "experiments",
        "REPORT_DIR": "reports",
    }
    paths = {key: resolve(env.get(key) or default) for key, default in defaults.items()}
    return Settings(path, values, paths)
