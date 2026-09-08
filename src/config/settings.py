"""Load validated profiles; resolve paths against the project, never the current directory."""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from src.config.evaluation import FIELDS as EVALUATION_FIELDS
from src.config.evaluation import validate_evaluation
from src.config.training import FIELDS as TRAINING_FIELDS
from src.config.training import validate_training

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
        "dataset": {
            "max_documents",
            "working_document_limit",
            "max_file_size_mb",
            "min_extracted_characters",
            "recursive",
        },
        "ingestion": {
            "output_format",
            "output_path",
            "manifest_dir",
            "preview_characters",
            "preview_documents",
            "max_extracted_characters",
            "max_docx_uncompressed_mb",
            "progress_interval",
            "max_upload_files",
            "max_upload_total_mb",
        },
        "csv": {"mode", "text_columns"},
        "preprocessing": {
            "unicode_normalization",
            "normalize_whitespace",
            "max_consecutive_blank_lines",
            "preserve_case",
            "preserve_punctuation",
            "strip_control_characters",
            "normalize_nonbreaking_spaces",
            "min_characters",
            "max_characters_per_document",
            "min_alpha_ratio",
            "max_control_ratio",
            "max_repeated_character_run",
            "long_document_policy",
            "preview_characters",
            "preview_documents",
            "progress_interval",
        },
        "split": {"train_ratio", "validation_ratio", "test_ratio"},
        "tokenizer": {
            "type",
            "vocab_size",
            "min_frequency",
            "continuing_subword_prefix",
            "special_tokens",
            "unk_token",
            "pad_token",
            "bos_token",
            "eos_token",
            "lowercase",
            "max_input_characters_per_word",
            "training_batch_documents",
        },
        "trigram": {"storage_backend", "smoothing", "add_k", "min_count", "generation"},
        "model": {
            "context_length",
            "embedding_dim",
            "num_layers",
            "num_heads",
            "feedforward_dim",
            "dropout",
            "positional_embedding",
            "layer_norm_eps",
            "bias",
            "tie_embeddings",
            "initialization_std",
        },
        "training": TRAINING_FIELDS,
        "evaluation": EVALUATION_FIELDS,
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
        if section == "evaluation":
            try:
                validate_evaluation(block)
            except ValueError as exc:
                raise ConfigurationError(str(exc)) from exc
            continue
        if section == "training":
            try:
                validate_training(block, values["model"]["context_length"])
            except ValueError as exc:
                raise ConfigurationError(str(exc)) from exc
            continue
        if not isinstance(block, dict) or set(block) != fields:
            raise ConfigurationError(f"{section} must contain exactly {sorted(fields)}")
        for field, value in block.items():
            name = f"{section}.{field}"
            if field in ("output_path", "manifest_dir"):
                if not isinstance(value, str) or not value.strip():
                    raise ConfigurationError(f"{name} must be a nonempty path")
                if Path(value).is_absolute() or ".." in Path(value).parts:
                    raise ConfigurationError(f"{name} must be relative without parent traversal")
            elif field == "output_format":
                if value != "jsonl":
                    raise ConfigurationError(f"{name} must be jsonl")
            elif field == "mode":
                if value not in ("rows", "file"):
                    raise ConfigurationError(f"{name} must be rows or file")
            elif field in ("text_columns", "special_tokens"):
                if (
                    not isinstance(value, list)
                    or any(not isinstance(v, str) or not v.strip() for v in value)
                    or len(value) != len(set(value))
                ):
                    raise ConfigurationError(f"{name} must be a list of unique column names")
            elif field in (
                "continuing_subword_prefix",
                "unk_token",
                "pad_token",
                "bos_token",
                "eos_token",
            ):
                if not isinstance(value, str) or not value:
                    raise ConfigurationError(f"{name} must be a nonempty string")
            elif field == "type":
                if value != "wordpiece":
                    raise ConfigurationError("tokenizer.type must be wordpiece")
            elif field == "positional_embedding":
                if value != "learned":
                    raise ConfigurationError("model.positional_embedding must be learned")
            elif field == "storage_backend":
                if value not in ("memory", "sqlite", "auto"):
                    raise ConfigurationError(
                        "trigram.storage_backend must be memory, sqlite, or auto"
                    )
            elif field == "smoothing":
                if value != "add_k":
                    raise ConfigurationError("trigram.smoothing must be add_k")
            elif field == "generation":
                continue
            elif field in ("add_k", "layer_norm_eps", "initialization_std"):
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ConfigurationError(f"{name} must be a finite positive number")
            elif field in (
                "mixed_precision",
                "recursive",
                "lowercase",
                "bias",
                "tie_embeddings",
                "normalize_whitespace",
                "preserve_case",
                "preserve_punctuation",
                "strip_control_characters",
                "normalize_nonbreaking_spaces",
            ):
                if type(value) is not bool:
                    raise ConfigurationError(f"{name} must be a boolean")
            elif field in (
                "dropout",
                "min_alpha_ratio",
                "max_control_ratio",
                "train_ratio",
                "validation_ratio",
                "test_ratio",
            ):
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not 0 <= value < 1
                ):
                    raise ConfigurationError(f"{name} must be a finite number in [0, 1)")
            elif field == "unicode_normalization":
                if value not in ("NFC", "NFKC", "NFD", "NFKD"):
                    raise ConfigurationError(
                        f"{name} must be a supported Unicode normalization form"
                    )
            elif field == "long_document_policy":
                if value not in ("truncate", "skip"):
                    raise ConfigurationError(f"{name} must be truncate or skip")
            else:
                integer(
                    value,
                    name,
                    0 if field == "num_workers" else 1,
                    100000 if field in ("max_documents", "working_document_limit") else None,
                )
    generation = values["trigram"]["generation"]
    if not isinstance(generation, dict) or set(generation) != {
        "strategy",
        "max_new_tokens",
        "temperature",
        "top_k",
    }:
        raise ConfigurationError(
            "trigram.generation must contain strategy, max_new_tokens, temperature, top_k"
        )
    if generation["strategy"] not in ("greedy", "sample"):
        raise ConfigurationError("trigram.generation.strategy must be greedy or sample")
    integer(generation["max_new_tokens"], "trigram.generation.max_new_tokens")
    integer(generation["top_k"], "trigram.generation.top_k")
    if type(generation["temperature"]) not in (int, float) or generation["temperature"] <= 0:
        raise ConfigurationError("trigram.generation.temperature must be > 0")
    if type(values["trigram"]["add_k"]) not in (int, float) or values["trigram"]["add_k"] <= 0:
        raise ConfigurationError("trigram.add_k must be > 0")
    if values["dataset"]["working_document_limit"] > values["dataset"]["max_documents"]:
        raise ConfigurationError("dataset.working_document_limit must be <= dataset.max_documents")
    if values["model"]["embedding_dim"] % values["model"]["num_heads"]:
        raise ConfigurationError("model.embedding_dim must be divisible by model.num_heads")
    if (
        values["dataset"]["min_extracted_characters"]
        > values["ingestion"]["max_extracted_characters"]
    ):
        raise ConfigurationError(
            "min_extracted_characters must not exceed max_extracted_characters"
        )
    if values["ingestion"]["preview_characters"] > 2000:
        raise ConfigurationError("ingestion.preview_characters must be <= 2000")
    if values["ingestion"]["preview_documents"] > 100:
        raise ConfigurationError("ingestion.preview_documents must be <= 100")
    preprocessing = values["preprocessing"]
    if preprocessing["min_characters"] > preprocessing["max_characters_per_document"]:
        raise ConfigurationError("preprocessing.min_characters must not exceed maximum")
    if preprocessing["preview_characters"] > 2000 or preprocessing["preview_documents"] > 100:
        raise ConfigurationError("preprocessing preview limits exceed the UI safety bounds")
    ratios = values["split"]
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ConfigurationError("split ratios must sum to 1.0")
    if any(value <= 0 for value in ratios.values()):
        raise ConfigurationError("split ratios must all be positive")
    if not preprocessing["preserve_case"] or not preprocessing["preserve_punctuation"]:
        raise ConfigurationError("Phase 3 must preserve case and punctuation")
    tokenizer = values["tokenizer"]
    if not 1000 <= tokenizer["vocab_size"] <= 50000:
        raise ConfigurationError("tokenizer.vocab_size must be between 1000 and 50000")
    required_tokens = {
        tokenizer[key] for key in ("unk_token", "pad_token", "bos_token", "eos_token")
    }
    if len(tokenizer["special_tokens"]) != len(set(tokenizer["special_tokens"])):
        raise ConfigurationError("tokenizer.special_tokens must be unique")
    if len(required_tokens) != 4 or not required_tokens.issubset(tokenizer["special_tokens"]):
        raise ConfigurationError("tokenizer special token fields must be distinct and listed")
    if tokenizer["vocab_size"] <= len(tokenizer["special_tokens"]):
        raise ConfigurationError("tokenizer.vocab_size must exceed special token count")


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
