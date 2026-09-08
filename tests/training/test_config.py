from copy import deepcopy

import pytest

from src.config.settings import ConfigurationError, load_settings, validate_config


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", 0),
        ("weight_decay", -1),
        ("beta1", 1),
        ("beta2", 0),
        ("epsilon", float("inf")),
        ("warmup_steps", -1),
        ("min_learning_rate_ratio", 1.01),
        ("max_grad_norm", float("nan")),
        ("sequence_stride", 129),
        ("sequence_stride", 0),
        ("shuffle_buffer_size", 0),
        ("logging_steps", 0),
        ("max_steps", 0),
        ("optimizer", "sgd"),
        ("scheduler", "linear"),
        ("precision", "half"),
        ("deterministic", 1),
        ("pin_memory", "yes"),
        ("persistent_workers", True),
    ],
)
def test_training_config_rejections(field, value):
    values = deepcopy(load_settings("config/local.yaml").values)
    values["training"]["num_workers"] = 0
    values["training"][field] = value
    with pytest.raises(ConfigurationError):
        validate_config(values)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("checkpoint", "keep_last_n", 0),
        ("checkpoint", "save_every_steps", 0),
        ("checkpoint", "save_every_epoch", 1),
        ("early_stopping", "patience", 0),
        ("early_stopping", "min_delta", -1),
        ("early_stopping", "enabled", "false"),
    ],
)
def test_nested_config_rejections(section, field, value):
    values = deepcopy(load_settings("config/local.yaml").values)
    values["training"][section][field] = value
    with pytest.raises(ConfigurationError):
        validate_config(values)


def test_positive_scalars_and_inclusive_endpoints():
    values = deepcopy(load_settings("config/local.yaml").values)
    values["training"].update(
        learning_rate=2,
        epsilon=2,
        max_grad_norm=2,
        weight_decay=0,
        warmup_steps=0,
        min_learning_rate_ratio=1,
    )
    validate_config(values)
