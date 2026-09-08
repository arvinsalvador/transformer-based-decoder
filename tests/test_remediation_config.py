"""Positive scalars must not inherit ratio bounds."""

from copy import deepcopy

import pytest

from src.config.settings import ConfigurationError, load_settings, validate_config


@pytest.mark.parametrize(
    ("section", "key"),
    [("trigram", "add_k"), ("model", "layer_norm_eps"), ("model", "initialization_std")],
)
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1", 1.0, 2.0, 1.25])
def test_positive_scalars(section, key, value):
    values = deepcopy(load_settings("config/local.yaml").values)
    values[section][key] = value
    if type(value) is float and value in (1.0, 2.0, 1.25):
        validate_config(values)
    else:
        with pytest.raises(ConfigurationError):
            validate_config(values)


@pytest.mark.parametrize("value", [-0.1, 1.0, 2.0, float("nan")])
def test_dropout_retains_ratio_bounds(value):
    values = deepcopy(load_settings("config/local.yaml").values)
    values["model"]["dropout"] = value
    with pytest.raises(ConfigurationError):
        validate_config(values)
