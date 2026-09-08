"""Production tokenizer configuration maintains safety constraints."""

from copy import deepcopy

import pytest

from src.config.settings import ConfigurationError, load_settings, validate_config


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("vocab_size", 999),
        ("min_frequency", 0),
        ("continuing_subword_prefix", ""),
        ("special_tokens", ["[UNK]", "[UNK]"]),
    ],
)
def test_invalid_tokenizer_values(key, value):
    values = deepcopy(load_settings("config/local.yaml").values)
    values["tokenizer"][key] = value
    with pytest.raises(ConfigurationError):
        validate_config(values)
