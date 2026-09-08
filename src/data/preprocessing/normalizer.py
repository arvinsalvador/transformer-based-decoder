"""Conservative, deterministic text normalization for language-model corpora."""

import re
import unicodedata


def normalize(text: str, config: dict) -> str:
    """Preserve case, punctuation, numbers, URLs and paragraph boundaries by default."""
    value = unicodedata.normalize(config["unicode_normalization"], text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    if config["normalize_nonbreaking_spaces"]:
        value = value.replace("\u00a0", " ")
    if config["strip_control_characters"]:
        value = "".join(
            char for char in value if char in "\n\t" or unicodedata.category(char) != "Cc"
        )
    if config["normalize_whitespace"]:
        value = re.sub(r"[\t \f\v]+", " ", value)
        maximum = config["max_consecutive_blank_lines"]
        value = re.sub(
            r"\n[ \t]*\n(?:[ \t]*\n){" + str(maximum) + r",}", "\n" * (maximum + 1), value
        )
    return value.strip()
