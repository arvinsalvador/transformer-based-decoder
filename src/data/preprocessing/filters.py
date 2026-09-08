"""Permissive, explainable quality checks without model-based scoring."""

import unicodedata

from src.data.preprocessing.models import PreprocessingStatus, ProcessingOutcome


def control_ratio(text: str) -> float:
    """Count invalid C0/C1 controls but allow tab/newline as meaningful structure."""
    if not text:
        return 0.0
    controls = sum(unicodedata.category(char) == "Cc" and char not in "\n\t\r" for char in text)
    return controls / len(text)


def alpha_ratio(text: str) -> float:
    """Unicode-aware alphabetic ratio; punctuation and technical symbols are allowed."""
    visible = [char for char in text if not char.isspace()]
    return sum(char.isalpha() for char in visible) / len(visible) if visible else 0.0


def longest_repeated_character_run(text: str) -> int:
    """Detect pathological single-character runs without penalizing normal technical repetition."""
    longest = current = 0
    previous = None
    for char in text:
        current = current + 1 if char == previous else 1
        longest = max(longest, current)
        previous = char
    return longest


def quality_check(original: str, normalized: str, config: dict) -> ProcessingOutcome | None:
    """Return the first deterministic rejection; callers apply long-document policy first."""
    if not normalized:
        return ProcessingOutcome(
            PreprocessingStatus.EMPTY_TEXT, "Text is empty after normalization"
        )
    if control_ratio(original) > config["max_control_ratio"]:
        return ProcessingOutcome(
            PreprocessingStatus.EXCESSIVE_CONTROL_CHARACTERS, "Control ratio exceeds limit"
        )
    if len(normalized) < config["min_characters"]:
        return ProcessingOutcome(
            PreprocessingStatus.TOO_SHORT, "Text is below minimum character count"
        )
    if longest_repeated_character_run(normalized) > config["max_repeated_character_run"]:
        return ProcessingOutcome(
            PreprocessingStatus.EXCESSIVE_REPETITION, "Repeated-character run exceeds limit"
        )
    if alpha_ratio(normalized) < config["min_alpha_ratio"]:
        return ProcessingOutcome(
            PreprocessingStatus.LOW_ALPHA_RATIO, "Alphabetic ratio is below limit"
        )
    return None
