"""Technical text remains permissive while invalid text is rejected predictably."""

from src.data.preprocessing.filters import quality_check
from src.data.preprocessing.models import PreprocessingStatus


def test_quality_reasons(prep_settings):
    cfg = prep_settings.values["preprocessing"]
    cfg["min_characters"] = 5
    assert quality_check("   ", "", cfg).status == PreprocessingStatus.EMPTY_TEXT
    assert quality_check("abc", "abc", cfg).status == PreprocessingStatus.TOO_SHORT
    assert quality_check("123456", "123456", cfg).status == PreprocessingStatus.LOW_ALPHA_RATIO
    assert (
        quality_check("a\x00\x00\x00text", "atext", cfg).status
        == PreprocessingStatus.EXCESSIVE_CONTROL_CHARACTERS
    )
    cfg["min_characters"] = 1
    cfg["max_repeated_character_run"] = 3
    assert quality_check("aaaa", "aaaa", cfg).status == PreprocessingStatus.EXCESSIVE_REPETITION


def test_technical_text_is_accepted(prep_settings):
    cfg = prep_settings.values["preprocessing"]
    text = "Run curl https://host/x --flag for CVE-2026-1234; path /etc/example.conf."
    assert quality_check(text, text, cfg) is None
