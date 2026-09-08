"""Conservative normalization preserves language-model structure."""

from src.data.preprocessing.normalizer import normalize


def test_normalization_preserves_case_punctuation_and_numbers(prep_settings):
    config = prep_settings.values["preprocessing"]
    text = "AI\u00a0  Works!  CVE-2026-42\r\n\r\n\r\n\x00Keep URLs: https://example.test"
    assert normalize(text, config) == "AI Works! CVE-2026-42\n\n\nKeep URLs: https://example.test"


def test_unicode_nfc(prep_settings):
    assert normalize("Cafe\u0301", prep_settings.values["preprocessing"]) == "Café"
