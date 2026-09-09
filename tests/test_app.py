"""Exercise Streamlit script execution, including unavailable CUDA and navigation."""

from streamlit.testing.v1 import AppTest

from src.config.settings import PROJECT_ROOT
from src.utils import device


def test_dashboard_and_all_implemented_pages(monkeypatch):
    monkeypatch.setenv("CONFIG_PATH", "config/local.yaml")
    monkeypatch.setenv("DEVICE", "cpu")
    app = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py")).run(timeout=30)
    assert not app.exception
    assert "Transformer-Based" in app.title[0].value
    for section in app.sidebar.radio[0].options[1:]:
        app.sidebar.radio[0].set_value(section).run()
        assert not app.exception
        if section == "Documents":
            assert "Large datasets" in app.info[0].value
        elif section == "Final Results":
            assert "FULL is the primary" in app.info[0].value
        elif section == "Preprocessing":
            assert "large corpora" in app.info[0].value
        elif section == "Tokenizer":
            assert "Vocabulary training" in app.info[0].value
        elif section == "Trigram Model":
            assert "never updates counts" in app.info[0].value
        elif section == "Transformer":
            assert "Architecture ready" in app.info[0].value
        elif section == "Training":
            assert "Training engine implemented" in app.info[0].value
        elif section == "Evaluation":
            assert "Read-only evaluation" in app.info[0].value
        elif section in ("Comparison", "Generate Text"):
            assert "Shared comparison framework" in app.info[0].value
        elif section == "Experiments":
            assert "Inspection only" in app.info[0].value
        else:
            raise AssertionError(f"Unvalidated navigation section: {section}")


def test_gpu_profile_warning(monkeypatch):
    monkeypatch.setenv("CONFIG_PATH", "config/gpu.yaml")
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: False)
    app = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py")).run(timeout=30)
    assert not app.exception
    assert any("CUDA was requested" in warning.value for warning in app.warning)


def test_invalid_config_is_controlled(monkeypatch):
    monkeypatch.setenv("CONFIG_PATH", "config/missing.yaml")
    app = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py")).run(timeout=30)
    assert not app.exception
    assert "Configuration error" in app.error[0].value
