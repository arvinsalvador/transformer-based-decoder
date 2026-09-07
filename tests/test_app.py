"""Exercise Streamlit script execution, including unavailable CUDA and navigation."""

from streamlit.testing.v1 import AppTest

from src.config.settings import PROJECT_ROOT
from src.utils import device


def test_dashboard_and_placeholders(monkeypatch):
    monkeypatch.setenv("CONFIG_PATH", "config/local.yaml")
    monkeypatch.setenv("DEVICE", "cpu")
    app = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py")).run(timeout=30)
    assert not app.exception
    assert "Transformer-Based" in app.title[0].value
    for section in app.sidebar.radio[0].options[1:]:
        app.sidebar.radio[0].set_value(section).run()
        assert not app.exception
        assert app.info[0].value == "Available in a later phase"


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
