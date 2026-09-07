"""CPU and CUDA behavior is tested with mocked CUDA metadata, never GPU allocations."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.utils import device


@pytest.fixture
def cuda(monkeypatch):
    mock = Mock()
    mock.is_available.return_value = False
    mock.device_count.return_value = 2
    mock.current_device.return_value = 0
    mock.get_device_properties.return_value = SimpleNamespace(
        name="Test GPU",
        total_memory=8 * 2**30,
        major=8,
    )
    monkeypatch.setattr(device.torch, "cuda", mock)
    return mock


def test_auto_cpu(cuda):
    info = device.detect_device()
    assert info.selected_device == "cpu"
    assert info.cuda_device_count == 0
    assert info.gpu_name is None
    assert not info.mixed_precision_potential
    cuda.get_device_properties.assert_not_called()


def test_auto_cuda(cuda):
    cuda.is_available.return_value = True
    info = device.detect_device()
    assert info.selected_device == "cuda"
    assert info.cuda_device_count == 2
    assert info.gpu_vram_bytes == 8 * 2**30
    assert info.gpu_name == "Test GPU"
    assert info.mixed_precision_potential


def test_force_cpu(cuda):
    cuda.is_available.return_value = True
    info = device.detect_device("cpu")
    assert info.selected_device == "cpu"
    assert not info.mixed_precision_potential


def test_cuda_unavailable(cuda):
    with pytest.raises(
        device.DeviceUnavailableError, match="CUDA was requested but is unavailable"
    ):
        device.detect_device("cuda")


def test_force_cuda(cuda):
    cuda.is_available.return_value = True
    assert device.detect_device("cuda").selected_device == "cuda"


def test_invalid_device(cuda):
    with pytest.raises(ValueError, match="device must be"):
        device.detect_device("invalid")
