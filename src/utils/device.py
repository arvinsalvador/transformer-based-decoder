"""Hardware diagnostics only: never allocate tensors or run benchmarks."""

from dataclasses import dataclass

import torch


class DeviceUnavailableError(RuntimeError):
    """A requested compute device is unavailable."""


@dataclass(frozen=True)
class DeviceInfo:
    """Detected hardware; mixed precision indicates potential CUDA FP16 use only."""

    selected_device: str
    cuda_available: bool
    cuda_device_count: int
    gpu_name: str | None
    gpu_vram_bytes: int | None
    pytorch_version: str
    cuda_version: str | None
    mixed_precision_potential: bool


def detect_device(requested: str = "auto") -> DeviceInfo:
    """Select CPU/CUDA explicitly and report the current CUDA device if present."""
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise DeviceUnavailableError(
            "CUDA was requested but is unavailable. Use the local profile or DEVICE=cpu; "
            "On the server verify the NVIDIA driver, CUDA PyTorch wheel, and container GPU access."
        )
    selected = "cuda" if available and requested != "cpu" else "cpu"
    properties = (
        torch.cuda.get_device_properties(torch.cuda.current_device()) if available else None
    )
    return DeviceInfo(
        selected_device=selected,
        cuda_available=available,
        cuda_device_count=torch.cuda.device_count() if available else 0,
        gpu_name=properties.name if properties else None,
        gpu_vram_bytes=properties.total_memory if properties else None,
        pytorch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        mixed_precision_potential=bool(selected == "cuda" and properties and properties.major >= 7),
    )
