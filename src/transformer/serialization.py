"""Safe JSON architecture-configuration serialization; no weights are persisted here."""

import json
from pathlib import Path

from src.transformer.config import TransformerConfig


def save_config(config: TransformerConfig, path: str | Path) -> Path:
    """Write only validated, non-executable architecture metadata."""
    target = Path(path)
    target.write_text(json.dumps(config.to_dict(), sort_keys=True, indent=2), encoding="utf-8")
    return target


def load_config(path: str | Path) -> TransformerConfig:
    """Load a JSON configuration without arbitrary-code deserialization."""
    return TransformerConfig(**json.loads(Path(path).read_text(encoding="utf-8")))
