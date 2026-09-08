"""Build and inspect an untrained Transformer; never trains."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import PROJECT_ROOT, load_settings  # noqa:E402

parser = argparse.ArgumentParser(
    description="Inspect an untrained custom decoder-only Transformer."
)
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--config")
parser.add_argument("--forward-test", action="store_true")
args = parser.parse_args()
from src.transformer.factory import build  # noqa:E402

path = (
    (PROJECT_ROOT / args.tokenizer).resolve()
    if not Path(args.tokenizer).is_absolute()
    else Path(args.tokenizer)
)
model, fingerprint = build(load_settings(args.config), path)
from src.transformer.inspection import inspect  # noqa:E402

result = {
    **inspect(model),
    "vocabulary_size": model.config.vocab_size,
    "tokenizer_fingerprint": fingerprint,
    "head_dimension": model.config.head_dim,
}
if args.forward_test:
    import torch

    result["forward_shape"] = list(
        model(torch.zeros((2, min(8, model.config.context_length)), dtype=torch.long)).shape
    )
print(json.dumps(result, indent=2))
