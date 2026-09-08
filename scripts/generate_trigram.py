"""Generate text with a saved trigram artifact."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import PROJECT_ROOT, load_settings  # noqa: E402


def path(value):
    return (
        (PROJECT_ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    )


parser = argparse.ArgumentParser(description="Generate using a saved trigram.")
parser.add_argument("--model", required=True)
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--prompt", default="")
parser.add_argument("--max-new-tokens", type=int)
parser.add_argument("--strategy", choices=("greedy", "sample"))
parser.add_argument("--seed", type=int)
parser.add_argument("--config")
args = parser.parse_args()
settings = load_settings(args.config)
from src.tokenizer.service import load_tokenizer  # noqa: E402
from src.trigram.serialization import load_model  # noqa: E402

cfg = settings.values["trigram"]["generation"]
tokenizer = load_tokenizer(path(args.tokenizer))
model = load_model(path(args.model) / "trigram_counts.sqlite")
ids, metrics = model.generate(
    tokenizer.encode(args.prompt, add_special_tokens=False).ids,
    max_new_tokens=args.max_new_tokens or cfg["max_new_tokens"],
    strategy=args.strategy or cfg["strategy"],
    temperature=cfg["temperature"],
    top_k=cfg["top_k"],
    seed=args.seed if args.seed is not None else settings.values["random_seed"],
)
print(
    json.dumps(
        {"text": tokenizer.decode(ids, skip_special_tokens=True), "token_ids": ids, **metrics},
        indent=2,
    )
)
