"""One canonical train/load/encode API for future model phases."""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import tokenizers
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)

from src.config.settings import Settings
from src.tokenizer.corpus import CorpusError, text_batches
from src.tokenizer.statistics import analyze_split

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncodedText:
    ids: list[int]
    tokens: list[str]
    token_count: int
    unknown_count: int


@dataclass(frozen=True)
class TokenizerResult:
    output_dir: Path
    manifest_path: Path
    statistics_path: Path
    manifest: dict


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configuration_fingerprint(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _special_ids(tokenizer: Tokenizer, config: dict) -> dict:
    values = {}
    for name in ("pad_token", "unk_token", "bos_token", "eos_token"):
        token = config[name]
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise RuntimeError(f"Required special token is absent after training: {token}")
        values[name] = token
        values[f"{name}_id"] = token_id
    return values


def _build(config: dict) -> Tokenizer:
    tokenizer = Tokenizer(
        models.WordPiece(
            unk_token=config["unk_token"],
            max_input_chars_per_word=config["max_input_characters_per_word"],
        )
    )
    if config["lowercase"]:
        tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece(prefix=config["continuing_subword_prefix"])
    return tokenizer


def train_wordpiece(
    train_path: str | Path,
    settings: Settings,
    *,
    output_dir: str | Path | None = None,
    validation_path: str | Path | None = None,
    test_path: str | Path | None = None,
    dataset_fingerprint: str | None = None,
    overwrite: bool = False,
) -> TokenizerResult:
    """Train genuine WordPiece only on `train_path`; other splits are analysis-only."""
    config = settings.values["tokenizer"]
    source = Path(train_path).absolute().resolve()
    if source.name != "train.jsonl":
        raise ValueError("Tokenizer training source must be the canonical train.jsonl split")
    target = Path(output_dir or settings.paths["MODEL_DIR"] / "tokenizer").absolute().resolve()
    model_root = settings.paths["MODEL_DIR"].absolute().resolve()
    if target != model_root and model_root not in target.parents:
        raise ValueError("Tokenizer artifacts must be written below the configured MODEL_DIR")
    existing = (
        [path for path in target.iterdir() if path.name != ".gitkeep"] if target.exists() else []
    )
    if existing and not overwrite:
        raise FileExistsError("Tokenizer output exists; use another path or --overwrite")
    target.mkdir(parents=True, exist_ok=True)
    tokenizer = _build(config)
    trainer = trainers.WordPieceTrainer(
        vocab_size=config["vocab_size"],
        min_frequency=config["min_frequency"],
        special_tokens=config["special_tokens"],
        continuing_subword_prefix=config["continuing_subword_prefix"],
    )
    documents = characters = 0

    def corpus():
        nonlocal documents, characters
        for batch in text_batches(source, config["training_batch_documents"]):
            documents += len(batch)
            characters += sum(len(text) for text in batch)
            yield batch

    started = datetime.now(UTC).isoformat()
    timer = perf_counter()
    logger.info(
        "WordPiece training started from %s with requested vocabulary %d",
        source,
        config["vocab_size"],
    )
    try:
        tokenizer.train_from_iterator(corpus(), trainer=trainer)
    except CorpusError:
        raise
    if not documents:
        raise CorpusError("Training corpus contains no valid documents")
    duration = perf_counter() - timer
    special_ids = _special_ids(tokenizer, config)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{config['bos_token']} $A {config['eos_token']}",
        special_tokens=[
            (config["bos_token"], special_ids["bos_token_id"]),
            (config["eos_token"], special_ids["eos_token_id"]),
        ],
    )
    tokenizer_path = target / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    tokenizer.model.save(str(target))
    actual_vocab = tokenizer.get_vocab_size()
    analysis = {
        "train": analyze_split(
            tokenizer, source, config["training_batch_documents"], special_ids["unk_token_id"]
        )
    }
    for name, path in (("validation", validation_path), ("test", test_path)):
        if path is not None:
            analysis[name] = analyze_split(
                tokenizer,
                Path(path),
                config["training_batch_documents"],
                special_ids["unk_token_id"],
            )
    statistics_path = target / "training_statistics.json"
    statistics_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    tokenizer_fingerprint = _sha256(tokenizer_path)
    tokenizer_config = {
        "type": "WordPiece",
        "pre_tokenizer": "BertPreTokenizer",
        **config,
        **special_ids,
    }
    (target / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2), encoding="utf-8"
    )
    manifest = {
        "tokenizer_type": "WordPiece",
        "requested_vocabulary_size": config["vocab_size"],
        "actual_vocabulary_size": actual_vocab,
        "min_frequency": config["min_frequency"],
        "continuing_subword_prefix": config["continuing_subword_prefix"],
        "special_tokens": special_ids,
        "source_train_dataset_path": str(source),
        "dataset_fingerprint": dataset_fingerprint,
        "training_corpus_documents": documents,
        "training_corpus_characters": characters,
        "training_started_at": started,
        "training_finished_at": datetime.now(UTC).isoformat(),
        "training_duration_seconds": duration,
        "tokenizers_version": tokenizers.__version__,
        "project_version": "0.4.0",
        "configuration_fingerprint": _configuration_fingerprint(config),
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "analysis": analysis,
    }
    manifest_path = target / "tokenizer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if actual_vocab < config["vocab_size"]:
        logger.warning(
            "Requested vocabulary %d; actual vocabulary %d", config["vocab_size"], actual_vocab
        )
    logger.info("WordPiece training finished: %d documents in %.3fs", documents, duration)
    return TokenizerResult(target, manifest_path, statistics_path, manifest)


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load the canonical serialized tokenizer artifact."""
    source = Path(path)
    if source.is_dir():
        source /= "tokenizer.json"
    if not source.is_file():
        raise FileNotFoundError(f"Tokenizer artifact not found: {source}")
    return Tokenizer.from_file(str(source))


def encode(
    tokenizer: Tokenizer, text: str, *, add_special_tokens: bool = False, unk_token: str = "[UNK]"
) -> EncodedText:
    """Encode without padding/truncation; BOS/EOS are explicit opt-in."""
    result = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    return EncodedText(result.ids, result.tokens, len(result.ids), result.tokens.count(unk_token))


def inspect_vocabulary(tokenizer: Tokenizer, limit: int = 100) -> list[tuple[str, int]]:
    """Return an ID-ordered bounded vocabulary preview."""
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    return sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])[:limit]
