"""Phase 7 validation for the central YAML training section."""

import math

FIELDS = {
    "batch_size",
    "epochs",
    "gradient_accumulation_steps",
    "mixed_precision",
    "num_workers",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "beta1",
    "beta2",
    "epsilon",
    "scheduler",
    "warmup_steps",
    "min_learning_rate_ratio",
    "max_grad_norm",
    "precision",
    "pin_memory",
    "persistent_workers",
    "shuffle_buffer_size",
    "sequence_stride",
    "logging_steps",
    "deterministic",
    "max_steps",
    "checkpoint",
    "early_stopping",
}


def validate_training(block, context_length):
    if not isinstance(block, dict) or set(block) != FIELDS:
        raise ValueError(f"training must contain exactly {sorted(FIELDS)}")

    def integer(name, value, minimum=1):
        if type(value) is not int or value < minimum:
            raise ValueError(f"training.{name} must be an integer >= {minimum}")

    def number(name, value, minimum=0, inclusive=False, maximum=None):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (value < minimum if inclusive else value <= minimum)
            or (maximum is not None and value > maximum)
        ):
            raise ValueError(f"training.{name} has invalid numeric bounds")

    def boolean(name, value):
        if type(value) is not bool:
            raise ValueError(f"training.{name} must be boolean")

    for name in (
        "batch_size",
        "epochs",
        "gradient_accumulation_steps",
        "shuffle_buffer_size",
        "sequence_stride",
        "logging_steps",
    ):
        integer(name, block[name])
    for name in ("num_workers", "warmup_steps"):
        integer(name, block[name], 0)
    if block["max_steps"] is not None:
        integer("max_steps", block["max_steps"])
    for name in ("mixed_precision", "pin_memory", "persistent_workers", "deterministic"):
        boolean(name, block[name])
    for name in ("learning_rate", "epsilon", "max_grad_norm"):
        number(name, block[name])
    number("weight_decay", block["weight_decay"], inclusive=True)
    number("min_learning_rate_ratio", block["min_learning_rate_ratio"], inclusive=True, maximum=1)
    for name in ("beta1", "beta2"):
        number(name, block[name], maximum=1)
        if block[name] == 1:
            raise ValueError(f"training.{name} must be < 1")
    for name, choices in (
        ("optimizer", ("adamw",)),
        ("scheduler", ("cosine",)),
        ("precision", ("auto", "fp32", "fp16", "bf16")),
    ):
        if block[name] not in choices:
            raise ValueError(f"training.{name} must be one of {choices}")
    if block["sequence_stride"] > context_length:
        raise ValueError("training.sequence_stride must not exceed context_length")
    if block["persistent_workers"] and not block["num_workers"]:
        raise ValueError("training.persistent_workers requires num_workers > 0")
    if not block["mixed_precision"] and block["precision"] in ("fp16", "bf16"):
        raise ValueError("training.precision requires mixed_precision=true")
    checkpoint = block["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "save_every_steps",
        "save_every_epoch",
        "keep_last_n",
    }:
        raise ValueError("training.checkpoint fields invalid")
    integer("checkpoint.save_every_steps", checkpoint["save_every_steps"])
    integer("checkpoint.keep_last_n", checkpoint["keep_last_n"])
    boolean("checkpoint.save_every_epoch", checkpoint["save_every_epoch"])
    early = block["early_stopping"]
    if not isinstance(early, dict) or set(early) != {"enabled", "patience", "min_delta"}:
        raise ValueError("training.early_stopping fields invalid")
    boolean("early_stopping.enabled", early["enabled"])
    integer("early_stopping.patience", early["patience"])
    number("early_stopping.min_delta", early["min_delta"], inclusive=True)
