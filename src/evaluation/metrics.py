"""Natural-log metrics, typed results and guarded relative comparisons."""

import math
from dataclasses import asdict, dataclass

from src.training.monitoring import perplexity


@dataclass(frozen=True)
class ModelEvaluationResult:
    model_type: str
    documents: int
    prediction_events: int
    total_nll: float
    evaluation_seconds: float
    device: str
    precision: str
    metadata: dict

    def to_dict(self):
        if self.prediction_events <= 0:
            raise ValueError("Evaluation has no prediction events")
        nll = self.total_nll / self.prediction_events
        if not math.isfinite(nll):
            raise ValueError("Nonfinite evaluation NLL")
        return {
            **asdict(self),
            "average_nll": nll,
            "total_log_likelihood": -self.total_nll,
            "perplexity": perplexity(nll),
            "bits_per_token": nll / math.log(2),
            "tokens_per_second": self.prediction_events / self.evaluation_seconds
            if self.evaluation_seconds > 0
            else None,
            **self.metadata,
        }


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    return numerator / denominator


def compare(trigram, transformer):
    for key in ("documents", "prediction_events", "test_fingerprint", "tokenizer_fingerprint"):
        if trigram[key] != transformer[key]:
            raise ValueError(f"Comparison mismatch: {key}")
    ppl_ratio = ratio(transformer["perplexity"], trigram["perplexity"])
    relative = {
        "perplexity_reduction_percent": (1 - ppl_ratio) * 100 if ppl_ratio is not None else None,
        "training_time_multiplier": ratio(
            transformer["training_duration_seconds"], trigram["training_duration_seconds"]
        ),
        "model_size_multiplier": ratio(
            transformer["model_size_bytes"], trigram["model_size_bytes"]
        ),
        "evaluation_throughput_ratio": ratio(
            transformer["tokens_per_second"], trigram["tokens_per_second"]
        ),
    }
    statements = []
    if ppl_ratio is not None:
        statements.append(
            "Transformer test perplexity is "
            + ("lower." if ppl_ratio < 1 else "higher." if ppl_ratio > 1 else "equal.")
        )
    multiplier = relative["training_time_multiplier"]
    if multiplier is not None:
        statements.append(
            f"Recorded Transformer model-training time is {multiplier:.3g} times the trigram time."
        )
    statements.append(
        "Timing depends on hardware and software; "
        "this is not an architecture-only speed comparison."
    )
    return relative, statements
