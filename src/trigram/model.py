"""CPU-only trigram counts, Lidstone probabilities, scoring, and generation."""

import math
import random
from collections import Counter
from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class Score:
    documents: int
    events: int
    log_likelihood: float
    average_nll: float
    perplexity: float
    duration_seconds: float


class TrigramModel:
    """Counts only token IDs; every supplied sequence is one independent document."""

    def __init__(self, vocabulary_size: int, bos_id: int, eos_id: int, add_k: float):
        self.vocabulary_size, self.bos_id, self.eos_id, self.add_k = (
            vocabulary_size,
            bos_id,
            eos_id,
            add_k,
        )
        self.unigrams: Counter[int] = Counter()
        self.bigrams: Counter[tuple[int, int]] = Counter()
        self.trigrams: Counter[tuple[int, int, int]] = Counter()

    def update(self, token_ids: list[int]) -> int:
        sequence = [self.bos_id, self.bos_id, *token_ids, self.eos_id]
        self.unigrams.update(sequence)
        self.bigrams.update(zip(sequence, sequence[1:], strict=False))
        triples = list(zip(sequence, sequence[1:], sequence[2:], strict=False))
        self.trigrams.update(triples)
        return len(triples)

    def probability(self, first: int, second: int, third: int) -> float:
        return (self.trigrams[(first, second, third)] + self.add_k) / (
            self.bigrams[(first, second)] + self.add_k * self.vocabulary_size
        )

    def score_sequences(self, sequences) -> Score:
        start, documents, events, log_likelihood = perf_counter(), 0, 0, 0.0
        for token_ids in sequences:
            documents += 1
            sequence = [self.bos_id, self.bos_id, *token_ids, self.eos_id]
            for first, second, third in zip(sequence, sequence[1:], sequence[2:], strict=False):
                log_likelihood += math.log(self.probability(first, second, third))
                events += 1
        nll = -log_likelihood / events if events else 0.0
        return Score(
            documents,
            events,
            log_likelihood,
            nll,
            math.exp(nll) if events else 1.0,
            perf_counter() - start,
        )

    def candidates(self, first: int, second: int) -> list[tuple[int, float]]:
        observed = [
            (third, count)
            for (left, middle, third), count in self.trigrams.items()
            if (left, middle) == (first, second)
        ]
        if observed:
            return [(token, self.probability(first, second, token)) for token, _ in observed]
        # Observed unigram fallback avoids an O(|V|) scan for unseen contexts.
        total = sum(self.unigrams.values()) or 1
        return [
            (token, count / total) for token, count in self.unigrams.items() if token != self.bos_id
        ]

    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 50,
        seed: int | None = None,
    ) -> tuple[list[int], dict]:
        start, rng, generated = perf_counter(), random.Random(seed), []
        context = (
            [self.bos_id, self.bos_id]
            if not prompt_ids
            else [self.bos_id, prompt_ids[-1]]
            if len(prompt_ids) == 1
            else prompt_ids[-2:]
        )
        stopped_on_eos = False
        for _ in range(max_new_tokens):
            choices = sorted(self.candidates(*context), key=lambda item: (-item[1], item[0]))[
                :top_k
            ]
            if not choices:
                break
            if strategy == "greedy":
                next_id = choices[0][0]
            else:
                weights = [prob ** (1 / temperature) for _, prob in choices]
                next_id = rng.choices([token for token, _ in choices], weights=weights, k=1)[0]
            if next_id == self.eos_id:
                stopped_on_eos = True
                break
            generated.append(next_id)
            context = [context[-1], next_id]
        duration = perf_counter() - start
        return generated, {
            "stopped_on_eos": stopped_on_eos,
            "generated_tokens": len(generated),
            "duration_seconds": duration,
            "tokens_per_second": len(generated) / duration if duration else 0.0,
        }

    def statistics(self) -> dict:
        return {
            "unique_unigrams": len(self.unigrams),
            "unique_bigrams": len(self.bigrams),
            "unique_trigrams": len(self.trigrams),
            "tokens_observed": sum(self.unigrams.values()),
            "trigram_events": sum(self.trigrams.values()),
        }

    def close(self):
        """Memory models own no external resources."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
