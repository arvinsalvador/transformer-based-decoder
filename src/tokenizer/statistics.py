"""Post-training split analysis; vocabulary is never updated here."""

from dataclasses import dataclass, field

from src.tokenizer.corpus import texts


@dataclass
class TokenStatistics:
    """Numeric token lengths are retained for percentiles; input texts are never retained."""

    lengths: list[int] = field(default_factory=list)
    documents: int = 0
    characters: int = 0
    tokens: int = 0
    unknowns: int = 0
    pretokenized_words: int = 0

    def add(self, text: str, encoding, pretokenized_count: int, unk_id: int) -> None:
        count = len(encoding.ids)
        self.documents += 1
        self.characters += len(text)
        self.tokens += count
        self.lengths.append(count)
        self.unknowns += sum(token_id == unk_id for token_id in encoding.ids)
        self.pretokenized_words += pretokenized_count

    def snapshot(self) -> dict:
        ordered = sorted(self.lengths)

        def percentile(point: float) -> float:
            if not ordered:
                return 0
            index = round((len(ordered) - 1) * point)
            return ordered[index]

        return {
            "documents_analyzed": self.documents,
            "total_characters": self.characters,
            "total_wordpiece_tokens": self.tokens,
            "average_tokens_per_document": self.tokens / self.documents if self.documents else 0,
            "min_tokens_per_document": min(ordered, default=0),
            "max_tokens_per_document": max(ordered, default=0),
            "median_tokens_per_document": percentile(0.5),
            "p95_tokens_per_document": percentile(0.95),
            "unknown_token_count": self.unknowns,
            "unknown_token_rate": self.unknowns / self.tokens if self.tokens else 0,
            "average_characters_per_token": self.characters / self.tokens if self.tokens else 0,
            "pretokenized_words": self.pretokenized_words,
            "wordpiece_fertility": self.tokens / self.pretokenized_words
            if self.pretokenized_words
            else 0,
        }


def analyze_split(tokenizer, path, batch_size: int, unk_id: int) -> dict:
    """Encode a split once for statistics without fitting/altering the tokenizer."""
    stats = TokenStatistics()
    for text in texts(path, batch_size):
        encoding = tokenizer.encode(text, add_special_tokens=False)
        words = len(tokenizer.pre_tokenizer.pre_tokenize_str(text))
        stats.add(text, encoding, words, unk_id)
    return stats.snapshot()
