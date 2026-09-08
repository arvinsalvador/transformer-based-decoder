"""Streaming preparation statistics; numeric lengths are retained only for median."""

from dataclasses import dataclass, field

from src.data.preprocessing.models import PreprocessingStatus


@dataclass
class PreparationStatistics:
    """At most one integer per accepted document is retained (<=100,000 by configuration)."""

    input_documents: int = 0
    accepted_documents: int = 0
    total_input_characters: int = 0
    total_output_characters: int = 0
    truncated: int = 0
    lengths: list[int] = field(default_factory=list)
    source_types: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, int] = field(default_factory=dict)
    splits: dict[str, int] = field(default_factory=lambda: {"train": 0, "validation": 0, "test": 0})

    def input(self, characters: int, source_type: str) -> None:
        self.input_documents += 1
        self.total_input_characters += characters
        self.source_types[source_type] = self.source_types.get(source_type, 0) + 1

    def reject(self, status: PreprocessingStatus) -> None:
        key = status.value
        self.reasons[key] = self.reasons.get(key, 0) + 1

    def accept(self, characters: int, split: str, truncated: bool) -> None:
        self.accepted_documents += 1
        self.total_output_characters += characters
        self.lengths.append(characters)
        self.splits[split] += 1
        self.truncated += truncated

    def snapshot(self) -> dict:
        ordered = sorted(self.lengths)
        middle = len(ordered) // 2
        median = (
            (ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2)
            if ordered
            else 0
        )
        accepted = self.accepted_documents
        return {
            "input_documents": self.input_documents,
            "accepted_documents": accepted,
            "rejected_documents": self.input_documents - accepted,
            "duplicates": self.reasons.get(PreprocessingStatus.DUPLICATE.value, 0),
            "truncated": self.truncated,
            "total_input_characters": self.total_input_characters,
            "total_output_characters": self.total_output_characters,
            "acceptance_rate": accepted / self.input_documents if self.input_documents else 0,
            "min_document_characters": min(ordered, default=0),
            "max_document_characters": max(ordered, default=0),
            "mean_document_characters": self.total_output_characters / accepted if accepted else 0,
            "median_document_characters": median,
            "document_type_distribution": dict(self.source_types),
            "rejection_reason_distribution": dict(self.reasons),
            "split_counts": dict(self.splits),
        }
