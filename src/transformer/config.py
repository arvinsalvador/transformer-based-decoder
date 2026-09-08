"""Validated architecture settings, independent of training concerns."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    pad_token_id: int | None
    context_length: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    layer_norm_eps: float = 1e-5
    bias: bool = True
    tie_embeddings: bool = True
    initialization_std: float = 0.02

    def __post_init__(self):
        if self.vocab_size <= 0 or self.context_length <= 0 or self.embedding_dim <= 0:
            raise ValueError(
                "vocabulary size, context length, and embedding dimension must be positive"
            )
        if self.num_layers < 1 or self.num_heads < 1 or self.feedforward_dim <= 0:
            raise ValueError("layers, heads, and feed-forward dimension must be positive")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if not 0 <= self.dropout < 1 or self.layer_norm_eps <= 0 or self.initialization_std <= 0:
            raise ValueError("dropout must be [0,1); eps and initialization std must be positive")

    @property
    def head_dim(self):
        return self.embedding_dim // self.num_heads

    def to_dict(self):
        return asdict(self)
