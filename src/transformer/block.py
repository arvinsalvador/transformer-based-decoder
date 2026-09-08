"""Pre-norm decoder block."""

from torch import nn

from src.transformer.attention import CausalSelfAttention


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feedforward_dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.feedforward_dim, config.embedding_dim, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attention(self.ln1(x), attention_mask)
        return x + self.ffn(self.ln2(x))
