"""Explicit Q/K/V causal multi-head attention using PyTorch SDPA."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads, self.head_dim = config.num_heads, config.head_dim
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim, bias=config.bias)
        self.proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias)
        self.dropout = config.dropout

    def forward(self, x, attention_mask=None):
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def heads(t):
            return t.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must match [B,T]")
            causal = torch.ones((length, length), device=x.device, dtype=torch.bool).tril()
            mask = causal[None, None, :, :] & attention_mask[:, None, None, :].to(dtype=torch.bool)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=mask is None,
        )
        return self.proj(output.transpose(1, 2).contiguous().view(batch, length, channels))
