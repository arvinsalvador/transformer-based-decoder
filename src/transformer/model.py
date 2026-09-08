"""Scratch-initialized decoder-only Transformer returning unnormalized logits."""

import torch
from torch import nn

from src.transformer.block import DecoderBlock


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=config.pad_token_id
        )
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.final_ln = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.apply(self._initialize)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        if config.pad_token_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[config.pad_token_id].zero_()

    def _initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initialization_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None):
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be nonempty [B,T]")
        if input_ids.dtype not in (torch.int64, torch.int32):
            raise ValueError("input_ids must be integer")
        if input_ids.shape[1] > self.config.context_length:
            raise ValueError("Sequence length exceeds configured context length")
        if input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size:
            raise ValueError("input IDs outside vocabulary")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.lm_head(self.final_ln(x))
