import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        assert (
            d_embed % n_heads == 0
        ), "Embedding dimension must be divisible by the number of heads"
        self.d_head = d_embed // n_heads
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj - nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads

    def forward(self, x, causal_mask=False):
        input_shape = x.shape
        batch_size, sequence_len, d_embed = input_shape

        interim_shape = (batch_size, sequence_len, self.n_heads, self.d_head)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if causal_mask:
            score = score.masked_fill(
                torch.triu(torch.ones_like(score), diagonal=1).bool(), -1e9
            )
        score = F.softmax(score, dim=-1)

        attn = (
            torch.matmul(score, v)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_len, d_embed)
        )

        return self.out_proj(attn)
