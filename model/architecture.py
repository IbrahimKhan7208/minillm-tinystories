"""
MiniLLM: A modern decoder-only Transformer language model.
Architecture: RMSNorm, RoPE, Grouped Query Attention, SwiGLU FFN, pre-norm blocks.
Trained from scratch on TinyStories with a custom byte-level BPE tokenizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization -- simplified LayerNorm (no mean-centering, no bias)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_rope_freqs(head_dim, max_seq_len, base=10000.0):
    """Precompute cosine and sine tables for Rotary Positional Embeddings."""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """Apply rotary embeddings to a Q or K tensor. x: [batch, n_heads, seq_len, head_dim]."""
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)


def repeat_kv(x, n_rep):
    """Repeat KV heads n_rep times to match the number of query heads (for GQA)."""
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    """Multi-head attention with fewer KV heads than query heads (memory/compute efficient)."""
    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, cos, sin):
        batch, seq_len, d_model = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(self.causal_mask[:seq_len, :seq_len], float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = attn_weights @ v

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    """Gated feed-forward network: SiLU-activated gate path multiplied with a linear up path."""
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: x = x + Sublayer(Norm(x)), for both attention and FFN."""
    def __init__(self, d_model, n_heads, n_kv_heads, ffn_hidden_dim, max_seq_len, dropout=0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, max_seq_len)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_hidden_dim)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        x = x + self.attn_dropout(self.attn(self.attn_norm(x), cos, sin))
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))
        return x


class MiniLLM(nn.Module):
    """Full decoder-only Transformer language model."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config["vocab_size"], config["d_model"])
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config["d_model"], n_heads=config["n_heads"],
                n_kv_heads=config["n_kv_heads"], ffn_hidden_dim=config["ffn_hidden_dim"],
                max_seq_len=config["max_seq_len"], dropout=config["dropout"],
            ) for _ in range(config["n_layers"])
        ])
        self.final_norm = RMSNorm(config["d_model"])
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        self.token_emb.weight = self.lm_head.weight

        head_dim = config["d_model"] // config["n_heads"]
        cos, sin = precompute_rope_freqs(head_dim, config["max_seq_len"])
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch, seq_len = idx.shape
        x = self.token_emb(idx)
        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss