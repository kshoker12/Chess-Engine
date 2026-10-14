"""
JointChessNet: board encoder (CLS) + FiLM-conditioned UCI decoder + policy/value heads.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


class BoardSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BoardEncoderBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = BoardSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class BoardEncoder(nn.Module):
    def __init__(
        self,
        n_embd: int = 384,
        n_head: int = 6,
        n_layer: int = 6,
        dropout: float = 0.1,
        n_rights_tokens: int = 2,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.piece_emb = nn.Embedding(13, n_embd)
        self.rank_emb = nn.Embedding(8, n_embd)
        self.file_emb = nn.Embedding(8, n_embd)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.castle_emb = nn.Embedding(16, n_embd)
        self.ep_emb = nn.Embedding(9, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [BoardEncoderBlock(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, board: torch.Tensor, rights: torch.Tensor) -> torch.Tensor:
        """
        board: [B, 64] long
        rights: [B, 8] long/byte
        returns CLS summary [B, d_b]
        """
        B = board.size(0)
        device = board.device
        ranks = torch.arange(64, device=device) // 8
        files = torch.arange(64, device=device) % 8
        ranks = ranks.unsqueeze(0).expand(B, -1)
        files = files.unsqueeze(0).expand(B, -1)

        e = self.piece_emb(board) + self.rank_emb(ranks) + self.file_emb(files)

        castle_ids = (
            rights[:, 0]
            + 2 * rights[:, 1]
            + 4 * rights[:, 2]
            + 8 * rights[:, 3]
        ).long()
        ep_ids = torch.where(
            rights[:, 4] > 0, 1 + rights[:, 5].long(), torch.zeros(B, device=device, dtype=torch.long)
        )

        cls = self.cls_token.expand(B, -1, -1)
        castle_tok = self.castle_emb(castle_ids).unsqueeze(1)
        ep_tok = self.ep_emb(ep_ids).unsqueeze(1)
        x = torch.cat([cls, e, castle_tok, ep_tok], dim=1)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x[:, 0]


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        head_dim = n_embd // n_head
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("_rope_inv_freq", inv_freq, persistent=False)
        self.register_buffer("_rope_cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("_rope_sin_cached", torch.empty(0), persistent=False)
        self._rope_seq_len_cached = 0
        self._rope_cache_dtype = None
        self._rope_cache_device = None
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
            )

    def _rope_cos_sin(self, seq_len, device, dtype):
        if (
            self._rope_seq_len_cached >= seq_len
            and self._rope_cache_device == device
            and self._rope_cache_dtype == dtype
            and self._rope_cos_cached.numel() != 0
        ):
            return self._rope_cos_cached[:seq_len], self._rope_sin_cached[:seq_len]
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self._rope_inv_freq.to(device=device))
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        self._rope_cos_cached = cos
        self._rope_sin_cached = sin
        self._rope_seq_len_cached = seq_len
        self._rope_cache_device = device
        self._rope_cache_dtype = dtype
        return cos, sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        cos, sin = self._rope_cos_sin(T, q.device, q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class FiLM(nn.Module):
    def __init__(self, d_c: int, d: int):
        super().__init__()
        self.proj = nn.Linear(d_c, 2 * d)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # h: [B,T,d], c: [B,d_c]
        gb = self.proj(c)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return (1.0 + gamma) * h + beta


class DecoderBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, d_c: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.film = FiLM(d_c, n_embd)
        self.sa = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.film(self.ln1(x), c)
        x = x + self.sa(h)
        x = x + self.ff(self.ln2(x))
        return x


class ValueHead(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd // 2),
            nn.GELU(),
            nn.Linear(n_embd // 2, 1),
            nn.Tanh(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(self.drop(s)).squeeze(-1)


class JointChessNet(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1793,
        block_size: int = 128,
        d_b: int = 384,
        n_head_b: int = 6,
        n_layer_b: int = 6,
        d: int = 500,
        n_head: int = 10,
        n_layer: int = 8,
        d_c: int | None = None,  # defaults to decoder width d
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_c is None:
            d_c = d
        if d_c != d:
            raise ValueError(f"FiLM condition dim d_c={d_c} must match decoder embedding d={d}")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_c = d_c
        self.encoder = BoardEncoder(d_b, n_head_b, n_layer_b, dropout)
        self.cond_proj = nn.Linear(d_b, d_c)  # board summary → decoder width
        self.token_emb = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d, n_head, block_size, d_c, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size)
        self.value_head = ValueHead(d_b, dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode_board(self, board: torch.Tensor, rights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.encoder(board.long(), rights.long())
        c = self.cond_proj(s)
        return s, c

    def forward_policy(
        self,
        hist: torch.Tensor,
        cond: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        hist: [B, T]
        cond: [B, d_c]
        targets: [B] next-move ids (single-target) or None for logits only
        """
        B, T = hist.shape
        if T > self.block_size:
            hist = hist[:, -self.block_size :]
            T = self.block_size
        x = self.token_emb(hist.long())
        for block in self.blocks:
            x = block(x, cond)
        x = self.ln_f(x)
        logits = self.lm_head(x[:, -1, :])  # [B, V]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets.long())
        return logits, loss

    def forward_value(self, summary: torch.Tensor, y_v: Optional[torch.Tensor] = None):
        v = self.value_head(summary)
        loss = None
        if y_v is not None:
            loss = F.mse_loss(v, y_v.float())
        return v, loss

    def forward(
        self,
        hist: torch.Tensor,
        board: torch.Tensor,
        rights: torch.Tensor,
        y_pi: Optional[torch.Tensor] = None,
        y_v: Optional[torch.Tensor] = None,
        compute_policy: bool = True,
        compute_value: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        s, c = self.encode_board(board, rights)
        out: Dict[str, Optional[torch.Tensor]] = {
            "summary": s,
            "cond": c,
            "logits": None,
            "v": None,
            "loss_pi": None,
            "loss_v": None,
        }
        if compute_policy:
            logits, loss_pi = self.forward_policy(hist, c, y_pi)
            out["logits"] = logits
            out["loss_pi"] = loss_pi
        if compute_value:
            v, loss_v = self.forward_value(s, y_v)
            out["v"] = v
            out["loss_v"] = loss_v
        return out

    def config_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "d_b": self.encoder.n_embd,
            "n_head_b": self.encoder.blocks[0].sa.n_head,
            "n_layer_b": len(self.encoder.blocks),
            "d": self.token_emb.embedding_dim,
            "n_head": self.blocks[0].sa.n_head,
            "n_layer": len(self.blocks),
            "d_c": self.d_c,
        }

    def encoder_params(self):
        return list(self.encoder.parameters()) + list(self.cond_proj.parameters())

    def value_params(self):
        return list(self.value_head.parameters())

    def decoder_params(self):
        return (
            list(self.token_emb.parameters())
            + list(self.blocks.parameters())
            + list(self.ln_f.parameters())
            + list(self.lm_head.parameters())
        )

    def set_train_mode(self, mode: str):
        """
        Freeze schedules:
          phase_a / ft_value  — train encoder+value; freeze decoder
          phase_b / ft_policy — train decoder; freeze encoder+value
          all                 — everything trainable
        """
        if mode in ("phase_a", "ft_value"):
            set_requires_grad(self.encoder, True)
            set_requires_grad(self.cond_proj, False)
            set_requires_grad(self.value_head, True)
            set_requires_grad(self.token_emb, False)
            set_requires_grad(self.blocks, False)
            set_requires_grad(self.ln_f, False)
            set_requires_grad(self.lm_head, False)
        elif mode in ("phase_b", "ft_policy"):
            set_requires_grad(self.encoder, False)
            set_requires_grad(self.cond_proj, True)
            set_requires_grad(self.value_head, False)
            set_requires_grad(self.token_emb, True)
            set_requires_grad(self.blocks, True)
            set_requires_grad(self.ln_f, True)
            set_requires_grad(self.lm_head, True)
        elif mode == "all":
            for p in self.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown train mode: {mode}")


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_trainable(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
