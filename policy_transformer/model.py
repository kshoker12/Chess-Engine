import torch
import torch.nn as nn
from torch.nn import functional as F
import math


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embedding to a tensor shaped [B, n_head, T, head_dim].
    cos/sin are shaped [T, head_dim//2] (broadcasted inside).
    """
    # x: [..., head_dim] where head_dim is even
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        head_dim = n_embd // n_head
        assert head_dim % 2 == 0, "RoPE requires an even head dimension"

        # RoPE (rotary position embeddings) cache (non-persistent so old checkpoints still load)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("_rope_inv_freq", inv_freq, persistent=False)
        self.register_buffer("_rope_cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("_rope_sin_cached", torch.empty(0), persistent=False)
        self._rope_seq_len_cached = 0
        self._rope_cache_dtype = None
        self._rope_cache_device = None

        # Flash Attention requires PyTorch 2.0+
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: Flash Attention not available. Using slow manual attention.")
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def _rope_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # Cache per (device, dtype, seq_len>=T). Compute in fp32 for stability, cast for use.
        if (
            self._rope_seq_len_cached >= seq_len
            and self._rope_cache_device == device
            and self._rope_cache_dtype == dtype
            and self._rope_cos_cached.numel() != 0
        ):
            return self._rope_cos_cached[:seq_len], self._rope_sin_cached[:seq_len]

        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self._rope_inv_freq.to(device=device))  # [T, head_dim//2]
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)

        self._rope_cos_cached = cos
        self._rope_sin_cached = sin
        self._rope_seq_len_cached = seq_len
        self._rope_cache_device = device
        self._rope_cache_dtype = dtype
        return cos, sin

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # RoPE: apply rotary embeddings to q/k (not v)
        cos, sin = self._rope_cos_sin(T, device=q.device, dtype=q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # manual implementation (fallback)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedFoward(n_embd, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class PolicyHead(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, device):
        super().__init__()
        self.block_size = block_size
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Ensure T is within block_size
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            T = self.block_size
            if targets is not None:
                targets = targets[:, -self.block_size:]

        tok_emb = self.token_embedding_table(idx) 
        # With RoPE applied inside attention, we don't add absolute positional embeddings here.
        x = tok_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        if targets is None:
            logits = self.lm_head(x[:, [-1], :]) # optimization: only compute last logit for inference
            loss = None
        else:
            logits = self.lm_head(x)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=-100) # Added ignore_index explicitly
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx