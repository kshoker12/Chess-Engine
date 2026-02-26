import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint # optimization

# -----------------------------------------------------------------------------
# OPTIMIZED ATTENTION (With Relative Position Bias)
# -----------------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, n_embed, n_heads, block_size, dropout):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.dropout = dropout

        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.c_proj = nn.Linear(n_embed, n_embed)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Flash check (same as policy)
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            print("Flash Attention not available - using manual.")
            self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # B nh T hd
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=False  # No causal for value model
            )
        else:
            att = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = torch.matmul(att, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
# -----------------------------------------------------------------------------
# FEED FORWARD
# -----------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(), 
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# TRANSFORMER BLOCK (With Gradient Checkpointing)
# -----------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_heads, block_size, dropout):
        super().__init__()
        self.sa = SelfAttention(n_embed, n_heads, block_size, dropout)
        self.ff = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # Standard Residual Connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# -----------------------------------------------------------------------------
# VALUE HEAD
# -----------------------------------------------------------------------------
class ValueHead(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed // 2),
            nn.GELU(),
            nn.Linear(n_embed // 2, 1),
            nn.Tanh(), 
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x.mean(dim=1) 
        out = self.dropout(out)
        out = self.net(out)
        return out

# -----------------------------------------------------------------------------
# CHESSFORMER (Value Only - Optimized)
# -----------------------------------------------------------------------------
class ChessFormer(nn.Module):
    def __init__(self, block_size=64, n_embed=256, n_heads=4, n_layers=6, dropout=0.1):  # Scaled down: embed 512→256, heads 8→4, layers 8→6 (~8M params)
        super().__init__()  
        self.block_size = block_size
        self.n_embed = n_embed
        
        # Embeddings
        self.piece_embedding = nn.Embedding(13, n_embed) 
        self.rank_embedding = nn.Embedding(8, n_embed)   
        self.file_embedding = nn.Embedding(8, n_embed)   
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, n_heads, block_size, dropout) 
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(n_embed)
        self.value_head = ValueHead(n_embed, dropout)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        B, T = x.shape
        device = x.device
        
        ranks = torch.arange(64, device=device) // 8
        files = torch.arange(64, device=device) % 8
        ranks = ranks.unsqueeze(0).expand(B, -1)
        files = files.unsqueeze(0).expand(B, -1)
        
        x = self.piece_embedding(x) + self.rank_embedding(ranks) + self.file_embedding(files)
        x = self.pos_drop(x)
        
        # Loop with Gradient Checkpointing for memory efficiency
        for block in self.blocks:
            if self.training:
                # Use checkpointing during training to save VRAM
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
                
        x = self.ln_f(x)
        value = self.value_head(x)
        
        return value