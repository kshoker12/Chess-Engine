"""
Warm-start helpers for the unified pipeline.

1) Production policy (vocab 4033) → vocab1 PolicyHead state (1793) via UCI-row remap.
2) Vocab1 PolicyHead → JointChessNet decoder (+ zero FiLM).
3) Production value (ChessFormer 384/6/6) → JointChessNet encoder + value head
   (CLS / castle / EP stay fresh).
"""
from __future__ import annotations

import pickle
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def _strip_orig_mod(state: dict) -> dict:
    return {k.removeprefix("_orig_mod."): v for k, v in state.items()}


def _load_state(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    return _strip_orig_mod(state)


def build_vocab1_policy_state(
    legacy_ckpt_path: str,
    vocab_old_path: str,
    vocab_new_path: str,
) -> Dict[str, torch.Tensor]:
    """
    Remap a production PolicyHead checkpoint (vocab.pkl / 4033) into a vocab1
    (1793) state dict. Every vocab1 token must exist in the old vocab.
    Transformer body (blocks / ln_f / position_embedding) is copied 1:1.
    """
    with open(vocab_old_path, "rb") as f:
        stoi_old = pickle.load(f)
    with open(vocab_new_path, "rb") as f:
        stoi_new = pickle.load(f)

    missing = [t for t in stoi_new if t not in stoi_old]
    if missing:
        raise ValueError(
            f"{len(missing)} vocab1 tokens missing from production vocab "
            f"(examples: {missing[:5]})"
        )

    state = _load_state(legacy_ckpt_path)
    emb_old = state["token_embedding_table.weight"]
    lm_w_old = state["lm_head.weight"]
    lm_b_old = state["lm_head.bias"]
    if emb_old.shape[0] != len(stoi_old):
        raise ValueError(
            f"Checkpoint vocab rows {emb_old.shape[0]} != len(vocab_old) {len(stoi_old)}"
        )

    d = emb_old.shape[1]
    v_new = len(stoi_new)
    # Build gather index: new_id -> old_id, sorted by new id
    itos_new = {int(i): s for s, i in stoi_new.items()}
    gather = torch.tensor(
        [int(stoi_old[itos_new[i]]) for i in range(v_new)],
        dtype=torch.long,
    )

    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k in (
            "token_embedding_table.weight",
            "lm_head.weight",
            "lm_head.bias",
        ):
            continue
        out[k] = v

    out["token_embedding_table.weight"] = emb_old.index_select(0, gather).contiguous()
    out["lm_head.weight"] = lm_w_old.index_select(0, gather).contiguous()
    out["lm_head.bias"] = lm_b_old.index_select(0, gather).contiguous()

    if out["token_embedding_table.weight"].shape != (v_new, d):
        raise ValueError(
            f"Remapped emb shape {tuple(out['token_embedding_table.weight'].shape)} "
            f"!= ({v_new}, {d})"
        )
    return out


def load_legacy_policy_decoder(model: nn.Module, ckpt_path: str, strict_vocab: bool = True) -> int:
    """
    Copy legacy PolicyHead weights into model's decoder and zero-init FiLM.
    Returns the number of tensors copied.
    """
    state = _load_state(ckpt_path)

    emb = state["token_embedding_table.weight"]
    if strict_vocab and emb.shape[0] != model.vocab_size:
        raise ValueError(
            f"Legacy vocab {emb.shape[0]} != model vocab {model.vocab_size}; "
            "expected a vocab1 (1793) checkpoint like ultra_v1_o1.pt"
        )
    if emb.shape[1] != model.token_emb.embedding_dim:
        raise ValueError(
            f"Legacy width {emb.shape[1]} != decoder width {model.token_emb.embedding_dim}"
        )

    key_map = {"token_embedding_table.weight": "token_emb.weight"}
    n_layers = len(model.blocks)
    for i in range(n_layers):
        for a, b in (
            (f"blocks.{i}.ln1.weight", f"blocks.{i}.ln1.weight"),
            (f"blocks.{i}.ln1.bias", f"blocks.{i}.ln1.bias"),
            (f"blocks.{i}.sa.c_attn.weight", f"blocks.{i}.sa.c_attn.weight"),
            (f"blocks.{i}.sa.c_proj.weight", f"blocks.{i}.sa.c_proj.weight"),
            (f"blocks.{i}.sa.c_proj.bias", f"blocks.{i}.sa.c_proj.bias"),
            (f"blocks.{i}.ln2.weight", f"blocks.{i}.ln2.weight"),
            (f"blocks.{i}.ln2.bias", f"blocks.{i}.ln2.bias"),
            (f"blocks.{i}.ffwd.net.0.weight", f"blocks.{i}.ff.net.0.weight"),
            (f"blocks.{i}.ffwd.net.0.bias", f"blocks.{i}.ff.net.0.bias"),
            (f"blocks.{i}.ffwd.net.2.weight", f"blocks.{i}.ff.net.2.weight"),
            (f"blocks.{i}.ffwd.net.2.bias", f"blocks.{i}.ff.net.2.bias"),
        ):
            key_map[a] = b
    key_map["ln_f.weight"] = "ln_f.weight"
    key_map["ln_f.bias"] = "ln_f.bias"
    key_map["lm_head.weight"] = "lm_head.weight"
    key_map["lm_head.bias"] = "lm_head.bias"

    own = model.state_dict()
    copied = 0
    for src, dst in key_map.items():
        if src not in state:
            raise KeyError(f"Legacy checkpoint missing key: {src}")
        if own[dst].shape != state[src].shape:
            raise ValueError(
                f"Shape mismatch {src} {tuple(state[src].shape)} vs {dst} {tuple(own[dst].shape)}"
            )
        own[dst] = state[src]
        copied += 1
    model.load_state_dict(own)

    # Zero FiLM so the warm-started decoder is exactly the legacy function at step 0.
    for block in model.blocks:
        nn.init.zeros_(block.film.proj.weight)
        nn.init.zeros_(block.film.proj.bias)
    return copied


def load_legacy_value_encoder(model: nn.Module, ckpt_path: str) -> int:
    """
    Copy production ChessFormer (384/6/6) weights into JointChessNet encoder + value head.
    Leaves cls_token / castle_emb / ep_emb at fresh init (unified-only tokens).
    Returns number of tensors copied.
    """
    state = _load_state(ckpt_path)
    own = model.state_dict()

    key_map = {
        "piece_embedding.weight": "encoder.piece_emb.weight",
        "rank_embedding.weight": "encoder.rank_emb.weight",
        "file_embedding.weight": "encoder.file_emb.weight",
        "ln_f.weight": "encoder.ln_f.weight",
        "ln_f.bias": "encoder.ln_f.bias",
        "value_head.net.0.weight": "value_head.net.0.weight",
        "value_head.net.0.bias": "value_head.net.0.bias",
        "value_head.net.2.weight": "value_head.net.2.weight",
        "value_head.net.2.bias": "value_head.net.2.bias",
    }
    n_layers = len(model.encoder.blocks)
    for i in range(n_layers):
        for a, b in (
            (f"blocks.{i}.ln1.weight", f"encoder.blocks.{i}.ln1.weight"),
            (f"blocks.{i}.ln1.bias", f"encoder.blocks.{i}.ln1.bias"),
            (f"blocks.{i}.sa.c_attn.weight", f"encoder.blocks.{i}.sa.c_attn.weight"),
            (f"blocks.{i}.sa.c_proj.weight", f"encoder.blocks.{i}.sa.c_proj.weight"),
            (f"blocks.{i}.sa.c_proj.bias", f"encoder.blocks.{i}.sa.c_proj.bias"),
            (f"blocks.{i}.ln2.weight", f"encoder.blocks.{i}.ln2.weight"),
            (f"blocks.{i}.ln2.bias", f"encoder.blocks.{i}.ln2.bias"),
            (f"blocks.{i}.ff.net.0.weight", f"encoder.blocks.{i}.ff.net.0.weight"),
            (f"blocks.{i}.ff.net.0.bias", f"encoder.blocks.{i}.ff.net.0.bias"),
            (f"blocks.{i}.ff.net.2.weight", f"encoder.blocks.{i}.ff.net.2.weight"),
            (f"blocks.{i}.ff.net.2.bias", f"encoder.blocks.{i}.ff.net.2.bias"),
        ):
            key_map[a] = b

    copied = 0
    for src, dst in key_map.items():
        if src not in state:
            raise KeyError(f"Legacy value checkpoint missing key: {src}")
        if own[dst].shape != state[src].shape:
            raise ValueError(
                f"Shape mismatch {src} {tuple(state[src].shape)} vs {dst} {tuple(own[dst].shape)}"
            )
        own[dst] = state[src]
        copied += 1
    model.load_state_dict(own)
    return copied


def fresh_and_pretrained_param_groups(model: nn.Module):
    """
    Split trainable Phase B params for two learning rates:
    fresh (FiLM + cond_proj, random/zero init) vs pretrained legacy decoder weights.
    """
    fresh, pretrained = [], []
    fresh += [p for p in model.cond_proj.parameters() if p.requires_grad]
    for block in model.blocks:
        fresh += [p for p in block.film.proj.parameters() if p.requires_grad]
        for mod in (block.ln1, block.sa, block.ln2, block.ff):
            pretrained += [p for p in mod.parameters() if p.requires_grad]
    for mod in (model.token_emb, model.ln_f, model.lm_head):
        pretrained += [p for p in mod.parameters() if p.requires_grad]
    return fresh, pretrained


def fresh_and_pretrained_value_groups(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Phase A dual LR: fresh = CLS / castle / EP embeds; pretrained = rest of encoder+value.
    """
    fresh_ids = {
        id(model.encoder.cls_token),
        *[id(p) for p in model.encoder.castle_emb.parameters()],
        *[id(p) for p in model.encoder.ep_emb.parameters()],
    }
    fresh, pretrained = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in fresh_ids:
            fresh.append(p)
        else:
            pretrained.append(p)
    return fresh, pretrained
