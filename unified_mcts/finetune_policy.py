#!/usr/bin/env python3
"""Fine-tune policy: freeze encoder + value; train decoder/FiLM. Mix FT + base replay."""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MixedPolicyFTDataset
from model import JointChessNet, count_parameters, count_trainable




def save_checkpoint(obj, path: str) -> None:
    # Legacy pickle .pt — default torch.save is a zip; Kaggle lists/downloads it as .zip.
    torch.save(obj, path, _use_new_zipfile_serialization=False)


def get_lr(it, warmup, decay, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > decay:
        return min_lr
    ratio = (it - warmup) / max(1, decay - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


def load_ckpt(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    own = model.state_dict()
    loaded = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    own.update(loaded)
    model.load_state_dict(own)
    return len(loaded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Joint checkpoint to fine-tune")
    ap.add_argument("--ft-data", required=True, help="Self-play policy_ft dir")
    ap.add_argument("--base-data", default=None, help="Base policy memmap dir (replay)")
    ap.add_argument("--ft-frac", type=float, default=0.8)
    ap.add_argument("--out", default="unified_ft_policy.pt")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-iters", type=int, default=10_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--vocab-size", type=int, default=1793)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MixedPolicyFTDataset(args.ft_data, args.base_data, ft_frac=args.ft_frac)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True
    )

    model = JointChessNet(vocab_size=args.vocab_size, dropout=0.1).to(device)
    n_loaded = load_ckpt(model, args.ckpt, device)
    model.set_train_mode("ft_policy")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    print(
        f"Loaded {n_loaded} tensors; trainable={count_trainable(model):.2f}M / "
        f"total={count_parameters(model):.2f}M ft_frac={args.ft_frac}"
    )

    it = 0
    data_iter = iter(loader)
    pbar = tqdm(total=args.max_iters, desc="FT_policy")
    while it < args.max_iters:
        for g in opt.param_groups:
            g["lr"] = get_lr(it, args.warmup, args.max_iters, args.lr, args.lr / 10)
        try:
            hist, board, rights, y_pi, _yv = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            hist, board, rights, y_pi, _yv = next(data_iter)
        hist = hist.to(device)
        board = board.to(device)
        rights = rights.to(device)
        y_pi = y_pi.to(device)

        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            out = model(
                hist, board, rights, y_pi=y_pi, compute_policy=True, compute_value=False
            )
            loss = out["loss_pi"]
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()

        if it % 50 == 0:
            pbar.set_postfix(loss=float(loss), lr=opt.param_groups[0]["lr"])
        it += 1
        pbar.update(1)
        if it % args.save_every == 0:
            save_checkpoint(
                {"model": model.state_dict(), "config": model.config_dict(), "iter": it},
                args.out,
            )
            print(f"\nSaved {args.out} at iter {it}")

    pbar.close()
    save_checkpoint({"model": model.state_dict(), "config": model.config_dict(), "iter": it}, args.out)
    print(f"Finished → {args.out}")


if __name__ == "__main__":
    main()
