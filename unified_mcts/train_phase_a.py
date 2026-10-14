#!/usr/bin/env python3
"""Phase A: value-only pretrain of BoardEncoder + ValueHead."""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ValueMemmapDataset
from model import JointChessNet, count_parameters
from warmstart import fresh_and_pretrained_value_groups, load_legacy_value_encoder




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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dir with value_*.memmap + value_meta.json")
    ap.add_argument("--out", default="unified_phase_a.pt")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-iters", type=int, default=30_000)
    ap.add_argument("--lr", type=float, default=3e-4, help="Peak LR for fresh params (CLS/castle/EP)")
    ap.add_argument(
        "--lr-pretrained",
        type=float,
        default=1e-4,
        help="Peak LR for production-warm-started encoder/value weights",
    )
    ap.add_argument(
        "--legacy-value",
        default=None,
        help="Production ChessFormer ckpt (e.g. mini_value_6o4.pt) to warm-start encoder+value",
    )
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ValueMemmapDataset(args.data, split="train")
    val_ds = ValueMemmapDataset(args.data, split="val")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = JointChessNet(dropout=0.1).to(device)
    if args.legacy_value:
        n_copied = load_legacy_value_encoder(model, args.legacy_value)
        print(f"Warm-started encoder+value from {args.legacy_value} ({n_copied} tensors; CLS/castle/EP fresh)")
    model.set_train_mode("phase_a")

    params = [p for p in model.parameters() if p.requires_grad]
    if args.legacy_value:
        fresh, pretrained = fresh_and_pretrained_value_groups(model)
        opt = torch.optim.AdamW(
            [
                {"params": fresh, "lr": args.lr, "base_lr": args.lr},
                {"params": pretrained, "lr": args.lr_pretrained, "base_lr": args.lr_pretrained},
            ],
            weight_decay=0.01,
        )
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        opt.param_groups[0]["base_lr"] = args.lr
    scaler = GradScaler(enabled=(device.type == "cuda"))
    print(f"Device={device} trainable params≈{sum(p.numel() for p in params)/1e6:.2f}M / total {count_parameters(model):.2f}M")

    it = 0
    best = 1e9
    train_iter = iter(train_loader)
    pbar = tqdm(total=args.max_iters, desc="Training", unit="iter", dynamic_ncols=True)
    t0 = time.time()

    while it < args.max_iters:
        for g in opt.param_groups:
            base = g["base_lr"]
            g["lr"] = get_lr(it, args.warmup, args.max_iters, base, base / 10)
        lr = opt.param_groups[0]["lr"]
        try:
            board, rights, yv = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            board, rights, yv = next(train_iter)
        board, rights, yv = board.to(device), rights.to(device), yv.to(device)

        opt.zero_grad(set_to_none=True)
        with autocast(device_type=device.type if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
            # dummy hist
            hist = torch.zeros(board.size(0), model.block_size, dtype=torch.long, device=device)
            out = model(hist, board, rights, y_pi=None, y_v=yv, compute_policy=False, compute_value=True)
            loss = out["loss_v"]
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()

        loss_f = float(loss.detach())
        # Same pattern as value_transformer/train.py (tqdm shows %, it/s, remaining).
        pbar.set_description(f"Iter {it} | MSE {loss_f:.5f} | LR {lr:.2e}")
        # Plain print for Kaggle Save Version logs (tqdm \\r bars often don't show there).
        if it % 50 == 0:
            elapsed = time.time() - t0
            rate = (it + 1) / max(elapsed, 1e-6)
            eta_s = (args.max_iters - it - 1) / max(rate, 1e-6)
            eta_m, eta_sec = divmod(int(eta_s), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            print(
                f"step {it}: train loss {loss_f:.5f}, lr {lr:.2e}, "
                f"{rate:.1f} it/s, remaining {eta_h:d}:{eta_m:02d}:{eta_sec:02d}",
                flush=True,
            )
        if it > 0 and it % args.eval_every == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for board, rights, yv in val_loader:
                    board, rights, yv = board.to(device), rights.to(device), yv.to(device)
                    hist = torch.zeros(board.size(0), model.block_size, dtype=torch.long, device=device)
                    out = model(hist, board, rights, compute_policy=False, compute_value=True, y_v=yv)
                    losses.append(float(out["loss_v"]))
                    if len(losses) >= 20:
                        break
            val = sum(losses) / len(losses)
            print(f"\niter {it}: val_mse={val:.5f}")
            if val < best:
                best = val
                save_checkpoint(
                    {"model": model.state_dict(), "config": model.config_dict(), "iter": it, "best_val": best},
                    args.out.replace(".pt", "_best.pt"),
                )
            model.train()

        if it > 0 and it % args.save_every == 0:
            save_checkpoint(
                {"model": model.state_dict(), "config": model.config_dict(), "iter": it, "best_val": best},
                args.out,
            )
        it += 1
        pbar.update(1)

    pbar.close()
    save_checkpoint(
        {"model": model.state_dict(), "config": model.config_dict(), "iter": it, "best_val": best},
        args.out,
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
