#!/usr/bin/env python3
"""Phase B: policy CE only with frozen encoder + value head (no SF / Bernoulli)."""
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
from dataset import MixedPolicyDataset, PtReplayPolicyDataset
from model import JointChessNet, count_parameters, count_trainable
from warmstart import fresh_and_pretrained_param_groups, load_legacy_policy_decoder


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


def load_policy_dataset(data_dir: str):
    meta_path = os.path.join(data_dir, "policy_meta.json")
    with open(meta_path) as f:
        import json

        meta = json.load(f)
    if meta.get("format") == "pt_replay":
        return PtReplayPolicyDataset(data_dir)
    return MixedPolicyDataset(data_dir)


@torch.no_grad()
def estimate_ce(model, data_iter, loader, device, n_batches: int) -> float:
    """Mean CE over n_batches (streaming replay — same mix as train, eval mode)."""
    model.eval()
    losses = []
    for _ in range(n_batches):
        try:
            hist, board, rights, y_pi, _yv = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            hist, board, rights, y_pi, _yv = next(data_iter)
        hist = hist.to(device, non_blocking=True)
        board = board.to(device, non_blocking=True)
        rights = rights.to(device, non_blocking=True)
        y_pi = y_pi.to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            out = model(hist, board, rights, y_pi=y_pi, compute_policy=True, compute_value=False)
            losses.append(float(out["loss_pi"].detach()))
    model.train()
    return sum(losses) / max(1, len(losses)), data_iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dir with policy_meta.json (+ memmaps or pt_replay indices)")
    ap.add_argument("--phase-a", required=True, help="Checkpoint from Phase A")
    ap.add_argument("--out", default="unified_phase_b.pt")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--grad-accum", type=int, default=2, help="Microbatches per optimizer step")
    ap.add_argument("--max-iters", type=int, default=30_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--vocab-size", type=int, default=1793)
    ap.add_argument(
        "--legacy-policy",
        default=None,
        help="Legacy vocab1 PolicyHead checkpoint (e.g. ultra_v1_o1.pt) to warm-start the decoder",
    )
    ap.add_argument(
        "--lr-pretrained",
        type=float,
        default=1e-4,
        help="Peak LR for warm-started decoder weights (fresh FiLM/cond_proj use --lr)",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_policy_dataset(args.data)
    print(f"Policy dataset: {type(ds).__name__} len={len(ds):,}")
    # Enough samples for train + periodic evals
    n_needed = (args.max_iters * args.grad_accum + (args.max_iters // args.eval_every + 1) * args.eval_batches) * args.batch_size
    sampler = ds.make_sampler(num_samples=max(len(ds), n_needed))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        drop_last=True,
    )

    model = JointChessNet(vocab_size=args.vocab_size, dropout=0.1).to(device)
    ckpt = torch.load(args.phase_a, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    own = model.state_dict()
    loaded = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    own.update(loaded)
    model.load_state_dict(own)
    print(f"Loaded {len(loaded)} tensors from Phase A")

    if args.legacy_policy:
        n_copied = load_legacy_policy_decoder(model, args.legacy_policy)
        print(f"Warm-started decoder from {args.legacy_policy} ({n_copied} tensors, FiLM zeroed)")

    model.set_train_mode("phase_b")
    params = [p for p in model.parameters() if p.requires_grad]
    if args.legacy_policy:
        fresh, pretrained = fresh_and_pretrained_param_groups(model)
        opt = torch.optim.AdamW(
            [
                {"params": fresh, "lr": args.lr, "base_lr": args.lr},
                {"params": pretrained, "lr": args.lr_pretrained, "base_lr": args.lr_pretrained},
            ],
            weight_decay=1e-4,
        )
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
        opt.param_groups[0]["base_lr"] = args.lr
    scaler = GradScaler(enabled=(device.type == "cuda"))
    print(
        f"Device={device} trainable={count_trainable(model):.2f}M / total={count_parameters(model):.2f}M "
        f"(encoder+value frozen) batch={args.batch_size} accum={args.grad_accum} "
        f"eff_batch={args.batch_size * args.grad_accum}"
    )

    it = 0
    best_val = 1e9
    data_iter = iter(loader)
    pbar = tqdm(total=args.max_iters, desc="Training", unit="iter", dynamic_ncols=True)
    t0 = time.time()

    while it < args.max_iters:
        for g in opt.param_groups:
            base = g["base_lr"]
            g["lr"] = get_lr(it, args.warmup, args.max_iters, base, base / 10)
        lr = opt.param_groups[0]["lr"]

        if it > 0 and it % args.eval_every == 0:
            val_ce, data_iter = estimate_ce(model, data_iter, loader, device, args.eval_batches)
            print(
                f"step {it}: train loss {float(last_train_loss):.4f}, val loss {val_ce:.4f}, lr {lr:.2e}",
                flush=True,
            )
            if val_ce < best_val:
                best_val = val_ce
                best_path = args.out.replace(".pt", "_best.pt")
                save_checkpoint(
                    {
                        "model": model.state_dict(),
                        "config": model.config_dict(),
                        "iter": it,
                        "best_val": best_val,
                    },
                    best_path,
                )
                print(f"saved best {best_path} val={best_val:.4f}", flush=True)

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                hist, board, rights, y_pi, _yv = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                hist, board, rights, y_pi, _yv = next(data_iter)
            hist = hist.to(device, non_blocking=True)
            board = board.to(device, non_blocking=True)
            rights = rights.to(device, non_blocking=True)
            y_pi = y_pi.to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                out = model(
                    hist,
                    board,
                    rights,
                    y_pi=y_pi,
                    compute_policy=True,
                    compute_value=False,
                )
                loss = out["loss_pi"] / args.grad_accum
            scaler.scale(loss).backward()
            accum_loss += float(loss.detach()) * args.grad_accum

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()

        last_train_loss = accum_loss / args.grad_accum
        pbar.set_description(f"Iter {it} | CE {last_train_loss:.4f} | LR {lr:.2e}")

        if it % args.log_every == 0:
            elapsed = time.time() - t0
            rate = (it + 1) / max(elapsed, 1e-6)
            eta_s = (args.max_iters - it - 1) / max(rate, 1e-6)
            eta_m, eta_sec = divmod(int(eta_s), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            print(
                f"step {it}: train loss {last_train_loss:.4f}, lr {lr:.2e}, "
                f"{rate:.1f} it/s, remaining {eta_h:d}:{eta_m:02d}:{eta_sec:02d}",
                flush=True,
            )

        it += 1
        pbar.update(1)
        if it % args.save_every == 0:
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "config": model.config_dict(),
                    "iter": it,
                    "best_val": best_val,
                    "optimizer": opt.state_dict(),
                },
                args.out,
            )
            print(f"saved {args.out} iter {it}", flush=True)

    pbar.close()
    save_checkpoint(
        {
            "model": model.state_dict(),
            "config": model.config_dict(),
            "iter": it,
            "best_val": best_val,
        },
        args.out,
    )
    print(f"Finished → {args.out} best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
