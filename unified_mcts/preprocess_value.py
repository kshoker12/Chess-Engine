#!/usr/bin/env python3
"""
Phase A preprocess: convert Lichess eval npy (or similar) to value memmaps.

Example:
  python preprocess_value.py \
    --x ../data/dataset_X_mega.npy \
    --y ../data/dataset_Y_mega.npy \
    --out ./data_export/value
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_utils import RIGHTS_DIM, tanh_cp
from dataset import write_memmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True, help="Path to board tokens .npy [N,64]")
    ap.add_argument("--y", required=True, help="Path to labels .npy [N] (tanh cp or raw cp)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--y-is-tanh", action="store_true", help="Y already in (-1,1) tanh space")
    ap.add_argument("--rights", default=None, help="Optional rights .npy [N,8]; else zeros")
    ap.add_argument("--max-n", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading X from {args.x} ...")
    X = np.load(args.x, mmap_mode="r")
    print(f"Loading Y from {args.y} ...")
    Y = np.load(args.y, mmap_mode="r")
    assert len(X) == len(Y), f"len X={len(X)} != len Y={len(Y)}"
    n = len(X) if args.max_n is None else min(len(X), args.max_n)
    assert X.shape[1] == 64, X.shape

    if args.rights:
        R = np.load(args.rights, mmap_mode="r")
        assert R.shape[0] >= n and R.shape[1] == RIGHTS_DIM
        R_out = np.asarray(R[:n], dtype=np.uint8)
    else:
        print("No rights file — writing zeros (legacy evals).")
        R_out = np.zeros((n, RIGHTS_DIM), dtype=np.uint8)

    print(f"Writing N={n} ...")
    X_out = np.asarray(X[:n], dtype=np.uint8)

    y_raw = np.asarray(Y[:n], dtype=np.float64)
    if args.y_is_tanh:
        y_out = y_raw.astype(np.float16)
    else:
        # assume centipawns (or already roughly cp-scale)
        y_out = np.array([tanh_cp(float(v)) for v in y_raw], dtype=np.float16)

    # drop nan/inf
    mask = np.isfinite(y_out.astype(np.float32))
    if not mask.all():
        print(f"Dropping {(~mask).sum()} non-finite labels")
        X_out, R_out, y_out = X_out[mask], R_out[mask], y_out[mask]
        n = int(mask.sum())

    write_memmap(os.path.join(args.out, "value_X.u8.memmap"), X_out)
    write_memmap(os.path.join(args.out, "value_R.u8.memmap"), R_out)
    write_memmap(os.path.join(args.out, "value_Y.f16.memmap"), y_out)

    meta = {
        "N": n,
        "K": RIGHTS_DIM,
        "piece_map": "unified_mcts.board_utils.PIECE_MAP",
        "y_space": "tanh(cp/400)",
        "source_x": os.path.abspath(args.x),
        "source_y": os.path.abspath(args.y),
    }
    with open(os.path.join(args.out, "value_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done → {args.out} (N={n})")


if __name__ == "__main__":
    main()
