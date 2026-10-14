#!/usr/bin/env python3
"""
Build Phase A value memmaps from scratch from Lichess Stockfish evaluations.

Source (deduplicated Lichess eval DB, max-depth per FEN):
  https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized
  (derived from official Lichess / HF chess-position-evaluations)

Convention (matches value_transformer / chess-ai-model §4.2):
  - Board tokens + rights via encode_board_stm_white (side-to-move always White)
  - Labels are STM advantage: tanh(white_pov_cp/400), negated when Black to move
  - Mate → ±10000 cp before tanh

Example (~50M rows, depth>=20):
  python build_value_from_lichess.py \\
    --out ./data_export/value \\
    --max-n 50000000 \\
    --min-depth 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import chess
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_utils import RIGHTS_DIM, encode_board_stm_white, stm_value_from_white_pov, tanh_cp
from dataset import write_memmap

HF_DATASET = "mateuszgrzyb/lichess-stockfish-normalized"
MATE_CP = 10000.0


def normalize_fen(fen: str) -> str:
    """HF FENs often omit halfmove/fullmove clocks."""
    fen = fen.strip()
    if fen.count(" ") == 3:
        fen = fen + " 0 1"
    return fen


def white_pov_cp(cp, mate) -> float | None:
    if mate is not None:
        try:
            m = int(mate)
        except (TypeError, ValueError):
            return None
        if m == 0:
            return None
        # Lichess: positive mate ⇒ White mates
        return MATE_CP if m > 0 else -MATE_CP
    if cp is None:
        return None
    try:
        return float(cp)
    except (TypeError, ValueError):
        return None


def row_to_sample(row, min_depth: int):
    depth = row.get("depth")
    try:
        if depth is None or int(depth) < min_depth:
            return None
    except (TypeError, ValueError):
        return None

    wcp = white_pov_cp(row.get("cp"), row.get("mate"))
    if wcp is None:
        return None

    fen = row.get("fen")
    if not fen:
        return None
    try:
        board = chess.Board(normalize_fen(fen))
    except ValueError:
        return None

    pieces, rights, mirrored = encode_board_stm_white(board)
    y = stm_value_from_white_pov(tanh_cp(wcp), mirrored)
    if not np.isfinite(y):
        return None
    return pieces, rights, np.float16(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-n", type=int, default=50_000_000)
    ap.add_argument("--min-depth", type=int, default=20)
    ap.add_argument("--shard-size", type=int, default=1_000_000)
    ap.add_argument(
        "--dataset",
        default=HF_DATASET,
        help="HF dataset id (deduped Lichess evals by default)",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    shard_dir = os.path.join(args.out, "_shards")
    os.makedirs(shard_dir, exist_ok=True)

    print(f"Streaming {args.dataset} …")
    print(f"max_n={args.max_n:,} min_depth={args.min_depth}")
    ds = load_dataset(args.dataset, split="train", streaming=True)

    xs, rs, ys = [], [], []
    n = 0
    shard_i = 0
    seen = 0
    pbar = tqdm(total=args.max_n, desc="value_rows", unit="pos")

    def flush():
        nonlocal xs, rs, ys, shard_i
        if not ys:
            return
        X = np.stack(xs).astype(np.uint8)
        R = np.stack(rs).astype(np.uint8)
        Y = np.asarray(ys, dtype=np.float16)
        np.save(os.path.join(shard_dir, f"X_{shard_i:04d}.npy"), X)
        np.save(os.path.join(shard_dir, f"R_{shard_i:04d}.npy"), R)
        np.save(os.path.join(shard_dir, f"Y_{shard_i:04d}.npy"), Y)
        print(f"\n  wrote shard {shard_i} size={len(Y):,}")
        shard_i += 1
        xs, rs, ys = [], [], []

    try:
        for row in ds:
            seen += 1
            sample = row_to_sample(row, args.min_depth)
            if sample is None:
                continue
            pieces, rights, y = sample
            xs.append(pieces)
            rs.append(rights)
            ys.append(y)
            n += 1
            pbar.update(1)
            if len(ys) >= args.shard_size:
                flush()
            if n >= args.max_n:
                break
    finally:
        pbar.close()
        flush()

    if n == 0:
        raise SystemExit("No rows written — check network / dataset id")

    print(f"Concatenating {shard_i} shards → memmaps (N={n:,}) …")
    X_out = np.empty((n, 64), dtype=np.uint8)
    R_out = np.empty((n, RIGHTS_DIM), dtype=np.uint8)
    Y_out = np.empty((n,), dtype=np.float16)
    cursor = 0
    for i in range(shard_i):
        X = np.load(os.path.join(shard_dir, f"X_{i:04d}.npy"))
        R = np.load(os.path.join(shard_dir, f"R_{i:04d}.npy"))
        Y = np.load(os.path.join(shard_dir, f"Y_{i:04d}.npy"))
        k = len(Y)
        X_out[cursor : cursor + k] = X
        R_out[cursor : cursor + k] = R
        Y_out[cursor : cursor + k] = Y
        cursor += k
    assert cursor == n

    write_memmap(os.path.join(args.out, "value_X.u8.memmap"), X_out)
    write_memmap(os.path.join(args.out, "value_R.u8.memmap"), R_out)
    write_memmap(os.path.join(args.out, "value_Y.f16.memmap"), Y_out)

    meta = {
        "N": n,
        "K": RIGHTS_DIM,
        "piece_map": "unified_mcts.board_utils.PIECE_MAP",
        "y_space": "tanh(cp/400)_stm",
        "source": args.dataset,
        "source_note": "Deduplicated Lichess Stockfish evals (max depth per FEN)",
        "min_depth": args.min_depth,
        "rows_seen_stream": seen,
        "rights": "from_fen",
        "stm_white": True,
        "cp_pov": "lichess_white; label converted to STM via negate-if-black",
    }
    with open(os.path.join(args.out, "value_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # cleanup shards
    for i in range(shard_i):
        for pref in ("X", "R", "Y"):
            try:
                os.remove(os.path.join(shard_dir, f"{pref}_{i:04d}.npy"))
            except OSError:
                pass
    try:
        os.rmdir(shard_dir)
    except OSError:
        pass

    print(f"Done → {args.out} N={n:,} (streamed ~{seen:,} source rows)")


if __name__ == "__main__":
    main()
