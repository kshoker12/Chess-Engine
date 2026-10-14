#!/usr/bin/env python3
"""
Build unified policy training indices from existing full/elite/weak *.pt streams.

Those tensors are 1D UCI token streams ('|' separates games; Y<0 means ignore).
Boards are NOT written (~150GB). We copy streams to memmap + compact start/end
indices; PtReplayPolicyDataset replays boards on the fly.

Example:
  python build_policy_from_pt.py \\
    --data-dir ../data \\
    --vocab ../policy_transformer/vocab1.pkl \\
    --out ./data_export/policy_pt
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import write_memmap


SOURCES = (
    ("full", "full_X.pt", "full_Y.pt"),
    ("elite", "elite_X.pt", "elite_Y.pt"),
    ("weak", "weak_X.pt", "weak_Y.pt"),
)


def build_source(
    name: str,
    x_path: str,
    y_path: str,
    out_dir: str,
    pad_id: int,
    itos: dict,
    copy_streams: bool,
) -> dict:
    print(f"Loading {name}: {x_path}", flush=True)
    x = torch.load(x_path, map_location="cpu", weights_only=True)
    y = torch.load(y_path, map_location="cpu", weights_only=True)
    x_np = np.asarray(x, dtype=np.int16)
    y_np = np.asarray(y, dtype=np.int16)
    del x, y
    n = int(x_np.shape[0])
    assert y_np.shape[0] == n

    if copy_streams:
        write_memmap(os.path.join(out_dir, f"{name}_X.i16.memmap"), x_np)
        write_memmap(os.path.join(out_dir, f"{name}_Y.i16.memmap"), y_np)
        x_rel, y_rel = f"{name}_X.i16.memmap", f"{name}_Y.i16.memmap"
        x_src, y_src = None, None
    else:
        x_rel = y_rel = None
        x_src, y_src = os.path.abspath(x_path), os.path.abspath(y_path)

    end_path = os.path.join(out_dir, f"{name}_end.u32.memmap")
    start_path = os.path.join(out_dir, f"{name}_start.u32.memmap")
    n_rows = 0
    games = 0
    game_start = 0

    with open(end_path, "wb") as fe, open(start_path, "wb") as fs:
        pbar = tqdm(total=n, desc=f"index:{name}", unit="tok")
        i = 0
        while i < n:
            if x_np[i] == pad_id:
                game_start = i + 1
                i += 1
                pbar.update(1)
                continue

            games += 1
            while i < n and x_np[i] != pad_id:
                yj = int(y_np[i])
                if yj >= 0 and yj != pad_id and yj in itos and len(itos[yj]) >= 4:
                    fe.write(np.asarray([i], dtype=np.uint32).tobytes())
                    fs.write(np.asarray([game_start], dtype=np.uint32).tobytes())
                    n_rows += 1
                i += 1
                pbar.update(1)
            # i at pad or EOF; loop handles pad
        pbar.close()

    meta = {
        "name": name,
        "stream_len": n,
        "n_rows": n_rows,
        "n_games": games,
        "x_memmap": x_rel,
        "y_memmap": y_rel,
        "x_pt": x_src,
        "y_pt": y_src,
        "end_memmap": f"{name}_end.u32.memmap",
        "start_memmap": f"{name}_start.u32.memmap",
    }
    print(f"  {name}: rows={n_rows:,} games={games:,}", flush=True)
    del x_np, y_np
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data", help="Dir with full_X.pt etc.")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", default="./data_export/policy_pt")
    ap.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Subset of full elite weak (default: all present)",
    )
    ap.add_argument(
        "--no-copy-streams",
        action="store_true",
        help="Do not copy X/Y to memmap (dataset will torch.load .pt paths).",
    )
    args = ap.parse_args()

    with open(args.vocab, "rb") as f:
        stoi = pickle.load(f)
    itos = {int(i): s for s, i in stoi.items()}
    pad_id = int(stoi["|"])
    os.makedirs(args.out, exist_ok=True)

    want = set(args.sources) if args.sources else {n for n, _, _ in SOURCES}
    sources_meta = []
    for name, xf, yf in SOURCES:
        if name not in want:
            continue
        xp = os.path.join(args.data_dir, xf)
        yp = os.path.join(args.data_dir, yf)
        if not (os.path.isfile(xp) and os.path.isfile(yp)):
            print(f"Skip {name}: missing {xp} or {yp}")
            continue
        sources_meta.append(
            build_source(
                name,
                xp,
                yp,
                args.out,
                pad_id,
                itos,
                copy_streams=not args.no_copy_streams,
            )
        )

    if not sources_meta:
        raise SystemExit("No sources built")

    meta = {
        "format": "pt_replay",
        "K": 8,
        "block_size": 128,
        "vocab": os.path.abspath(args.vocab),
        "pad_id": pad_id,
        "vocab_size": len(stoi),
        "convention": "winner_as_white_stm_from_pt_streams",
        "mix_default": {"full": 0.80, "elite": 0.10, "weak": 0.10},
        "sources": sources_meta,
        "N": int(sum(s["n_rows"] for s in sources_meta)),
    }
    with open(os.path.join(args.out, "policy_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done → {args.out} total_rows={meta['N']:,}", flush=True)


if __name__ == "__main__":
    main()
