#!/usr/bin/env python3
"""
Fuse Kaggle results.zip dumps into one value_ft/ + one policy_ft/.

Drop zips in unified_mcts/shards/, then:

  python merge_ft_shards.py

Writes data_export/sp_cycle1/ and deletes the zip files.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import write_memmap

VALUE_FILES = ("value_meta.json", "value_X.u8.memmap", "value_R.u8.memmap", "value_Y.f16.memmap")
POLICY_FILES = (
    "policy_meta.json",
    "policy_hist.u16.memmap",
    "policy_board.u8.memmap",
    "policy_R.u8.memmap",
    "policy_y.u16.memmap",
)


def _open(path, dtype, shape):
    return np.memmap(path, dtype=dtype, mode="r", shape=tuple(shape))


def _is_value_ft(d: str) -> bool:
    return all(os.path.isfile(os.path.join(d, f)) for f in VALUE_FILES)


def _is_policy_ft(d: str) -> bool:
    return all(os.path.isfile(os.path.join(d, f)) for f in POLICY_FILES)


def _walk_kind(root: str, pred) -> list[str]:
    found = []
    for dirpath, dirnames, _files in os.walk(root):
        dirnames[:] = [x for x in dirnames if x not in (".git", "__pycache__")]
        if pred(dirpath):
            found.append(os.path.abspath(dirpath))
            dirnames[:] = []
    return found


def _safe_extract(zp: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zp, "r") as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if name.startswith("/") or ".." in name.split("/"):
                continue
            zf.extract(info, dest)


def _unique_extract_dir(extract_root: str, zip_path: str, used: set[str]) -> str:
    stem = os.path.splitext(os.path.basename(zip_path))[0]
    stem = stem.replace(" ", "_") or "shard"
    cand = os.path.join(extract_root, stem)
    i = 2
    while cand in used or (os.path.isdir(cand) and cand not in used):
        # allow reuse if already extracted from this zip name; if collision, suffix
        if cand not in used and os.path.isdir(cand):
            break
        cand = os.path.join(extract_root, f"{stem}_{i}")
        i += 1
    used.add(cand)
    return cand


def collect_ft_dirs(shard_args: list[str]) -> tuple[list[str], list[str], list[str], str | None]:
    """Return (value_ft dirs, policy_ft dirs, zip paths, temp extract dir)."""
    import tempfile

    paths = []
    for s in shard_args:
        hits = sorted(glob.glob(s)) if any(c in s for c in "*?[]") else [s]
        paths.extend(os.path.abspath(h) for h in hits if os.path.exists(h))

    zip_files = []
    scan_roots = []
    for p in paths:
        if os.path.isfile(p) and p.lower().endswith(".zip"):
            zip_files.append(p)
        elif os.path.isdir(p):
            scan_roots.append(p)
            for fn in sorted(os.listdir(p)):
                fp = os.path.join(p, fn)
                if os.path.isfile(fp) and fn.lower().endswith(".zip"):
                    zip_files.append(fp)

    seen_z = set()
    zips = []
    for z in zip_files:
        if z not in seen_z:
            seen_z.add(z)
            zips.append(z)

    tmp = None
    if zips:
        tmp = tempfile.mkdtemp(prefix="sp_merge_")
        used = set()
        for zp in zips:
            dest = _unique_extract_dir(tmp, zp, used)
            print(f"unzip {os.path.basename(zp)}")
            _safe_extract(zp, dest)
            scan_roots.append(dest)

    value_dirs, policy_dirs = [], []
    seen_v, seen_p = set(), set()
    for root in scan_roots:
        if os.path.basename(root) == "_extracted":
            continue
        for d in _walk_kind(root, _is_value_ft):
            if d not in seen_v:
                seen_v.add(d)
                value_dirs.append(d)
        for d in _walk_kind(root, _is_policy_ft):
            if d not in seen_p:
                seen_p.add(d)
                policy_dirs.append(d)

    return value_dirs, policy_dirs, zips, tmp


def merge_value(value_dirs: list[str], out_dir: str) -> int:
    metas, Xs, Rs, Ys = [], [], [], []
    for vdir in value_dirs:
        with open(os.path.join(vdir, "value_meta.json")) as f:
            meta = json.load(f)
        n = int(meta["N"])
        k = int(meta.get("K", 8))
        Xs.append(_open(os.path.join(vdir, "value_X.u8.memmap"), np.uint8, (n, 64)))
        Rs.append(_open(os.path.join(vdir, "value_R.u8.memmap"), np.uint8, (n, k)))
        Ys.append(_open(os.path.join(vdir, "value_Y.f16.memmap"), np.float16, (n,)))
        metas.append((vdir, meta, n))
        print(f"  value +{n} from {vdir}")
    if not metas:
        raise SystemExit("No value_ft shards found")
    n = sum(ni for _, _, ni in metas)
    k = int(metas[0][1].get("K", 8))
    os.makedirs(out_dir, exist_ok=True)
    X = np.concatenate([np.asarray(x) for x in Xs]).astype(np.uint8)
    R = np.concatenate([np.asarray(r) for r in Rs]).astype(np.uint8)
    Y = np.concatenate([np.asarray(y) for y in Ys]).astype(np.float16)
    write_memmap(os.path.join(out_dir, "value_X.u8.memmap"), X)
    write_memmap(os.path.join(out_dir, "value_R.u8.memmap"), R)
    write_memmap(os.path.join(out_dir, "value_Y.f16.memmap"), Y)
    sources, start = [], 0
    wins_sf = wins_model = draws = 0
    for d, meta, ni in metas:
        sources.append(
            {
                "path": d,
                "start": start,
                "end": start + ni,
                "N": ni,
                **{k: meta.get(k) for k in ("shard_id", "ckpt", "sf_depth")},
            }
        )
        start += ni
        wins_sf += int(meta.get("wins_sf") or 0)
        wins_model += int(meta.get("wins_model") or 0)
        draws += int(meta.get("draws") or 0)
    fused = {
        "N": n,
        "K": k,
        "y_space": metas[0][1].get("y_space", "tanh(cp/400)"),
        "source": "merge_ft_shards",
        "n_shards": len(metas),
        "wins_sf": wins_sf,
        "wins_model": wins_model,
        "draws": draws,
        "sources": sources,
    }
    with open(os.path.join(out_dir, "value_meta.json"), "w") as f:
        json.dump(fused, f, indent=2)
    print(f"fused value N={n} from {len(metas)} shards → {out_dir}")
    return n


def merge_policy(policy_dirs: list[str], out_dir: str) -> int:
    metas, hists, boards, rights, ys, yvs = [], [], [], [], [], []
    for pdir in policy_dirs:
        with open(os.path.join(pdir, "policy_meta.json")) as f:
            meta = json.load(f)
        n = int(meta["N"])
        bs = int(meta.get("block_size", 128))
        k = int(meta.get("K", 8))
        hists.append(_open(os.path.join(pdir, "policy_hist.u16.memmap"), np.uint16, (n, bs)))
        boards.append(_open(os.path.join(pdir, "policy_board.u8.memmap"), np.uint8, (n, 64)))
        rights.append(_open(os.path.join(pdir, "policy_R.u8.memmap"), np.uint8, (n, k)))
        ys.append(_open(os.path.join(pdir, "policy_y.u16.memmap"), np.uint16, (n,)))
        yv_path = os.path.join(pdir, "policy_yv.f16.memmap")
        if os.path.isfile(yv_path):
            yvs.append(_open(yv_path, np.float16, (n,)))
        else:
            yvs.append(np.zeros(n, dtype=np.float16))
        metas.append((pdir, meta, n))
        print(f"  policy +{n} from {pdir}")
    if not metas:
        raise SystemExit("No policy_ft shards found")
    n = sum(ni for _, _, ni in metas)
    os.makedirs(out_dir, exist_ok=True)
    hist = np.concatenate([np.asarray(x) for x in hists]).astype(np.uint16)
    board = np.concatenate([np.asarray(x) for x in boards]).astype(np.uint8)
    rights_a = np.concatenate([np.asarray(x) for x in rights]).astype(np.uint8)
    y = np.concatenate([np.asarray(x) for x in ys]).astype(np.uint16)
    yv = np.concatenate([np.asarray(x) for x in yvs]).astype(np.float16)
    write_memmap(os.path.join(out_dir, "policy_hist.u16.memmap"), hist)
    write_memmap(os.path.join(out_dir, "policy_board.u8.memmap"), board)
    write_memmap(os.path.join(out_dir, "policy_R.u8.memmap"), rights_a)
    write_memmap(os.path.join(out_dir, "policy_y.u16.memmap"), y)
    write_memmap(os.path.join(out_dir, "policy_yv.f16.memmap"), yv)
    sources, start = [], 0
    for d, meta, ni in metas:
        sources.append(
            {
                "path": d,
                "start": start,
                "end": start + ni,
                "N": ni,
                **{k: meta.get(k) for k in ("shard_id", "ckpt", "vocab", "pad_id")},
            }
        )
        start += ni
    fused = {
        "N": n,
        "K": int(metas[0][1].get("K", 8)),
        "block_size": int(metas[0][1].get("block_size", 128)),
        "vocab": metas[0][1].get("vocab"),
        "pad_id": metas[0][1].get("pad_id"),
        "has_policy_yv": True,
        "sf_labeled": True,
        "source": "merge_ft_shards",
        "n_shards": len(metas),
        "sources": sources,
    }
    with open(os.path.join(out_dir, "policy_meta.json"), "w") as f:
        json.dump(fused, f, indent=2)
    print(f"fused policy N={n} from {len(metas)} shards → {out_dir}")
    return n


def main():
    import shutil

    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shards",
        nargs="+",
        default=[os.path.join(here, "shards")],
        help="Folder of results.zip files",
    )
    ap.add_argument("--out", default=os.path.join(here, "data_export", "sp_cycle1"))
    args = ap.parse_args()
    value_dirs, policy_dirs, zips, tmp = collect_ft_dirs(args.shards)
    print(f"found {len(value_dirs)} value_ft, {len(policy_dirs)} policy_ft")
    if not value_dirs and not policy_dirs:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(f"Nothing found under {args.shards}. Drop Kaggle results.zip files there.")
    try:
        merge_value(value_dirs, os.path.join(args.out, "value_ft"))
        merge_policy(policy_dirs, os.path.join(args.out, "policy_ft"))
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        leftover = os.path.join(here, "shards", "_extracted")
        if os.path.isdir(leftover):
            shutil.rmtree(leftover, ignore_errors=True)
    for zp in zips:
        try:
            os.remove(zp)
            print("deleted", os.path.basename(zp))
        except OSError as e:
            print("could not delete", zp, e)
    print("done →", args.out)


if __name__ == "__main__":
    main()
