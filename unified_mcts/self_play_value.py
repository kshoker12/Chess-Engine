#!/usr/bin/env python3
"""
Collect value fine-tune data via policy-guided positions + Stockfish labels.

Emphasizes disagreement |v_θ − y_SF| while keeping an exploration floor.
Writes value memmaps under --out (compatible with ValueMemmapDataset / finetune_value).

Example:
  python self_play_value.py \\
    --ckpt unified_phase_b.pt \\
    --vocab ../policy_transformer/vocab1.pkl \\
    --stockfish /opt/homebrew/bin/stockfish \\
    --out ./data_export/value_ft \\
    --games 500 --max-plies 120 --sf-depth 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import chess
import chess.engine
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_utils import RIGHTS_DIM, encode_board_stm_white, stm_value_from_white_pov, tanh_cp
from dataset import write_memmap
from inference import UnifiedInference


def eval_sf_white_pov(board: chess.Board, engine: chess.engine.SimpleEngine, depth: int) -> float:
    """Absolute White-POV tanh(cp/400) on the given (unmirrored) board."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white()
    if score.is_mate():
        m = score.mate()
        cp = 10000.0 if m and m > 0 else -10000.0
    else:
        cp = float(score.score(mate_score=10000))
    return tanh_cp(cp)


def eval_sf_stm(board: chess.Board, engine: chess.engine.SimpleEngine, depth: int, mirrored: bool) -> float:
    """STM-white label matching encode_board_stm_white + Phase A mega data."""
    return stm_value_from_white_pov(eval_sf_white_pov(board, engine, depth), mirrored)


def sample_policy_move(
    inf: UnifiedInference,
    board: chess.Board,
    moves: List[chess.Move],
    top_k: int,
    temperature: float,
    explore: bool,
) -> chess.Move:
    if explore:
        return inf.choose_move(
            board,
            moves,
            top_k=top_k,
            temperature=max(temperature, 0.5),
            dirichlet_alpha=0.25,
            dirichlet_eps=0.25,
        ) or list(board.legal_moves)[0]
    return inf.choose_move(board, moves, top_k=top_k, temperature=0.0) or list(board.legal_moves)[0]


def collect_game(
    inf: UnifiedInference,
    engine: chess.engine.SimpleEngine,
    sf_depth: int,
    max_plies: int,
    top_k: int,
    temperature: float,
    explore_frac: float,
    cand_k: int,
) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
    """
    Returns rows: (pieces, rights, y_sf, |v_theta - y_sf|).
    Labels current + a few policy-candidate leaf boards each ply.
    """
    board = chess.Board()
    moves: List[chess.Move] = []
    rows: List[Tuple[np.ndarray, np.ndarray, float, float]] = []

    for ply in range(max_plies):
        if board.is_game_over():
            break
        explore = np.random.rand() < explore_frac

        # Label current position (STM-white tokens + STM label)
        pieces, rights, mirrored = encode_board_stm_white(board)
        v_theta = inf.evaluate_board(board)
        try:
            y_sf = eval_sf_stm(board, engine, sf_depth, mirrored)
        except Exception:
            y_sf = 0.0
        rows.append((pieces, rights, float(y_sf), abs(v_theta - y_sf)))

        # Optional: label top candidate leaves (1-ply)
        cands = inf.predict_policy_board(board, moves, top_k=min(cand_k, top_k))
        for _, uci in cands[:cand_k]:
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                continue
            board.push(mv)
            pieces_c, rights_c, mir_c = encode_board_stm_white(board)
            v_c = inf.evaluate_board(board)
            try:
                y_c = eval_sf_stm(board, engine, sf_depth, mir_c)
            except Exception:
                y_c = 0.0
            rows.append((pieces_c, rights_c, float(y_c), abs(v_c - y_c)))
            board.pop()

        mv = sample_policy_move(inf, board, moves, top_k, temperature, explore)
        board.push(mv)
        moves.append(mv)

    return rows


def mine_keep(
    candidates: List[Tuple[np.ndarray, np.ndarray, float, float]],
    keep_n: int,
    disagree_frac: float,
    floor_frac: float,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Prefer high disagreement; keep a random floor."""
    if not candidates:
        return []
    keep_n = min(keep_n, len(candidates))
    n_dis = int(keep_n * disagree_frac)
    n_floor = max(0, min(int(keep_n * floor_frac), keep_n - n_dis))
    n_dis = keep_n - n_floor

    order = np.argsort([-d for *_, d in candidates])
    picked = set()
    out: List[Tuple[np.ndarray, np.ndarray, float]] = []

    for idx in order:
        if len(out) >= n_dis:
            break
        i = int(idx)
        if i in picked:
            continue
        pieces, rights, y, _ = candidates[i]
        out.append((pieces, rights, y))
        picked.add(i)

    remaining = [i for i in range(len(candidates)) if i not in picked]
    if remaining and n_floor > 0:
        choice = np.random.choice(remaining, size=min(n_floor, len(remaining)), replace=False)
        for i in choice:
            pieces, rights, y, _ = candidates[int(i)]
            out.append((pieces, rights, y))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--sf-depth", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--cand-k", type=int, default=2, help="Extra 1-ply leaves to label per ply")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--explore-frac", type=float, default=0.3)
    ap.add_argument("--keep-per-game", type=int, default=40)
    ap.add_argument("--disagree-frac", type=float, default=0.7)
    ap.add_argument("--floor-frac", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    inf = UnifiedInference(args.ckpt, args.vocab, device=args.device)
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine.configure({"Threads": 1})

    all_X, all_R, all_Y = [], [], []
    try:
        for g in tqdm(range(args.games), desc="value_sp"):
            rows = collect_game(
                inf,
                engine,
                args.sf_depth,
                args.max_plies,
                args.top_k,
                args.temperature,
                args.explore_frac,
                args.cand_k,
            )
            kept = mine_keep(rows, args.keep_per_game, args.disagree_frac, args.floor_frac)
            for pieces, rights, y in kept:
                all_X.append(pieces)
                all_R.append(rights)
                all_Y.append(np.float16(y))
    finally:
        engine.quit()

    n = len(all_Y)
    if n == 0:
        raise SystemExit("No value rows collected")

    X = np.stack(all_X).astype(np.uint8)
    R = np.stack(all_R).astype(np.uint8)
    Y = np.asarray(all_Y, dtype=np.float16)
    write_memmap(os.path.join(args.out, "value_X.u8.memmap"), X)
    write_memmap(os.path.join(args.out, "value_R.u8.memmap"), R)
    write_memmap(os.path.join(args.out, "value_Y.f16.memmap"), Y)
    meta = {
        "N": n,
        "K": RIGHTS_DIM,
        "y_space": "tanh(cp/400)",
        "sf_depth": args.sf_depth,
        "games": args.games,
        "source": "self_play_value",
        "ckpt": os.path.abspath(args.ckpt),
    }
    with open(os.path.join(args.out, "value_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done → {args.out} N={n}")


if __name__ == "__main__":
    main()
