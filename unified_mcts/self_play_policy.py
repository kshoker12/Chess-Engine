#!/usr/bin/env python3
"""
Collect policy fine-tune data: model vs Stockfish, then take the winner's move
sequence (usually SF), mirrored to White STM convention.

Example:
  python self_play_policy.py \\
    --ckpt unified_ft_value.pt \\
    --vocab ../policy_transformer/vocab1.pkl \\
    --stockfish /opt/homebrew/bin/stockfish \\
    --out ./data_export/policy_ft \\
    --games 200 --max-plies 160 --sf-depth 12
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import List, Optional, Tuple

import chess
import chess.engine
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_utils import (
    RIGHTS_DIM,
    encode_board_stm_white,
    get_uci4_token,
    mirror_uci_string,
    pad_history,
)
from dataset import write_memmap
from inference import UnifiedInference


def sf_move(board: chess.Board, engine: chess.engine.SimpleEngine, depth: int, elo: Optional[int]) -> chess.Move:
    limit = chess.engine.Limit(depth=depth)
    if elo is not None:
        # Soft strength limit when supported
        try:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": int(elo)})
        except Exception:
            pass
    result = engine.play(board, limit)
    return result.move


def play_game(
    inf: UnifiedInference,
    engine: chess.engine.SimpleEngine,
    model_is_white: bool,
    max_plies: int,
    sf_depth: int,
    sf_elo: Optional[int],
    top_k: int,
    temperature: float,
) -> Tuple[List[chess.Move], Optional[bool]]:
    """
    Returns (moves, winner_is_white) where winner_is_white is True/False/None(draw).
    """
    board = chess.Board()
    moves: List[chess.Move] = []
    for _ in range(max_plies):
        if board.is_game_over():
            break
        white_to_move = board.turn == chess.WHITE
        model_turn = (white_to_move and model_is_white) or ((not white_to_move) and (not model_is_white))
        if model_turn:
            # Blend policy with a light value check on top candidates for midgame quality
            cands = inf.predict_policy_board(board, moves, top_k=top_k)
            if not cands:
                break
            if temperature <= 0:
                mv = chess.Move.from_uci(cands[0][1])
            else:
                mv = inf.choose_move(board, moves, top_k=top_k, temperature=temperature)
            if mv is None:
                break
        else:
            try:
                mv = sf_move(board, engine, sf_depth, sf_elo)
            except Exception:
                break
            if mv is None:
                break
        board.push(mv)
        moves.append(mv)

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return moves, None
    return moves, outcome.winner == chess.WHITE


def winner_rows(
    moves: List[chess.Move],
    winner_is_white: bool,
    stoi: dict,
    pad_id: int,
    block_size: int,
    include_draws: bool,
    was_draw: bool,
) -> List[dict]:
    """Emit single-target policy rows from the winner's moves, STM-white convention."""
    if was_draw and not include_draws:
        return []
    # Draws (if included): use White's plies only
    use_white_moves = True if was_draw else winner_is_white

    board = chess.Board()
    rows = []
    for ply, mv in enumerate(moves):
        is_white_ply = ply % 2 == 0
        keep = is_white_ply if use_white_moves else (not is_white_ply)
        if keep:
            pieces, rights, mirrored = encode_board_stm_white(board)
            # Remirror all prior moves with *current* STM (matches inference / preprocess_policy)
            hist_ids = []
            for prev in moves[:ply]:
                uci_p = prev.uci()
                tok_p = get_uci4_token(mirror_uci_string(uci_p) if mirrored else uci_p)
                hist_ids.append(stoi.get(tok_p, pad_id))
            uci = mv.uci()
            token = get_uci4_token(mirror_uci_string(uci) if mirrored else uci)
            if token in stoi:
                hist = pad_history(hist_ids, block_size, pad_id)
                rows.append(
                    {
                        "hist": hist,
                        "board": pieces,
                        "rights": rights,
                        "y": np.uint16(stoi[token]),
                    }
                )
        board.push(mv)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--sf-depth", type=int, default=12)
    ap.add_argument("--sf-elo", type=int, default=None, help="Optional UCI_Elo strength limit")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--include-draws", action="store_true")
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.vocab, "rb") as f:
        stoi = pickle.load(f)
    pad_id = int(stoi["|"])

    os.makedirs(args.out, exist_ok=True)
    inf = UnifiedInference(args.ckpt, args.vocab, device=args.device)
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine.configure({"Threads": 1})

    all_hist, all_board, all_R, all_y = [], [], [], []
    wins_sf = wins_model = draws = 0
    try:
        for g in tqdm(range(args.games), desc="policy_sp"):
            model_is_white = g % 2 == 0
            moves, winner_white = play_game(
                inf,
                engine,
                model_is_white,
                args.max_plies,
                args.sf_depth,
                args.sf_elo,
                args.top_k,
                args.temperature,
            )
            was_draw = winner_white is None
            if was_draw:
                draws += 1
                if not args.include_draws:
                    continue
            else:
                model_won = (winner_white and model_is_white) or ((not winner_white) and (not model_is_white))
                if model_won:
                    wins_model += 1
                else:
                    wins_sf += 1

            rows = winner_rows(
                moves,
                bool(winner_white) if winner_white is not None else True,
                stoi,
                pad_id,
                args.block_size,
                args.include_draws,
                was_draw,
            )
            for row in rows:
                all_hist.append(row["hist"])
                all_board.append(row["board"])
                all_R.append(row["rights"])
                all_y.append(row["y"])
    finally:
        engine.quit()

    n = len(all_y)
    if n == 0:
        raise SystemExit("No policy rows collected (all draws?). Try --include-draws or more games.")

    hist = np.stack(all_hist).astype(np.uint16)
    board = np.stack(all_board).astype(np.uint8)
    rights = np.stack(all_R).astype(np.uint8)
    y = np.asarray(all_y, dtype=np.uint16)
    yv = np.zeros(n, dtype=np.float16)

    write_memmap(os.path.join(args.out, "policy_hist.u16.memmap"), hist)
    write_memmap(os.path.join(args.out, "policy_board.u8.memmap"), board)
    write_memmap(os.path.join(args.out, "policy_R.u8.memmap"), rights)
    write_memmap(os.path.join(args.out, "policy_y.u16.memmap"), y)
    write_memmap(os.path.join(args.out, "policy_yv.f16.memmap"), yv)

    meta = {
        "N": n,
        "K": RIGHTS_DIM,
        "block_size": args.block_size,
        "vocab": os.path.abspath(args.vocab),
        "pad_id": pad_id,
        "vocab_size": len(stoi),
        "sf_depth": args.sf_depth,
        "sf_elo": args.sf_elo,
        "games": args.games,
        "wins_sf": wins_sf,
        "wins_model": wins_model,
        "draws": draws,
        "sf_labeled": False,
        "has_policy_yv": True,
        "sources": [{"name": "self_play", "start": 0, "end": n}],
        "source": "self_play_policy",
        "ckpt": os.path.abspath(args.ckpt),
    }
    with open(os.path.join(args.out, "policy_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done → {args.out} N={n} (SF wins={wins_sf} model={wins_model} draws={draws})")


if __name__ == "__main__":
    main()
