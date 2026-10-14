#!/usr/bin/env python3
"""
Head-to-head: production policy+value vs unified Phase B.

Live SVG board (same idea as policy_transformer/simulate_viz):
  unified_mcts/simulate_viz/board.svg

Modes
  policy  — each side samples from its own policy top-k (no value)
  value   — each side scores all legal moves with its own value net, plays argmax
  mix     — Bernoulli: with p_value, pick best value among policy top-k;
            otherwise sample from policy top-k

Colors are randomized each game. Production is never modified.
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import re
import sys
import time
from typing import List, Optional, Tuple

import chess
import chess.svg
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from policy_transformer.model import PolicyHead
from value_transformer.inference import ChessEvaluator
from unified_mcts.board_utils import get_uci4_token, mirror_uci_string
from unified_mcts.inference import UnifiedInference

VIZ_PATH = os.path.join(HERE, "simulate_viz", "board.svg")
DEFAULT_UNIFIED = os.path.join(HERE, "ckpts", "unified_phase_b_best.pt")
DEFAULT_PROD_POLICY = os.path.join(ROOT, "policy_transformer", "ultra_3o9.pt")
DEFAULT_PROD_VALUE = os.path.join(ROOT, "value_transformer", "mini_value_6o4.pt")
DEFAULT_VOCAB_PROD = os.path.join(ROOT, "policy_transformer", "vocab.pkl")
DEFAULT_VOCAB1 = os.path.join(ROOT, "policy_transformer", "vocab1.pkl")


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _svg_legend(svg_str: str, white_name: str, black_name: str, extra: str = "") -> str:
    match = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg_str)
    if not match:
        return svg_str
    w, h = int(match.group(1)), int(match.group(2))
    extra_h = 44 if extra else 28
    new_h = h + extra_h
    svg_str = svg_str.replace(f'viewBox="0 0 {w} {h}"', f'viewBox="0 0 {w} {new_h}"', 1)
    line1 = f"White: {white_name}   Black: {black_name}"
    legend = (
        f'<rect x="0" y="{h}" width="{w}" height="{extra_h}" fill="#ffffff"/>'
        f'<text x="{w / 2}" y="{h + 18}" text-anchor="middle" font-size="15" '
        f'font-family="sans-serif" fill="#1a1a1a">{line1}</text>'
    )
    if extra:
        legend += (
            f'<text x="{w / 2}" y="{h + 36}" text-anchor="middle" font-size="12" '
            f'font-family="sans-serif" fill="#444">{extra}</text>'
        )
    if svg_str.strip().endswith("</svg>"):
        svg_str = svg_str[:-6] + "\n" + legend + "\n</svg>"
    return svg_str


def write_board(board: chess.Board, white_name: str, black_name: str, extra: str = "") -> None:
    os.makedirs(os.path.dirname(VIZ_PATH), exist_ok=True)
    svg = chess.svg.board(board=board, size=400)
    svg = _svg_legend(svg, white_name, black_name, extra)
    with open(VIZ_PATH, "w") as f:
        f.write(svg)


def _promo_queen(board: chess.Board, uci: str) -> str:
    if len(uci) < 4:
        return uci
    try:
        from_sq = chess.parse_square(uci[0:2])
        to_sq = chess.parse_square(uci[2:4])
    except (ValueError, IndexError):
        return uci
    piece = board.piece_at(from_sq)
    if piece is None or piece.piece_type != chess.PAWN:
        return uci
    if chess.square_rank(to_sq) in (0, 7) and len(uci) == 4:
        return uci + "q"
    return uci


class ProdPolicy:
    def __init__(self, ckpt: str, vocab_path: str, device: str):
        with open(vocab_path, "rb") as f:
            self.stoi = pickle.load(f)
        self.device = device
        self.model = PolicyHead(
            vocab_size=len(self.stoi),
            n_embd=500,
            block_size=128,
            n_head=10,
            n_layer=8,
            dropout=0.0,
            device=device,
        )
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        if isinstance(state, dict) and any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()

    def top_k(self, board: chess.Board, moves_uci: List[str], k: int) -> List[Tuple[float, str]]:
        legal = list(board.legal_moves)
        if not legal:
            return []
        should_mirror = board.turn == chess.BLACK
        ids = []
        for uci in moves_uci:
            enc = mirror_uci_string(uci) if should_mirror else uci
            tok = get_uci4_token(enc)
            ids.append(self.stoi.get(tok, self.stoi.get("|", 0)))
        if not ids:
            ids = [0]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        if idx.shape[1] > self.model.block_size:
            idx = idx[:, -self.model.block_size :]
        with torch.no_grad():
            logits, _ = self.model(idx)
            last = logits[0, -1, :]
        cands = []
        for mv in legal:
            uci = mv.uci()
            search = mirror_uci_string(uci) if should_mirror else uci
            tok = get_uci4_token(search)
            tid = self.stoi.get(tok)
            score = float(last[tid].item()) if tid is not None else float("-inf")
            cands.append((score, uci))
        cands.sort(key=lambda x: x[0], reverse=True)
        return cands[:k]


def _sample_topk(cands: List[Tuple[float, str]]) -> Optional[str]:
    if not cands:
        return None
    logits = torch.tensor([c[0] for c in cands], dtype=torch.float32)
    logits = torch.where(torch.isfinite(logits), logits, torch.tensor(-1e9))
    probs = F.softmax(logits, dim=0)
    ix = int(torch.multinomial(probs, 1).item())
    return cands[ix][1]


def _value_best_legal(board: chess.Board, score_fn) -> Optional[str]:
    legal = list(board.legal_moves)
    if not legal:
        return None
    best_uci, best_s = None, -1e9
    for mv in legal:
        board.push(mv)
        s = score_fn(board)
        board.pop()
        if s > best_s:
            best_s, best_uci = s, mv.uci()
    return best_uci


def _value_best_among(board: chess.Board, ucis: List[str], score_fn) -> Optional[str]:
    best_uci, best_s = None, -1e9
    for uci in ucis:
        try:
            mv = chess.Move.from_uci(_promo_queen(board, uci))
            if mv not in board.legal_moves:
                continue
        except ValueError:
            continue
        board.push(mv)
        s = score_fn(board)
        board.pop()
        if s > best_s:
            best_s, best_uci = s, mv.uci()
    return best_uci


def _prod_stm_score(evalr: ChessEvaluator, board: chess.Board) -> float:
    """Production value is White POV in (-1,1); convert to side-to-move."""
    v_white = float(evalr.evaluate(board.fen()))
    return v_white if board.turn == chess.WHITE else -v_white


def _unified_stm_score(inf: UnifiedInference, board: chess.Board) -> float:
    return float(inf.evaluate_board(board))


def _mover_score_after(score_after_stm_fn, board: chess.Board) -> float:
    """After a candidate is pushed, STM is the opponent — mover wants -v_opp."""
    return -score_after_stm_fn(board)


class Side:
    def __init__(self, name: str, kind: str, policy, value):
        self.name = name
        self.kind = kind  # "prod" | "unified"
        self.policy = policy
        self.value = value

    def policy_topk(self, board, moves_uci, k) -> List[Tuple[float, str]]:
        if self.kind == "prod":
            return self.policy.top_k(board, moves_uci, k)
        moves = []
        tmp = chess.Board()
        for u in moves_uci:
            moves.append(chess.Move.from_uci(u) if isinstance(u, str) else u)
            tmp.push(moves[-1])
        return self.policy.predict_policy_board(board, moves, top_k=k)

    def stm_after(self, board: chess.Board) -> float:
        if self.kind == "prod":
            return _prod_stm_score(self.value, board)
        return _unified_stm_score(self.value, board)


def choose_move(
    side: Side,
    board: chess.Board,
    moves_uci: List[str],
    mode: str,
    top_k: int,
    p_value: float,
) -> Tuple[Optional[str], str]:
    """Return (uci, how) where how is policy|value|mix-policy|mix-value."""
    if mode == "policy":
        cands = side.policy_topk(board, moves_uci, top_k)
        return _sample_topk(cands), "policy"
    if mode == "value":
        uci = _value_best_legal(board, lambda b: _mover_score_after(side.stm_after, b))
        return uci, "value"
    # mix
    cands = side.policy_topk(board, moves_uci, top_k)
    if not cands:
        return None, "mix"
    if random.random() < p_value:
        ucis = [u for _, u in cands]
        uci = _value_best_among(board, ucis, lambda b: _mover_score_after(side.stm_after, b))
        return uci or _sample_topk(cands), "mix-value"
    return _sample_topk(cands), "mix-policy"


def play_game(
    white: Side,
    black: Side,
    mode: str,
    top_k: int,
    p_value: float,
    max_plies: int,
    delay: float,
) -> str:
    board = chess.Board()
    moves_uci: List[str] = []
    write_board(board, white.name, black.name, extra=f"mode={mode} ply=0")
    if delay:
        time.sleep(delay)

    for ply in range(max_plies):
        if board.is_game_over():
            break
        side = white if board.turn == chess.WHITE else black
        uci, how = choose_move(side, board, moves_uci, mode, top_k, p_value)
        if uci is None:
            break
        uci = _promo_queen(board, uci)
        board.push_uci(uci)
        moves_uci.append(uci)
        extra = f"mode={mode} ply={ply + 1} last={uci} via={how} by={side.name}"
        write_board(board, white.name, black.name, extra=extra)
        print(f"  {ply + 1:3d} {side.name:12s} {how:11s} {uci}")
        if delay:
            time.sleep(delay)

    result = board.result()
    extra = f"mode={mode} result={result}"
    write_board(board, white.name, black.name, extra=extra)
    return result


def main():
    ap = argparse.ArgumentParser(description="Production vs unified Phase B, live SVG board")
    ap.add_argument("--mode", choices=["policy", "value", "mix"], default="policy")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--delay", type=float, default=0.45)
    ap.add_argument("--top-k", type=int, default=7)
    ap.add_argument("--p-value", type=float, default=0.5, help="Bernoulli P(use value) in mix mode")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--unified", default=DEFAULT_UNIFIED)
    ap.add_argument("--prod-policy", default=DEFAULT_PROD_POLICY)
    ap.add_argument("--prod-value", default=DEFAULT_PROD_VALUE)
    ap.add_argument("--vocab-prod", default=DEFAULT_VOCAB_PROD)
    ap.add_argument("--vocab1", default=DEFAULT_VOCAB1)
    args = ap.parse_args()

    for p in (args.unified, args.prod_policy, args.prod_value, args.vocab_prod, args.vocab1):
        if not os.path.exists(p):
            print(f"Missing: {p}")
            sys.exit(1)

    device = _device()
    print(f"Device: {device}")
    print(f"Mode: {args.mode}  games={args.games}  top_k={args.top_k}  p_value={args.p_value}")
    print(f"Production policy: {args.prod_policy}")
    print(f"Production value:  {args.prod_value}")
    print(f"Unified:           {args.unified}")
    print(f"Board SVG:         {VIZ_PATH}")
    print("Open that SVG in a viewer that auto-reloads (VS Code preview, browser).")

    prod_pol = ProdPolicy(args.prod_policy, args.vocab_prod, device)
    prod_val = ChessEvaluator(args.prod_value, device=device)
    unified = UnifiedInference(args.unified, args.vocab1, device=device)

    prod = Side("production", "prod", prod_pol, prod_val)
    uni = Side("unified", "unified", unified, unified)

    w_prod = w_uni = draws = 0
    for g in range(args.games):
        if random.random() < 0.5:
            white, black = prod, uni
        else:
            white, black = uni, prod
        print(f"\n=== Game {g + 1}/{args.games}  White={white.name}  Black={black.name} ===")
        result = play_game(
            white, black, args.mode, args.top_k, args.p_value, args.max_plies, args.delay
        )
        print(f"Result {result}")
        if result == "1-0":
            if white.name == "production":
                w_prod += 1
            else:
                w_uni += 1
        elif result == "0-1":
            if black.name == "production":
                w_prod += 1
            else:
                w_uni += 1
        else:
            draws += 1

    print("\n===== tally =====")
    print(f"production wins: {w_prod}")
    print(f"unified wins:    {w_uni}")
    print(f"draws:           {draws}")


if __name__ == "__main__":
    main()
