#!/usr/bin/env python3
"""
Production vs unified.

Modes (same move logic as the production simulate_game scripts, plus MCTS):
  policy  — policy_transformer/simulate_game.py
            each side samples softmax over its own policy top-k
  value   — value_transformer/simulate_game.py get_move
            own policy top-k, one-ply own value, small policy-prob mix
  mcts    — engine/mcts.py, 100 sims (default), visit-count pick

Pass --vis to write a live SVG board (unified_mcts/simulate_viz/board.svg).
Colors randomized each game. Production weights are read-only.
"""
from __future__ import annotations

import argparse
import io
import os
import pickle
import random
import re
import sys
import time
from typing import Callable, List, Optional, Tuple

import chess
import chess.pgn
import chess.svg
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from engine.mcts import MCTS
from policy_transformer.model import PolicyHead
from value_transformer.inference import ChessEvaluator
from unified_mcts.board_utils import get_uci4_token, mirror_uci_string
from unified_mcts.inference import UnifiedInference

DEFAULT_UNIFIED = os.path.join(HERE, "ckpts", "unified_phase_b_best.pt")
DEFAULT_PROD_POLICY = os.path.join(ROOT, "policy_transformer", "ultra_3o9.pt")
DEFAULT_PROD_VALUE = os.path.join(ROOT, "value_transformer", "mini_value_6o4.pt")
DEFAULT_VOCAB_PROD = os.path.join(ROOT, "policy_transformer", "vocab.pkl")
DEFAULT_VOCAB1 = os.path.join(ROOT, "policy_transformer", "vocab1.pkl")

POLICY_TOP_K = 7
VALUE_TOP_K = 7
MCTS_TOP_K = 10
MAX_PLIES = 200
VIZ_PATH = os.path.join(HERE, "simulate_viz", "board.svg")


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


def _promotion_to_queen(board: chess.Board, uci: str) -> str:
    """Match policy_transformer/simulate_game.py: force queen on last rank."""
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
    if chess.square_rank(to_sq) in (0, 7):
        return uci[:4] + "q"
    return uci


def _pgn_from_moves(moves_uci: List[str]) -> str:
    """Match value_transformer/simulate_game.py."""
    if not moves_uci:
        return ""
    board = chess.Board()
    parts = []
    for uci in moves_uci:
        move = chess.Move.from_uci(uci)
        san = board.san(move)
        if board.turn == chess.WHITE:
            parts.append(f"{board.fullmove_number}. {san}")
        else:
            parts.append(san)
        board.push(move)
    return " ".join(parts)


def _moves_from_uci(moves_uci: List[str]) -> List[chess.Move]:
    tmp = chess.Board()
    out = []
    for u in moves_uci:
        mv = chess.Move.from_uci(u)
        tmp.push(mv)
        out.append(mv)
    return out


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


def _sample_policy(cands: List[Tuple[float, str]]) -> Optional[str]:
    if not cands:
        return None
    logits = torch.tensor([c[0] for c in cands], dtype=torch.float32)
    logits = torch.where(torch.isfinite(logits), logits, torch.tensor(-1e9))
    probs = F.softmax(logits, dim=0)
    ix = int(torch.multinomial(probs, num_samples=1).item())
    return cands[ix][1]


def _value_get_move(
    cands: List[Tuple[float, str]],
    board: chess.Board,
    mover_score_fn: Callable[[chess.Board], float],
) -> Optional[str]:
    """value_transformer/simulate_game.py get_move: 1-ply value + lam * policy prob."""
    if not cands:
        return None
    logits = torch.tensor([s for s, _ in cands], dtype=torch.float32)
    logits = torch.where(torch.isfinite(logits), logits, torch.tensor(-1e9))
    probs = F.softmax(logits, dim=0).tolist()
    best_move = None
    best_score = -float("inf")
    for i, (_, move) in enumerate(cands):
        b = board.copy()
        try:
            b.push_uci(_promotion_to_queen(b, move))
        except ValueError:
            continue
        score = mover_score_fn(b)
        lam = random.uniform(0, 0.05)
        combined = (1 - lam) * score + lam * probs[i]
        if combined > best_score:
            best_score = combined
            best_move = move
    return best_move


def _prod_mover_score(evalr: ChessEvaluator, board_after: chess.Board) -> float:
    """After a push, production evaluate on STM-white FEN, then negate (mover POV)."""
    fen = board_after.fen() if board_after.turn == chess.WHITE else board_after.mirror().fen()
    return -float(evalr.evaluate(fen))


def _uni_mover_score(inf: UnifiedInference, board_after: chess.Board) -> float:
    return -float(inf.evaluate_board(board_after))


class QuietMCTS(MCTS):
    """Same search as engine.mcts.MCTS without per-move stdout."""

    def search(self):
        for i in range(self.num_simulations):
            node = self._select(self.root)
            value = self._evaluate(node)
            if i < self.num_simulations - 1 and not node.board.is_game_over():
                self._expand(node)
            self._backpropagate(node, value)

        best_move = None
        best_score = -float("inf")
        for key, child in self.root.children.items():
            score = child.N
            if (self.root_fen, key) in self.recent_pairs:
                score *= 0.95
            if score > best_score:
                best_score = score
                best_move = key
        return best_move, best_score


def _make_prod_mcts_funcs(prod_pol: ProdPolicy, prod_val: ChessEvaluator):
    def policy_func(pgn: str, top_k: int = MCTS_TOP_K):
        game = chess.pgn.read_game(io.StringIO(pgn)) if pgn else None
        board = chess.Board()
        moves_uci: List[str] = []
        if game is not None:
            for m in game.mainline_moves():
                board.push(m)
                moves_uci.append(m.uci())
        return prod_pol.top_k(board, moves_uci, top_k)

    def value_func(fens: List[str]) -> List[float]:
        return prod_val.batch_evaluate(fens)

    return policy_func, value_func


def _make_uni_mcts_funcs(inf: UnifiedInference):
    def policy_func(pgn: str, top_k: int = MCTS_TOP_K):
        return inf.predict_policy(pgn, top_k=top_k)

    def value_func(fens: List[str]) -> List[float]:
        return [inf.evaluate_board(chess.Board(fen)) for fen in fens]

    return policy_func, value_func


class Player:
    def __init__(self, name: str, kind: str, policy, value, mcts_funcs=None):
        self.name = name
        self.kind = kind
        self.policy = policy
        self.value = value
        self.mcts_funcs = mcts_funcs

    def policy_topk(self, board, moves_uci, k) -> List[Tuple[float, str]]:
        if self.kind == "prod":
            return self.policy.top_k(board, moves_uci, k)
        return self.policy.predict_policy_board(board, _moves_from_uci(moves_uci), top_k=k)

    def mover_score(self, board_after: chess.Board) -> float:
        if self.kind == "prod":
            return _prod_mover_score(self.value, board_after)
        return _uni_mover_score(self.value, board_after)


def choose_move(player: Player, board: chess.Board, moves_uci: List[str], mode: str, sims: int):
    if mode == "policy":
        cands = player.policy_topk(board, moves_uci, POLICY_TOP_K)
        return _sample_policy(cands)
    if mode == "value":
        cands = player.policy_topk(board, moves_uci, VALUE_TOP_K)
        return _value_get_move(cands, board, player.mover_score)
    pgn = _pgn_from_moves(moves_uci)
    policy_func, value_func = player.mcts_funcs
    searcher = QuietMCTS(pgn, value_func, policy_func, num_simulations=sims)
    searcher.k = MCTS_TOP_K
    best, _ = searcher.search()
    return best


def play_game(
    white: Player,
    black: Player,
    mode: str,
    max_plies: int,
    sims: int,
    vis: bool,
    delay: float,
) -> str:
    board = chess.Board()
    moves_uci: List[str] = []
    if vis:
        write_board(board, white.name, black.name, extra=f"mode={mode} ply=0")
        if delay:
            time.sleep(delay)
    for ply in range(max_plies):
        if board.is_game_over():
            break
        side = white if board.turn == chess.WHITE else black
        uci = choose_move(side, board, moves_uci, mode, sims)
        if uci is None:
            break
        uci = _promotion_to_queen(board, uci)
        try:
            board.push_uci(uci)
        except ValueError:
            break
        moves_uci.append(uci)
        extra = f"mode={mode} ply={ply + 1} last={uci} by={side.name}"
        if vis:
            write_board(board, white.name, black.name, extra=extra)
            if delay:
                time.sleep(delay)
            print(f"    {ply + 1:3d} {side.name:12s} {uci}")
    result = board.result()
    if vis:
        write_board(board, white.name, black.name, extra=f"mode={mode} result={result}")
    return result


def run_match(
    prod: Player,
    uni: Player,
    mode: str,
    games: int,
    max_plies: int,
    sims: int,
    vis: bool,
    delay: float,
):
    w_prod = w_uni = draws = unfinished = 0
    for g in range(games):
        if random.random() < 0.5:
            white, black = prod, uni
        else:
            white, black = uni, prod
        if vis:
            print(f"\n=== Game {g + 1}/{games}  White={white.name}  Black={black.name} ===")
        result = play_game(white, black, mode, max_plies, sims, vis, delay)
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
        elif result == "1/2-1/2":
            draws += 1
        else:
            unfinished += 1
        print(f"  Game {g + 1}/{games}: {result}  W={white.name} B={black.name}")

    print()
    print("===== tally =====")
    print(f"production wins: {w_prod}")
    print(f"unified wins:    {w_uni}")
    print(f"draws:           {draws}")
    print(f"unfinished (*):  {unfinished}")
    if w_prod > w_uni:
        champ = "production"
    elif w_uni > w_prod:
        champ = "unified"
    else:
        champ = "tie"
    print(f"Champion: {champ}")


def main():
    ap = argparse.ArgumentParser(description="Production vs unified: policy / value / mcts")
    ap.add_argument("--mode", choices=["policy", "value", "mcts"], required=True)
    ap.add_argument("--games", type=int, default=None, help="Default 100 (policy/value) or 10 (mcts)")
    ap.add_argument("--sims", type=int, default=100, help="MCTS simulations per move")
    ap.add_argument("--max-plies", type=int, default=MAX_PLIES)
    ap.add_argument("--vis", action="store_true", help="Write live SVG to simulate_viz/board.svg")
    ap.add_argument("--delay", type=float, default=None, help="Seconds between moves when --vis (default 0.4)")
    ap.add_argument("--unified", default=DEFAULT_UNIFIED)
    ap.add_argument("--prod-policy", default=DEFAULT_PROD_POLICY)
    ap.add_argument("--prod-value", default=DEFAULT_PROD_VALUE)
    ap.add_argument("--vocab-prod", default=DEFAULT_VOCAB_PROD)
    ap.add_argument("--vocab1", default=DEFAULT_VOCAB1)
    args = ap.parse_args()

    if args.games is None:
        args.games = 10 if args.mode == "mcts" else 100
    delay = 0.0 if not args.vis else (0.4 if args.delay is None else args.delay)

    for p in (args.unified, args.prod_policy, args.prod_value, args.vocab_prod, args.vocab1):
        if not os.path.exists(p):
            print(f"Missing: {p}")
            sys.exit(1)

    device = _device()
    print(f"Device: {device}")
    print(f"Mode: {args.mode}  games={args.games}  max_plies={args.max_plies}"
          + (f"  sims={args.sims}" if args.mode == "mcts" else ""))
    print(f"Production policy: {args.prod_policy}")
    print(f"Production value:  {args.prod_value}")
    print(f"Unified:           {args.unified}")
    if args.vis:
        print(f"Board SVG:         {VIZ_PATH}")
        print("Open that SVG in a viewer that auto-reloads (VS Code preview, browser).")

    prod_pol = ProdPolicy(args.prod_policy, args.vocab_prod, device)
    prod_val = ChessEvaluator(args.prod_value, device=device)
    unified = UnifiedInference(args.unified, args.vocab1, device=device)

    prod = Player("production", "prod", prod_pol, prod_val, _make_prod_mcts_funcs(prod_pol, prod_val))
    uni = Player("unified", "unified", unified, unified, _make_uni_mcts_funcs(unified))

    run_match(prod, uni, args.mode, args.games, args.max_plies, args.sims, args.vis, delay)


if __name__ == "__main__":
    main()
