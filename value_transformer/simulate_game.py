"""
Compare two value models by simulating 10 games between them.
Move logic matches get_transformer_move (one-ply lookahead + value + policy + approximate_cp).
Randomly assign models to white/black each game; print champion (no persistence).
"""
import os
import sys
import io
import random
import numpy as np
import torch
import torch.nn.functional as F
import chess
import chess.pgn

# Project root for imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from policy_transformer.inference import predict_next_move
from value_transformer.inference import ChessEvaluator
from value_transformer.approximate import approximate_cp

MAX_PLIES = 200
NUM_GAMES = 400


def _pgn_from_moves(moves_uci):
    """Build PGN string from list of UCI moves (replay from start, collect SANs)."""
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


def get_move(pgn: str, evaluator: ChessEvaluator):
    """Mirror get_transformer_move: top_k=12, one-ply lookahead, return best move UCI or None."""
    pgn_io = io.StringIO(pgn)
    game = chess.pgn.read_game(pgn_io)
    root_board = game.end().board() if game else chess.Board()

    top_moves = predict_next_move(pgn, top_k=7)
    if not top_moves:
        return None

    logits = torch.tensor([s for s, _ in top_moves], dtype=torch.float32)
    probs = F.softmax(logits, dim=0).tolist()

    best_move = None
    best_score = -float("inf")

    for i, (score, move) in enumerate(top_moves):
        prob = probs[i]
        board = root_board.copy()
        try:
            board.push_uci(move)
        except ValueError:
            continue

        score = -evaluator.evaluate(board.fen() if board.turn == chess.WHITE else board.mirror().fen())
        lam = random.uniform(0, 0.05)
        combined_score = (1 - lam) * score + lam * prob
        if combined_score > best_score:
            best_score = combined_score
            best_move = move

    return best_move


def play_game(white_evaluator: ChessEvaluator, black_evaluator: ChessEvaluator):
    """Play one game; return '1-0', '0-1', or '1/2-1/2'."""
    board = chess.Board()
    moves_uci = []

    for _ in range(MAX_PLIES):
        if board.is_game_over():
            break
        evaluator = white_evaluator if board.turn == chess.WHITE else black_evaluator
        pgn_str = _pgn_from_moves(moves_uci)
        move_uci = get_move(pgn_str, evaluator)
        if move_uci is None:
            break
        board.push_uci(move_uci)
        moves_uci.append(move_uci)

    return board.result()


def main():
    default_a = os.path.join(ROOT, "value_transformer", "mini_value_6o4.pt")
    default_b = os.path.join(ROOT, "value_transformer", "checkpoints", "mini_value_2o2.pt")
    path_a = sys.argv[1] if len(sys.argv) > 1 else default_a
    path_b = sys.argv[2] if len(sys.argv) > 2 else default_b

    if not os.path.exists(path_a) or not os.path.exists(path_b):
        print(f"Model not found. A: {path_a}, B: {path_b}")
        sys.exit(1)

    print(f"Model A: {path_a}")
    print(f"Model B: {path_b}")
    print(f"Playing {NUM_GAMES} games (random white/black each game)...")

    model_a = ChessEvaluator(path_a)
    model_b = ChessEvaluator(path_b)

    wins_a = 0
    wins_b = 0
    draws = 0

    for g in range(NUM_GAMES):
        if random.random() < 0.5:
            white_eval, black_eval = model_a, model_b
            white_name, black_name = "A", "B"
        else:
            white_eval, black_eval = model_b, model_a
            white_name, black_name = "B", "A"
        result = play_game(white_eval, black_eval)
        if result == "1-0":
            wins_a += 1 if white_name == "A" else 0
            wins_b += 1 if white_name == "B" else 0
        elif result == "0-1":
            wins_a += 1 if black_name == "A" else 0
            wins_b += 1 if black_name == "B" else 0
        else:
            draws += 1
        print(f"  Game {g + 1}: {result} (W={white_name} B={black_name})")

    print()
    print(f"Model A wins: {wins_a}")
    print(f"Model B wins: {wins_b}")
    print(f"Draws: {draws}")
    if wins_a > wins_b:
        print(f"Champion: Model A ({path_a})")
    elif wins_b > wins_a:
        print(f"Champion: Model B ({path_b})")
    else:
        print("Champion: Tie")


if __name__ == "__main__":
    main()
