"""
Compare two policy models by simulating games between them.

Move selection per side:
1) sample candidate moves from that side's policy model
2) rank sampled moves with a shared value baseline (mini_value_2o2)
3) play the highest-valued sampled move
"""

import os
import sys
import random
import pickle
import re
import time
import torch
import torch.nn.functional as F
import chess

# Project root for imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from policy_transformer.model import PolicyHead
import policy_transformer.inference as policy_infer
from value_transformer.inference import ChessEvaluator

MAX_PLIES = 200
NUM_GAMES = 10
MOVE_DELAY_SECONDS = 0.5
POLICY_TOP_K = 7
VIZ_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulate_viz", "board.svg")


def _device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.backends.mps.is_available():
        device = "mps"
    return device


def _load_vocab():
    vocab_path = os.path.join(ROOT, "policy_transformer", "vocab.pkl")
    with open(vocab_path, "rb") as f:
        stoi = pickle.load(f)
    return stoi


def _load_policy_model(checkpoint_path: str, vocab_size: int, device: str):
    if "mega" in os.path.basename(checkpoint_path).lower():
        model = PolicyHead(
            vocab_size=vocab_size,
            n_embd=768,
            block_size=256,
            n_head=12,
            n_layer=10,
            dropout=0.0,
            device=device,
        )
    else:
        model = PolicyHead(
            vocab_size=vocab_size,
            n_embd=500,
            block_size=128,
            n_head=10,
            n_layer=8,
            dropout=0.0,
            device=device,
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if isinstance(state_dict, dict) and any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _encode_history_tokens(moves_uci, should_mirror, stoi):
    token_ids = []
    if not moves_uci:
        return [0]

    for uci in moves_uci:
        encoded_uci = policy_infer.mirror_uci_string(uci) if should_mirror else uci
        token = policy_infer.get_token(encoded_uci)
        token_ids.append(stoi.get(token, stoi.get("|", 0)))

    if not token_ids:
        token_ids = [0]
    return token_ids


def _policy_candidates(board, moves_uci, model, stoi, device, top_k):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return []

    should_mirror = board.turn == chess.BLACK
    token_ids = _encode_history_tokens(moves_uci, should_mirror, stoi)
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        if idx.shape[1] > model.block_size:
            idx = idx[:, -model.block_size :]
        logits, _ = model(idx)
        last_logits = logits[0, -1, :]

    candidates = []
    for move in legal_moves:
        uci = move.uci()
        search_uci = policy_infer.mirror_uci_string(uci) if should_mirror else uci
        token = policy_infer.get_token(search_uci)
        token_id = stoi.get(token)
        score = last_logits[token_id].item() if token_id is not None else -float("inf")
        candidates.append((score, uci))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:top_k]


def _promotion_to_queen(board, uci):
    """If uci is a pawn promotion, return uci with promotion piece set to queen."""
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


def choose_move(board, moves_uci, policy_model, stoi, device):
    candidates = _policy_candidates(
        board=board,
        moves_uci=moves_uci,
        model=policy_model,
        stoi=stoi,
        device=device,
        top_k=POLICY_TOP_K,
    )
    if not candidates:
        return None

    logits = torch.tensor([c[0] for c in candidates], dtype=torch.float32)
    probs = F.softmax(logits, dim=0)
    sampled_ix = torch.multinomial(probs, num_samples=1, replacement=False).tolist()
    best_move = candidates[sampled_ix[0]][1]
    return best_move


def _svg_with_legend(svg_str: str, white_name: str, black_name: str) -> str:
    """Add a one-line legend at the bottom: White: X  Black: Y (white background)."""
    match = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg_str)
    if not match:
        return svg_str
    w, h = int(match.group(1)), int(match.group(2))
    extra = 28
    new_h = h + extra
    svg_str = svg_str.replace(f'viewBox="0 0 {w} {h}"', f'viewBox="0 0 {w} {new_h}"', 1)
    legend = f'<rect x="0" y="{h}" width="{w}" height="{extra}" fill="#ffffff"/><text x="{w/2}" y="{h + 20}" text-anchor="middle" font-size="16" font-family="sans-serif" fill="#1a1a1a">White: {white_name}   Black: {black_name}</text>'
    if svg_str.strip().endswith("</svg>"):
        svg_str = svg_str[:-6] + "\n" + legend + "\n</svg>"
    return svg_str


def play_game(white_model, black_model, stoi, device, move_delay_sec=0, viz_path=None, white_name="A", black_name="B"):
    board = chess.Board()
    moves_uci = []

    for _ in range(MAX_PLIES):
        if board.is_game_over():
            break

        model = white_model if board.turn == chess.WHITE else black_model
        move_uci = choose_move(board, moves_uci, model, stoi, device)
        if move_uci is None:
            break
        move_uci = _promotion_to_queen(board, move_uci)

        board.push_uci(move_uci)
        moves_uci.append(move_uci)

        if viz_path:
            os.makedirs(os.path.dirname(viz_path) or ".", exist_ok=True)
            svg_str = chess.svg.board(board=board, size=400)
            svg_str = _svg_with_legend(svg_str, white_name, black_name)
            with open(viz_path, "w") as f:
                f.write(svg_str)
            if move_delay_sec > 0:
                time.sleep(move_delay_sec)

    return board.result()


def main():
    default_a = os.path.join(ROOT, "policy_transformer", "ultra_3o9.pt")
    default_b = os.path.join(ROOT, "policy_transformer", "checkpoints", "mega_ultra_3o3.pt")

    path_a = sys.argv[1] if len(sys.argv) > 1 else default_a
    path_b = sys.argv[2] if len(sys.argv) > 2 else default_b
    num_games = int(sys.argv[3]) if len(sys.argv) > 3 else NUM_GAMES

    if not os.path.exists(path_a) or not os.path.exists(path_b):
        print(f"Policy model not found. A: {path_a}, B: {path_b}")
        sys.exit(1)

    device = _device()
    print(f"Policy A: {path_a}")
    print(f"Policy B: {path_b}")
    print(f"Device: {device}")
    print(f"Board: {VIZ_OUTPUT_PATH}")
    print(f"Playing {num_games} games (random white/black each game)...")

    stoi = _load_vocab()
    model_a = _load_policy_model(path_a, len(stoi), device)
    model_b = _load_policy_model(path_b, len(stoi), device)

    wins_a = 0
    wins_b = 0
    draws = 0

    for g in range(num_games):
        if random.random() < 0.5:
            white_model, black_model = model_a, model_b
            white_name, black_name = "A", "B"
        else:
            white_model, black_model = model_b, model_a
            white_name, black_name = "B", "A"

        result = play_game(
            white_model, black_model, stoi, device,
            move_delay_sec=MOVE_DELAY_SECONDS,
            viz_path=VIZ_OUTPUT_PATH,
            white_name=white_name,
            black_name=black_name,
        )
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
    print(f"Policy A wins: {wins_a}")
    print(f"Policy B wins: {wins_b}")
    print(f"Draws: {draws}")
    if wins_a > wins_b:
        print(f"Champion: Policy A ({path_a})")
    elif wins_b > wins_a:
        print(f"Champion: Policy B ({path_b})")
    else:
        print("Champion: Tie")


if __name__ == "__main__":
    main()
