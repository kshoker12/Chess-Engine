"""
Top-k agreement of a policy model with Stockfish on held-out positions.

Pick the model with the MODEL constant ("base" or "sidekick"), or override it
with --model. Both models are scored on the same positions: the sampled
positions and Stockfish labels are cached in CACHE_PATH, so the second run only
does model inference.

    python policy_transformer/eval_topk_stockfish.py                 # MODEL constant
    python policy_transformer/eval_topk_stockfish.py --model base

Protocol (same as evaluate_accuracy.py / compare_models.py): games with at least 7
plies, one position per game after a random ply k in [5, len-1] (evaluate_accuracy.py's
rule), label = Stockfish best move at 0.1s, a model's top-k = its ranking of the legal
moves by logit, history mirrored when Black is to move. Games are drawn at random from
the whole PGN (as compare_models.py does) rather than read from the start.
"""
import argparse
import io
import json
import os
import pickle
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine
import chess.pgn
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from model import PolicyHead
from inference import mirror_uci_string, get_token

# ── Which model to evaluate: "base" or "sidekick" ──
MODEL = "sidekick"

MODELS = {
    "base": dict(
        checkpoint=os.path.join(HERE, "ultra_3o9.pt"),
        vocab=os.path.join(HERE, "vocab.pkl"),
        n_embd=500, n_head=10, n_layer=8,
    ),
    "sidekick": dict(
        checkpoint=os.path.join(HERE, "best_sidekick_v1.pt"),
        vocab=os.path.join(HERE, "vocab1.pkl"),
        n_embd=256, n_head=8, n_layer=4,
    ),
}
# Both models are built with block_size 128 (as in compare_models.py). PolicyHead clips any
# longer history to its last 128 tokens, so evaluate_accuracy.py's 256-token truncation was
# effectively 128 too.
BLOCK_SIZE = 128

# ── Evaluation data and protocol ──
PGN_PATH = os.path.join(ROOT_DIR, "data", "eval_holdout", "lichess_elite_2025-02.pgn")
CACHE_PATH = os.path.join(ROOT_DIR, "data", "eval_holdout", "positions_10k_sf0.1.jsonl")
N_SAMPLES = 10_000
KS = (1, 3, 5, 7, 9)
MIN_PLIES = 7           # games shorter than this are skipped
MIN_K = 5               # earliest ply a position is sampled from
SEED = 123
STOCKFISH_TIME = 0.1    # seconds per position, same as evaluate_accuracy.py
STOCKFISH_WORKERS = 4   # one engine process each


def find_stockfish():
    path = os.environ.get("STOCKFISH_PATH")
    if path:
        return path
    for p in ("/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish", "/usr/bin/stockfish"):
        if os.path.exists(p):
            return p
    return "stockfish"


# ── Positions + Stockfish labels (shared by both models, cached) ──
def index_games(pgn_path):
    """(start, end) byte spans of every game, found from the '[Event ' header lines."""
    starts = []
    pos = 0
    with open(pgn_path, "rb") as f:
        for line in f:
            if line.startswith(b"[Event "):
                starts.append(pos)
            pos += len(line)
    return list(zip(starts, starts[1:] + [pos]))


def sample_positions(pgn_path, n, seed):
    """n random games -> one history (the first k moves, k random) each."""
    print(f"Indexing games in {pgn_path} ...")
    spans = index_games(pgn_path)
    print(f"  {len(spans):,} games")
    rng = random.Random(seed)
    rng.shuffle(spans)

    out = []
    with open(pgn_path, "rb") as f, tqdm(total=n, desc="Sampling positions") as pbar:
        for start, end in spans:
            if len(out) >= n:
                break
            f.seek(start)
            text = f.read(end - start).decode("utf-8", errors="replace")
            game = chess.pgn.read_game(io.StringIO(text))
            if game is None:
                continue
            moves = [m.uci() for m in game.mainline_moves()]
            if len(moves) < MIN_PLIES:
                continue
            k = rng.randint(MIN_K, len(moves) - 1)
            out.append({"moves": moves[:k]})
            pbar.update(1)
    return out


def label_positions(positions, stockfish_path, time_limit, workers):
    """Add Stockfish's best move to each position. Finished positions are dropped."""
    local = threading.local()
    engines = []
    lock = threading.Lock()

    def engine():
        if not hasattr(local, "e"):
            e = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            e.configure({"Threads": 1})
            local.e = e
            with lock:
                engines.append(e)
        return local.e

    def label(pos):
        board = chess.Board()
        for u in pos["moves"]:
            board.push_uci(u)
        if board.is_game_over():
            return None
        res = engine().play(board, chess.engine.Limit(time=time_limit))
        return {**pos, "best": res.move.uci()} if res.move else None

    try:
        with ThreadPoolExecutor(workers) as ex:
            labelled = list(tqdm(ex.map(label, positions), total=len(positions), desc="Stockfish labels"))
    finally:
        for e in engines:
            e.quit()
    return [p for p in labelled if p]


def load_or_build_positions():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            data = [json.loads(line) for line in f]
        print(f"Loaded {len(data):,} cached labelled positions from {CACHE_PATH}")
        return data

    # Over-sample slightly so dropped (finished) positions still leave N_SAMPLES.
    raw = sample_positions(PGN_PATH, N_SAMPLES + 200, SEED)
    positions = label_positions(raw, find_stockfish(), STOCKFISH_TIME, STOCKFISH_WORKERS)
    assert len(positions) >= N_SAMPLES, f"only {len(positions)} labelled positions"
    positions = positions[:N_SAMPLES]
    with open(CACHE_PATH, "w") as f:
        for p in positions:
            f.write(json.dumps(p) + "\n")
    print(f"Cached {len(positions):,} labelled positions to {CACHE_PATH}")
    return positions


# ── Model inference ──
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name, device):
    spec = MODELS[name]
    with open(spec["vocab"], "rb") as f:
        stoi = pickle.load(f)
    model = PolicyHead(
        vocab_size=len(stoi), n_embd=spec["n_embd"], block_size=BLOCK_SIZE,
        n_head=spec["n_head"], n_layer=spec["n_layer"], dropout=0.0, device=device,
    )
    ckpt = torch.load(spec["checkpoint"], map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded '{name}': {n_params:.2f}M params, vocab {len(stoi)}, {spec['checkpoint']}")
    return model, stoi


@torch.no_grad()
def ranked_legal_moves(model, stoi, device, moves, board):
    """Legal moves ranked by logit. History is mirrored when Black is to move
    (the models only ever see White-to-move), same convention as inference.py."""
    mirror = board.turn == chess.BLACK
    pad = stoi.get("|", 0)
    ids = []
    for u in moves:
        tok = get_token(mirror_uci_string(u) if mirror else u)
        ids.append(stoi.get(tok, pad))
    idx = torch.tensor([ids[-BLOCK_SIZE:]], dtype=torch.long, device=device)
    logits, _ = model(idx)
    last = logits[0, -1, :].float().cpu()

    scored = []
    for m in board.legal_moves:
        u = m.uci()
        tok = get_token(mirror_uci_string(u) if mirror else u)
        scored.append((last[stoi[tok]].item() if tok in stoi else float("-inf"), u))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [u for _, u in scored]


def evaluate(name, positions):
    device = pick_device()
    print(f"Device: {device}")
    model, stoi = load_model(name, device)

    hits = {k: 0 for k in KS}
    total = 0
    pbar = tqdm(positions, desc=f"Evaluating {name}")
    for pos in pbar:
        board = chess.Board()
        for u in pos["moves"]:
            board.push_uci(u)
        ranked = ranked_legal_moves(model, stoi, device, pos["moves"], board)
        for k in KS:
            hits[k] += pos["best"] in ranked[:k]
        total += 1
        pbar.set_postfix({f"Top-{k}": f"{hits[k] / total:.1%}" for k in (1, 7)})

    print(f"\nResults for '{name}' ({total} positions, Stockfish {STOCKFISH_TIME}s best move):")
    for k in KS:
        print(f"  Top-{k}: {hits[k]}/{total} ({100 * hits[k] / total:.2f}%)")
    return {k: hits[k] / total for k in KS}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default=MODEL)
    args = ap.parse_args()
    evaluate(args.model, load_or_build_positions())
