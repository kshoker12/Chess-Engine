# data_gen.py
import os, sys, json, random, time
import numpy as np
from tqdm import tqdm
import chess
import chess.pgn
import chess.engine
from features import encode_features

# ---------------- Config ----------------
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH")  # set this!
OUT_PREFIX = "dataset"          # will write dataset_X.npy, dataset_y.npy
N_POSITIONS = 100_000
MAX_PLIES_FROM_START = 80       # randomize phase (open/mid/end)
SAMPLE_FROM_PGN = None          # e.g., "games.pgn" or leave None
ENGINE_TIME_MS = 300            # 200–500ms/position works well
CP_CLAMP = 1000                 # clamp labels to ±1000 cp
RANDOM_SEED = 42

# ----------------------------------------

def ensure_engine():
    path = STOCKFISH_PATH
    if not path or not os.path.exists(path):
        print("ERROR: STOCKFISH_PATH env var not set or invalid path.")
        print("Set it, e.g.: export STOCKFISH_PATH=/opt/homebrew/bin/stockfish")
        sys.exit(1)
    return chess.engine.SimpleEngine.popen_uci(path)


def clamp_cp(score: chess.engine.PovScore, turn_white: bool) -> int:
    # Turn score into cp from side-to-move perspective
    if score.is_mate():
        # Represent mates as large cp with sign; but clamp to CP_CLAMP for regression
        sign = 1 if score.white().mate() and turn_white else -1 if score.black().mate() and not turn_white else 1
        return int(np.sign(sign) * CP_CLAMP)
    cp = score.white().score(mate_score=CP_CLAMP)  # from White's perspective
    # if side to move is black, invert
    if not turn_white:
        cp = -cp
    # clamp
    cp = max(min(cp, CP_CLAMP), -CP_CLAMP)
    return int(cp)

def random_position_from_start() -> chess.Board:
    b = chess.Board()
    plies = random.randint(6, MAX_PLIES_FROM_START)
    for _ in range(plies):
        if b.is_game_over():
            break
        moves = list(b.legal_moves)
        if not moves:
            break
        b.push(random.choice(moves))
    return b

def positions_from_pgn(pgn_path: str):
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            b = game.board()
            for move in game.mainline_moves():
                b.push(move)
                yield b.copy()

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    engine = ensure_engine()
    engine.configure({"Threads": 1})  # stability per process
    limit = chess.engine.Limit(time=ENGINE_TIME_MS / 1000.0)

    X, y = [], []
    seen = set()  # transposition keys for de-dup
    pbar = tqdm(total=N_POSITIONS, desc="Labeling positions")

    def maybe_add(board: chess.Board):
        # Skip trivial terminal or illegal (shouldn't happen)
        if board.is_game_over():
            return False
        z = hash(board._transposition_key())
        if z in seen:
            return False
        # avoid extremely early plies for diversity
        if board.fullmove_number <= 2:
            return False

        # Evaluate with engine
        try:
            info = engine.analyse(board, limit=limit)
            score = clamp_cp(info["score"], board.turn)
        except Exception:
            return False

        X.append(encode_features(board))
        y.append(score)
        seen.add(z)
        pbar.update(1)
        return True

    # Source positions either from PGN or random playouts
    try:
        if SAMPLE_FROM_PGN:
            for b in positions_from_pgn(SAMPLE_FROM_PGN):
                if len(X) >= N_POSITIONS: break
                maybe_add(b)
        # If not enough, top up with random playouts
        while len(X) < N_POSITIONS:
            b = random_position_from_start()
            maybe_add(b)
    finally:
        pbar.close()
        engine.quit()

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int16)
    np.save(f"{OUT_PREFIX}_X.npy", X)
    np.save(f"{OUT_PREFIX}_y.npy", y)

    # Write a small manifest
    meta = {
        "n": int(len(X)),
        "engine_ms": ENGINE_TIME_MS,
        "cp_clamp": CP_CLAMP,
        "pgn": SAMPLE_FROM_PGN,
        "seed": RANDOM_SEED
    }
    with open(f"{OUT_PREFIX}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved {len(X)} samples to {OUT_PREFIX}_X.npy / {OUT_PREFIX}_y.npy")

if __name__ == "__main__":
    main()