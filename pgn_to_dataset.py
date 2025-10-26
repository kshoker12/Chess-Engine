# pgn_to_dataset.py
import os, json, glob
import numpy as np
import chess
import chess.pgn
import chess.engine
import argparse
from tqdm import tqdm
from features import encode_features

# Configuration
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH") or "/opt/homebrew/bin/stockfish"
ENGINE_TIME_MS = 300  # Evaluation time per position
CP_CLAMP = 1000  # Clamp evaluations to ±1000 centipawns

def clamp_cp(score: chess.engine.PovScore, turn_white: bool) -> int:
    """Convert Stockfish score to centipawns from side-to-move perspective."""
    if score.is_mate():
        # Represent mates as large cp with sign
        sign = 1 if score.white().mate() and turn_white else -1 if score.black().mate() and not turn_white else 1
        return int(np.sign(sign) * CP_CLAMP)
    cp = score.white().score(mate_score=CP_CLAMP)  # from White's perspective
    # if side to move is black, invert
    if not turn_white:
        cp = -cp
    # clamp
    cp = max(min(cp, CP_CLAMP), -CP_CLAMP)
    return int(cp)

def ensure_engine():
    """Get a Stockfish engine instance."""
    path = STOCKFISH_PATH
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"STOCKFISH_PATH not set or invalid: {path}\n"
            "Set it, e.g.: export STOCKFISH_PATH=/opt/homebrew/bin/stockfish"
        )
    engine = chess.engine.SimpleEngine.popen_uci(path)
    engine.configure({"Threads": 1})
    return engine

def process_pgn_to_dataset(pgn_file, X_out, y_out):
    """Process PGN file and append positions to X_out and y_out numpy files."""
    
    # Initialize Stockfish engine
    engine = ensure_engine()
    limit = chess.engine.Limit(time=ENGINE_TIME_MS / 1000.0)
    
    # Check if output files exist, load them if they do
    X_list = []
    y_list = []
    
    if os.path.exists(X_out) and os.path.exists(y_out):
        X_list = list(np.load(X_out))
        y_list = list(np.load(y_out))
        print(f"Loaded {len(X_list):,} existing positions from {X_out}")
    
    seen_positions = set()
    
    # Process PGN file
    with open(pgn_file, 'r') as f:
        game_count = 0
        position_count = 0
        
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            game_count += 1
            board = game.board()
            
            # Extract moves from the game
            for node in game.mainline():
                board.push(node.move)
                
                # Skip terminal positions
                if board.is_game_over():
                    continue
                
                # Deduplication using transposition key
                z = hash(board._transposition_key())
                if z in seen_positions:
                    continue
                seen_positions.add(z)
                
                # Encode features
                features = encode_features(board)
                X_list.append(features)
                
                # Evaluate position with Stockfish
                try:
                    info = engine.analyse(board, limit=limit)
                    score = clamp_cp(info["score"], board.turn == chess.WHITE)
                    y_list.append(score)
                except Exception as e:
                    print(f"Error evaluating position: {e}")
                    y_list.append(0)  # Fallback to 0 on error
                
                position_count += 1
                
                # Progress indicator
                if position_count % 100 == 0:
                    print(f"Processed {position_count:,} positions from {game_count} games", end='\r')
        
        print(f"\nProcessed {game_count} games, {position_count:,} positions")
    
    # Clean up engine
    engine.quit()
    
    # Convert to numpy arrays and save
    X_new = np.array(X_list, dtype=np.float32)
    y_new = np.array(y_list, dtype=np.int16)
    
    print(f"\nSaving {len(X_new):,} positions to {X_out} and {y_out}")
    np.save(X_out, X_new)
    np.save(y_out, y_new)
    
    # Update metadata if exists
    meta_file = X_out.replace('_X', '_meta').replace('.npy', '.json')
    meta = {
        "n": int(len(X_new)),
        "source": pgn_file,
        "note": "PGN positions evaluated with Stockfish",
        "engine_ms": ENGINE_TIME_MS,
        "cp_clamp": CP_CLAMP
    }
    
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_file}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", help="Path to PGN file (or folder)")
    args = ap.parse_args()
    
    # Collect all PGN files
    pgn_files = []
    if args.pgn:
        if os.path.isdir(args.pgn):
            pgn_files = glob.glob(os.path.join(args.pgn, "*.pgn"))
        else:
            pgn_files = [args.pgn]
    else:
        # Default: process games/ folder
        pgn_files = glob.glob("games/*.pgn")
    
    print(f"Found {len(pgn_files)} PGN files to process")
    
    # Process to both alt and copy datasets
    print("\n=== Processing to X_alt/y_alt ===")
    for pgn_file in pgn_files:
        print(f"\nProcessing: {pgn_file}")
        process_pgn_to_dataset(pgn_file, "dataset_X_best.npy", "dataset_y_best.npy")
    
    print("\nDone!")

if __name__ == "__main__":
    main()

