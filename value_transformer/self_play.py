import sys
import os
import shutil
import numpy as np
import chess
import chess.engine
import chess.pgn
import random
import time
import torch.multiprocessing as mp


# --- Imports (Flat Structure for Kaggle) ---
# Ensure your inference files are in the current directory or python path
try:
    from policy_transformer.inference import predict_next_move
    # Assuming you will name your value inference file 'value_inference.py'
    from value_transformer.inference import ChessEvaluator 
except ImportError:
    print("WARNING: Could not import local modules. Ensure inference.py and value_inference.py are uploaded.")
    sys.exit(1)

# --- Configuration ---
TOTAL_GAMES = 300
CORES = 4

# Path where you uploaded the stockfish binary in Kaggle inputs
# UPDATE THIS PATH to match your actual uploaded file location on Kaggle
KAGGLE_SF_INPUT_PATH = "/kaggle/input/stockfish-16-linux/stockfish-ubuntu-x86-64-avx2"
SF_EXECUTABLE = "./stockfish_exec"

# --- Stockfish Setup for Kaggle ---
def setup_stockfish():
    """Copies Stockfish from input to working dir and makes it executable."""
    if os.path.exists(SF_EXECUTABLE) and os.access(SF_EXECUTABLE, os.X_OK):
        return SF_EXECUTABLE
        
    if not os.path.exists(KAGGLE_SF_INPUT_PATH):
        # Fallback: Look in current dir if user uploaded it directly to working
        if os.path.exists("stockfish"):
            print("Found stockfish in current dir, setting permissions...")
            os.system("chmod +x stockfish")
            return "./stockfish"
        print(f"ERROR: Stockfish not found at {KAGGLE_SF_INPUT_PATH}")
        return None

    try:
        print(f"Copying Stockfish from {KAGGLE_SF_INPUT_PATH} to {SF_EXECUTABLE}...")
        shutil.copy(KAGGLE_SF_INPUT_PATH, SF_EXECUTABLE)
        os.system(f"chmod +x {SF_EXECUTABLE}")
        time.sleep(1)
    except:
        pass
        
    return SF_EXECUTABLE

# ---------------- MAPPINGS ----------------
piece_map = {
    '.': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}

def get_board_tokens(board):
    tokens = []
    # Standard 8x8 flat tokenization (Rank 7->0, File 0->7)
    for rank in range(7, -1, -1):
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            tokens.append(piece_map[piece.symbol()] if piece else 0)
    return tokens

def eval_stockfish(board, engine):
    if not engine:
        return 0 
        
    try:
        info = engine.analyse(board, chess.engine.Limit(time=0.5, depth = 22))
        return info["score"].white().score(mate_score=10000)
    except Exception as e:
        print(f"Engine Error: {e}")
        return 0

def get_pgn(moves):
    game = chess.pgn.Game()
    node = game
    for move_uci in moves:
        node = node.add_variation(chess.Move.from_uci(move_uci))
    
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    return game.accept(exporter)

def get_random_move(top_moves):
    # top_moves is list of (score, uci)
    moves = [move for (score, move) in top_moves]
    logits = [score for (score, move) in top_moves]
    probs = np.exp(np.array(logits))
    probs /= probs.sum() + 1e-12
    dir_alpha = 0.25
    dirichlet = np.random.dirichlet([dir_alpha] * len(probs))
    noised_probs = 0.75 * probs + 0.25 * dirichlet
    noised_probs /= noised_probs.sum() + 1e-12
    return np.random.choice(moves, p = noised_probs)

def play_single_game(game_id):
    print(f"Process {os.getpid()} starting Game {game_id}")

    # 1. Setup Resources Local to this Process
    sf_path = SF_EXECUTABLE
    evaluator = ChessEvaluator()

    board = chess.Board()
    moves = []
    turn = 0
    temp = 0.7

    local_X = []
    local_Y = []

    consecutive_high_score = 0

    white_exploitative = random.uniform(0, 1) > 0.5
    white_alt_policy = random.uniform(0, 1) > 0.5

    engine = None
    try:
        if sf_path and os.path.exists(sf_path):
            engine = chess.engine.SimpleEngine.popen_uci(sf_path)
            engine.configure({"Threads": 1})

        while not board.is_game_over() and consecutive_high_score < 60:
            if turn % 2 == 0:
                should_mirror = True
                mode = 1 if random.uniform(0, 1) < temp else 0 
            else:
                should_mirror = False
                mode = 1 if random.uniform(0, 1) < temp else 0

            # --- Policy Step ---
            if not moves:
                top_moves = [(0.0, "e2e4"), (0.0, "d2d4"), (0.0, "g1f3"), (0.0, "c2c4"), (0.0, "e2e3")]
            else:
                pgn = get_pgn(moves)
                # This calls the global inference logic. 
                # Since we are in a new process, it will load its own model copy.
                top_moves = predict_next_move(pgn) 

            best_score = -float('inf')
            best_move = None

            # --- Value Step ---
            for score, move in top_moves:
                tempBoard = board.copy()
                try:
                    tempBoard.push_uci(move)
                except:
                    continue
                
                if should_mirror:
                    tempBoard = tempBoard.mirror()
                
                # Label Generation (Stockfish)
                if tempBoard.is_game_over():
                    cp = -10000 if tempBoard.is_checkmate() else 0
                else:
                    cp = eval_stockfish(tempBoard, engine)
                
                # Data Collection
                tokens = get_board_tokens(tempBoard)
                norm_cp = np.tanh(cp / 400)
                
                local_X.append(tokens)
                local_Y.append(norm_cp)

                if abs(norm_cp) >= 0.98:
                    consecutive_high_score += 1
                else: 
                    consecutive_high_score = 0

                norm_cp = -norm_cp
                
                # Decision
                if mode == 1:
                    value_cp = -evaluator.evaluate(tempBoard.fen())
                    if white_exploitative and turn % 2 == 0:              
                        weighted_score = norm_cp * 0.3 + value_cp * 0.7
                    elif not white_exploitative and turn % 2 == 1:              
                        weighted_score = norm_cp * 0.3 + value_cp * 0.7
                    else: 
                        weighted_score = value_cp
                    if weighted_score > best_score:
                        best_score = weighted_score
                        best_move = move

            if mode == 0 or best_move is None:
                best_move = get_random_move(top_moves) 
            
            board.push_uci(best_move)
            moves.append(best_move)
            turn += 1
            
            # if turn % 2 == 0 and temp > 0.1:
            #     temp = temp * 0.99

            if turn > 150: 
                print("Game is a draw")
                break;
    except Exception as e:
        print(f"Error in Game {game_id}: {e}")
    finally:
        if engine:
            engine.quit()
    
    print(f"Game {game_id} finished. Result: {board.result()}")
    return local_X, local_Y

def main():
    try:
        mp.set_start_method('spawn', force = True)
    except:
        pass

    setup_stockfish()

    # Verify it exists before starting workers
    if not os.path.exists(SF_EXECUTABLE):
        print("CRITICAL: Stockfish setup failed. Exiting.")
        return

    print(f"Starting pool with {CORES} cores...")
    final_X = []
    final_Y = []

    with mp.Pool(processes = CORES) as pool:
        results = pool.map(play_single_game, range(TOTAL_GAMES))

    for res_X, res_Y in results:
        final_X.extend(res_X)
        final_Y.extend(res_Y)

    if not final_X:
        print('No Data Collcted')
        return

    X = np.array(final_X, dtype = np.int8)
    Y = np.array(final_Y, dtype = np.float32)

    x_path = 'dataset_X_sp.npy'
    y_path = 'dataset_Y_sp.npy'

    if os.path.exists(x_path) and os.path.exists(y_path):
        print('Appending to existing datasets...')
        try:
            old_X = np.load(x_path)
            old_Y = np.load(y_path)
            X = np.concatenate((old_X, X))
            Y = np.concatenate((old_Y, Y))
        except:
            pass

    # Shuffle Data
    print("Shuffling data...")
    perm = np.random.permutation(len(X))
    X = X[perm]
    Y = Y[perm]

    np.save(x_path, X)
    np.save(y_path, Y)
    print(f"Saved {len(X)} samples to disk")

if __name__ == "__main__":
    main()