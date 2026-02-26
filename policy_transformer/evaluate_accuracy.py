
import chess
import chess.pgn
import chess.engine
import os
import sys
import random
import torch
import io
from tqdm import tqdm

# Add root dir to path to find packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import resources ONLY from inference.py
import policy_transformer.inference as infer_module

# Configuration
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH")
if not STOCKFISH_PATH:
    # Try common paths
    paths = [
        '/opt/homebrew/bin/stockfish',
        '/usr/local/bin/stockfish',
        '/usr/bin/stockfish',
        'stockfish'
    ]
    for p in paths:
        if os.path.exists(p) or p == 'stockfish':
            STOCKFISH_PATH = p
            break

def get_stockfish_best_move(board, time_limit=0.1):
    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            result = engine.play(board, chess.engine.Limit(time=time_limit))
            return result.move.uci() if result.move else None
    except Exception as e:
        print(f"Stockfish error: {e}")
        return None

def predict_top_k_custom(pgn_str, board, k=3):
    # Ensure resources loaded
    infer_module.load_resources()
    
    # 1. Parse PG to get moves
    pgn_io = io.StringIO(pgn_str)
    try:
        game = chess.pgn.read_game(pgn_io)
    except:
        game = None
        return []
        
    moves_to_process = []
    if game:
        moves_to_process = list(game.mainline_moves())
    
    # 2. Replay for tokens
    should_mirror = (board.turn == chess.BLACK)
    
    token_ids = []
    temp_board = chess.Board()
    for move in moves_to_process:
        uci = move.uci()
        if should_mirror:
            uci = infer_module.mirror_uci_string(uci)
            
        token = infer_module.get_token(uci)
        # Use _stoi directly via module
        if infer_module._stoi:
             token_ids.append(infer_module._stoi.get(token, infer_module._stoi.get("|", 0)))
        
        temp_board.push(move)
        
    # 3. Model Inference
    # Convert to tensor and add batch dim
    idx = torch.tensor([token_ids], dtype=torch.long, device=infer_module.DEVICE)
    
    with torch.no_grad():
        # Truncate if too long (BLOCK_SIZE=256)
        if idx.shape[1] > 256: 
             idx = idx[:, -256:]
             
        logits, _ = infer_module._model(idx)
        # Get last logits [Batch, VocabSize] -> [1, V]
        last_logits = logits[0, -1, :] # Shape [V]
        
    # 4. Mask Legal Moves & Sort
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return []
        
    candidates = []
    
    for move in legal_moves:
        uci = move.uci()
        
        # Apply parity mirroring for lookup
        if should_mirror:
            search_uci = infer_module.mirror_uci_string(uci)
        else:
            search_uci = uci
            
        token = infer_module.get_token(search_uci)
        
        # Look up logit
        if infer_module._stoi and token in infer_module._stoi:
            token_id = infer_module._stoi[token]
            score = last_logits[token_id].item()
            candidates.append((score, uci))
        else:
            # Token not in vocab? (Should act as -inf)
            candidates.append((-float('inf'), uci))
            
    # Sort by score descending (Transformer Probabilities)
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Return top K move strings
    return [c[1] for c in candidates[:k]]

def evaluate_accuracy(pgn_file, num_samples=1000):
    print(f"Loading games from {pgn_file}...")
    print(f"Goal: {num_samples} random samples (1 per game)")
    
    matches = 0
    top7_matches = 0 # Top-7 Recall
    total = 0
    
    with open(pgn_file) as f:
        # Use tqdm to track progress
        pbar = tqdm(total=num_samples, desc="Evaluating")
        
        while total < num_samples:
            try:
                game = chess.pgn.read_game(f)
            except ValueError:
                continue
                
            if game is None:
                print("End of PGN file reached.")
                break
                
            moves = list(game.mainline_moves())
            if not moves:
                continue
                
            # Random Sampling: Pick ONE random split point in this game
            # We want to predict the move at index `k`.
            
            # Constraint: We need some history for the Transformer to have context.
            if len(moves) < 7:
                 continue

            k = random.randint(5, len(moves) - 1)
            if k < 0: continue 

            # Replay to point k
            board = game.board()
            
            # Efficient replay with SAN generation
            pgn_parts = []
            
            for i in range(k):
                move = moves[i]
                
                # Generate SAN before pushing
                move_san = board.san(move)
                board.push(move)
                
                if i % 2 == 0:
                    move_num = (i // 2) + 1
                    pgn_parts.append(f"{move_num}. {move_san}")
                else:
                    pgn_parts.append(f"{move_san}")
            
            pgn_str = " ".join(pgn_parts)
            
            # 1. Ground Truth (Stockfish) on current board
            gf_move = get_stockfish_best_move(board)
            
            if not gf_move:
                continue
            
            # 2. Prediction (Using Custom Logic)
            try:
                # Get Top 7 candidates from Transformer directly
                candidates = predict_top_k_custom(pgn_str, board, k=9)
                pred_move = candidates[0] if candidates else None
                
                # DEBUG: Print details for first few samples to diagnose
                if total < 5:
                    print(f"\n--- Sample {total+1} ---")
                    print(f"PGN: {pgn_str}")
                    print(f"Turn: {'Black' if board.turn == chess.BLACK else 'White'}")
                    print(f"Stockfish: {gf_move}")
                    print(f"Candidates (Top 7): {candidates}")
                    if candidates and gf_move in candidates:
                         print("MATCH FOUND in Top 7")
                    else:
                         print("NO MATCH in Top 7")
                    print("----------------------")

            except Exception as e:
                print(f"Error making prediction: {e}")
                pred_move = None
                candidates = []
                
            # 3. Compare
            is_match = (pred_move == gf_move)
            if is_match:
                matches += 1
                
            if gf_move in candidates:
                top7_matches += 1
            
            total += 1
            pbar.update(1)
            pbar.set_postfix({
                "Top-1": f"{matches/total:.2%}",
                "Top-7": f"{top7_matches/total:.2%}"
            })
            
    print(f"\nFinal Results ({total} samples):")
    print(f"Top-1 Accuracy: {matches}/{total} ({matches/total*100:.2f}%)")
    print(f"Top-7 Recall:   {top7_matches}/{total} ({top7_matches/total*100:.2f}%)")

if __name__ == "__main__":
    random.seed(123) 
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pgn_path = os.path.join(ROOT_DIR, "weaker", "lichess_db_standard_rated_2013-09.pgn")
    if not os.path.exists(pgn_path):
        print(f"PGN not found: {pgn_path}")
    else:
        evaluate_accuracy(pgn_path, num_samples=1000)
