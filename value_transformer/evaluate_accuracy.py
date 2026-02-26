
import chess
import chess.pgn
import chess.engine
import os
import sys
import random
import torch
import torch.nn.functional as F
import io
import numpy as np
from tqdm import tqdm
from approximate import approximate_cp

# Add root dir to path to find packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import resources
import policy_transformer.inference as policy_infer
from value_transformer.inference import ChessEvaluator

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

def get_policy_candidates(pgn_str, board, k=7):
    """
    Get top k moves from Policy Transformer.
    Returns list of (logit, uci_move).
    """
    # Ensure resources loaded
    policy_infer.load_resources()
    
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
            uci = policy_infer.mirror_uci_string(uci)
            
        token = policy_infer.get_token(uci)
        # Use _stoi directly via module
        if policy_infer._stoi:
             token_ids.append(policy_infer._stoi.get(token, policy_infer._stoi.get("|", 0)))
        
        temp_board.push(move)
        
    # 3. Model Inference
    # Convert to tensor and add batch dim
    idx = torch.tensor([token_ids], dtype=torch.long, device=policy_infer.DEVICE)
    
    with torch.no_grad():
        # Truncate if too long (BLOCK_SIZE=256)
        if idx.shape[1] > 256: 
             idx = idx[:, -256:]
             
        logits, _ = policy_infer._model(idx)
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
            search_uci = policy_infer.mirror_uci_string(uci)
        else:
            search_uci = uci
            
        token = policy_infer.get_token(search_uci)
        
        # Look up logit
        if policy_infer._stoi and token in policy_infer._stoi:
            token_id = policy_infer._stoi[token]
            score = last_logits[token_id].item()
            candidates.append((score, uci))
        else:
            # Token not in vocab? (Should act as -inf)
            candidates.append((-float('inf'), uci))
            
    # Sort by score descending (Transformer Probabilities)
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Return top k candidates (logit, uci)
    return candidates[:k]

def evaluate_accuracy(pgn_file, num_samples=1000):
    print(f"Loading games from {pgn_file}...")
    print(f"Goal: {num_samples} random samples (1 per game)")
    
    # Initialize Value Model
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    value_model_path = os.path.join(ROOT_DIR, "value_transformer", "checkpoints", "mini_value_2o2.pt")
    
    if not os.path.exists(value_model_path):
        alt_path = os.path.join(ROOT_DIR, "value_transformer", "checkpoints", "mini_value_2o2.pt")
        if os.path.exists(alt_path):
            value_model_path = alt_path
            
    try:
        val_evaluator = ChessEvaluator(value_model_path)
    except Exception as e:
        print(f"Failed to load Value Model: {e}")
        return

    # Initialize Stockfish Persistent Instance
    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        print(f"Failed to start Stockfish: {e}")
        return

    matches = 0
    total = 0
    
    with open(pgn_file) as f:
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
                
            # Random Sampling
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
                move_san = board.san(move)
                board.push(move)
                if i % 2 == 0:
                    move_num = (i // 2) + 1
                    pgn_parts.append(f"{move_num}. {move_san}")
                else:
                    pgn_parts.append(f"{move_san}")
            
            pgn_str = " ".join(pgn_parts)
            
            # 1. Get Candidates
            try:
                candidates = get_policy_candidates(pgn_str, board, k=5) # Get top 5
                if not candidates:
                    continue
                
                # candidates is List[(logit, uci)]
                
                # 2. Evaluate Candidates with Stockfish & Value Model
                candidate_evals = []
                
                for logit, uci in candidates:
                    # Make Move
                    temp_board = board.copy()
                    move_obj = chess.Move.from_uci(uci)
                    
                    if move_obj not in board.legal_moves:
                        continue
                        
                    temp_board.push(move_obj)
                    
                    # --- Stockfish Eval ---
                    info = engine.analyse(temp_board, chess.engine.Limit(time=0.05))
                    sf_score_obj = info["score"].pov(board.turn) # Relative to player whose turn it is
                    sf_score = sf_score_obj.score(mate_score=10000)
                    
                    
                    eval_board = temp_board.copy()
                    if board.turn == chess.BLACK:
                         eval_board = eval_board.mirror()
                         
                    vm_score = val_evaluator.evaluate(eval_board.fen())
                    
                    candidate_evals.append({
                        'uci': uci,
                        'sf_score': sf_score,
                        'vm_score': vm_score
                    })
                
                if not candidate_evals:
                    continue
                    
                # 3. Find Best Moves
                # Best SF Move: Max sf_score
                best_sf_move = max(candidate_evals, key=lambda x: x['sf_score'])
                
                # Best VM Move: Max vm_score
                best_vm_move = max(candidate_evals, key=lambda x: x['vm_score'])
                
                # 4. Compare
                pred_move = best_vm_move['uci']
                target_move = best_sf_move['uci']
                
                is_match = (pred_move == target_move)
                
                if is_match:
                    matches += 1

                # DEBUG Details
                if total < 5:
                    print(f"\n--- Sample {total+1} ---")
                    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
                    print(f"Candidates:")
                    for c in candidate_evals:
                        marker = ""
                        if c['uci'] == pred_move: marker += " [VM]"
                        if c['uci'] == target_move: marker += " [SF]"
                        print(f"  {c['uci']}: SF={c['sf_score']}, VM={c['vm_score']:.3f}{marker}")
                        
                    if is_match:
                        print("RESULT: MATCH!")
                    else:
                        print("RESULT: MISS")

            except Exception as e:
                print(f"Error evaluating sample: {e}")
                # traceback.print_exc()
            
            total += 1
            pbar.update(1)
            pbar.set_postfix({
                "Acc": f"{matches/total:.2%}"
            })
            
    engine.quit() # Cleanup
            
    print(f"\nFinal Results ({total} samples):")
    print(f"Ranking Accuracy: {matches}/{total} ({matches/total*100:.2f}%)")

if __name__ == "__main__":
    random.seed(123) 
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pgn_path = os.path.join(ROOT_DIR, "games", "lichess_elite_2025-11.pgn")
    if not os.path.exists(pgn_path):
        print(f"PGN not found: {pgn_path}")
    else:
        evaluate_accuracy(pgn_path, num_samples=1000)
