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

# Add root dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import resources
import policy_transformer.inference as policy_infer
from value_transformer.inference import ChessEvaluator

# Configuration
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH")
if not STOCKFISH_PATH:
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

def get_policy_candidates(pgn_str, board, k=5):
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
        if policy_infer._stoi:
             token_ids.append(policy_infer._stoi.get(token, policy_infer._stoi.get("|", 0)))
        
        temp_board.push(move)
        
    # 3. Model Inference
    idx = torch.tensor([token_ids], dtype=torch.long, device=policy_infer.DEVICE)
    
    with torch.no_grad():
        if idx.shape[1] > 256: 
             idx = idx[:, -256:]
             
        logits, _ = policy_infer._model(idx)
        last_logits = logits[0, -1, :] 
        
    # 4. Mask Legal Moves & Sort
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return []
        
    candidates = []
    
    for move in legal_moves:
        uci = move.uci()
        if should_mirror:
            search_uci = policy_infer.mirror_uci_string(uci)
        else:
            search_uci = uci
            
        token = policy_infer.get_token(search_uci)
        
        if policy_infer._stoi and token in policy_infer._stoi:
            token_id = policy_infer._stoi[token]
            score = last_logits[token_id].item()
            candidates.append((score, uci))
        else:
            candidates.append((-float('inf'), uci))
            
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:k]

def compare_models(pgn_file, model_path_a, model_path_b, num_samples=1000):
    print(f"Comparing Value Models:\n A: {os.path.basename(model_path_a)}\n B: {os.path.basename(model_path_b)}")
    print(f"Game Source: {os.path.basename(pgn_file)}")
    
    # Load Value Models
    try:
        model_a = ChessEvaluator(model_path_a)
        model_b = ChessEvaluator(model_path_b)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Load Stockfish
    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        print(f"Failed to start Stockfish: {e}")
        return

    stats = {
        'total': 0,
        'matches_a': 0,
        'matches_b': 0
    }
    
    with open(pgn_file) as f:
        pbar = tqdm(total=num_samples, desc="Comparing")
        
        while stats['total'] < num_samples:
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
                
            if len(moves) < 7:
                 continue

            k = random.randint(5, min(len(moves)-1, 15))
            
            # Replay to point k
            board = game.board()
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
                candidates = get_policy_candidates(pgn_str, board, k=5)
                if not candidates:
                    continue
                
                candidate_evals = []
                
                for logit, uci in candidates:
                    temp_board = board.copy()
                    move_obj = chess.Move.from_uci(uci)
                    if move_obj not in board.legal_moves: continue
                    temp_board.push(move_obj)
                    
                    # Stockfish Eval
                    info = engine.analyse(temp_board, chess.engine.Limit(time=0.05))
                    sf_score = info["score"].pov(board.turn).score(mate_score=10000)
                    
                    # Value Eval Setup
                    eval_board = temp_board.copy()
                    if board.turn == chess.BLACK:
                         eval_board = eval_board.mirror()
                         
                    vm_a_score = model_a.evaluate(eval_board.fen())
                    vm_b_score = model_b.evaluate(eval_board.fen())
                    
                    candidate_evals.append({
                        'uci': uci,
                        'sf': sf_score,
                        'vm_a': vm_a_score,
                        'vm_b': vm_b_score
                    })
                
                if not candidate_evals: continue
                    
                # Find Best Moves
                best_sf = max(candidate_evals, key=lambda x: x['sf'])
                best_a = max(candidate_evals, key=lambda x: x['vm_a'])
                best_b = max(candidate_evals, key=lambda x: x['vm_b'])
                
                target_move = best_sf['uci']
                
                if best_a['uci'] == target_move: stats['matches_a'] += 1
                if best_b['uci'] == target_move: stats['matches_b'] += 1
                
                stats['total'] += 1
                pbar.update(1)
                pbar.set_postfix({
                    "Acc A": f"{stats['matches_a']/stats['total']:.2%}",
                    "Acc B": f"{stats['matches_b']/stats['total']:.2%}"
                })

            except Exception as e:
                print(f"Error: {e}")
                continue
            
    engine.quit()
    
    print(f"\nFinal Comparison ({stats['total']} samples):")
    print(f"Model A ({os.path.basename(model_path_a)}):")
    print(f"  Accuracy: {stats['matches_a']}/{stats['total']} ({stats['matches_a']/stats['total']:.2%})")
    print(f"Model B ({os.path.basename(model_path_b)}):")
    print(f"  Accuracy: {stats['matches_b']}/{stats['total']} ({stats['matches_b']/stats['total']:.2%})")

if __name__ == "__main__":
    random.seed(567)
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pgn_path = os.path.join(ROOT_DIR, "games", "lichess_elite_2025-09.pgn")
    
    # Update checkpoit names as needed
    path_a = os.path.join(ROOT_DIR, "value_transformer", "checkpoints", "mini_value_2o2.pt")
    path_b = os.path.join(ROOT_DIR, "value_transformer", "checkpoints", "mini_value_3o8.pt")
    
    if os.path.exists(pgn_path) and os.path.exists(path_a) and os.path.exists(path_b):
        compare_models(pgn_path, path_a, path_b)
    else:
        print("Missing files. Please check paths.")
