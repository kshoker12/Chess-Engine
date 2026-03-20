import chess
import chess.pgn
import chess.engine
import os
import sys
import random
import torch
import io
import pickle
from tqdm import tqdm

# Add root dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model definition
from policy_transformer.model import PolicyHead
import policy_transformer.inference as infer_module

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

# Model Constants (Must match training)
N_EMBD = 768
N_HEAD = 12
N_LAYER = 10
BLOCK_SIZE = 256
DROPOUT = 0.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu' and torch.backends.mps.is_available():
    DEVICE = 'mps'

def get_stockfish_best_move(board, time_limit=0.1):
    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            result = engine.play(board, chess.engine.Limit(time=time_limit))
            return result.move.uci() if result.move else None
    except Exception as e:
        print(f"Stockfish error: {e}")
        return None

def load_model(checkpoint_path, vocab_size):
    print(f"Loading model from {checkpoint_path}...")
    if 'mega' in checkpoint_path:
        model = PolicyHead(
            vocab_size=vocab_size,
            n_embd=N_EMBD,
            block_size=BLOCK_SIZE,
            n_head=N_HEAD,
            n_layer=N_LAYER,
            dropout=DROPOUT,
            device=DEVICE
        )
    else: 
        model = PolicyHead(
            vocab_size=vocab_size,
            n_embd=500,
            block_size=128,
            n_head=10,
            n_layer=8,
            dropout=DROPOUT,
            device=DEVICE
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    # Strip _orig_mod. prefix if checkpoint was saved from torch.compile() model
    if isinstance(state_dict, dict) and any(k.startswith('_orig_mod.') for k in state_dict):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
        
    model.to(DEVICE)
    model.eval()
    return model

def predict_with_model(model, pgn_str, board, vocab, stoi, k=1):
    # Reuse inference logic but with passed model
    
    # 1. Parse PGN to get moves
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
    
    # We need a fresh replay to get tokens correctly
    # infer_module.get_token is simple UCI tokenization
    # But we need access to 'mirror_uci_string'
    
    temp_board = chess.Board()
    for move in moves_to_process:
        uci = move.uci()
        if should_mirror:
            uci = infer_module.mirror_uci_string(uci)
            
        token = infer_module.get_token(uci)
        token_ids.append(stoi.get(token, stoi.get("|", 0)))
        temp_board.push(move)
        
    # 3. Model Inference
    idx = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        if idx.shape[1] > 256: 
             idx = idx[:, -256:]
             
        logits, _ = model(idx)
        last_logits = logits[0, -1, :]
        
    # 4. Mask Legal Moves
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return []
        
    candidates = []
    
    for move in legal_moves:
        uci = move.uci()
        if should_mirror:
            search_uci = infer_module.mirror_uci_string(uci)
        else:
            search_uci = uci
            
        token = infer_module.get_token(search_uci)
        
        if token in stoi:
            token_id = stoi[token]
            score = last_logits[token_id].item()
            candidates.append((score, uci))
        else:
            candidates.append((-float('inf'), uci))
            
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in candidates[:k]]

def compare_models(pgn_file, model_path_a, model_path_b, num_samples=1000):
    # Load Vocab
    vocab_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "policy_transformer", "vocab.pkl")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    stoi = vocab
    
    # Load Models
    model_a = load_model(model_path_a, len(vocab))
    model_b = load_model(model_path_b, len(vocab))
    
    name_a = os.path.basename(model_path_a)
    name_b = os.path.basename(model_path_b)
    
    print(f"\nComparing:\n A: {name_a}\n B: {name_b}\n")
    
    stats = {
        'A_top1': 0, 'A_top7': 0,
        'B_top1': 0, 'B_top7': 0,
        'total': 0
    }
    
    with open(pgn_file, 'rb') as f: # Open as binary for seek/tell safety if needed, but text is fine usually. using 'r' and seek(0,2) works.
        pass
    
    # Re-open in text mode
    with open(pgn_file, 'r', errors='replace') as f:
        f.seek(0, 2)
        file_size = f.tell()
        
        pbar = tqdm(total=num_samples, desc="Comparing")
        
        while stats['total'] < num_samples:
            # Random Seek (Copied logic)
            random_pos = random.randint(0, file_size)
            f.seek(random_pos)
            f.readline() # Skip partial
            
            game_start_found = False
            # Try to find next [Event ...
            # Limit search to avoid infinite loop
            for _ in range(1000): 
                line = f.readline()
                if not line: break
                if line.startswith("[Event"):
                    # Seek back
                    try:
                        f.seek(f.tell() - len(line.encode('utf-8'))) # approx
                        game_start_found = True
                    except:
                        pass
                    break
            
            if not game_start_found:
                continue
                
            try:
                game = chess.pgn.read_game(f)
            except ValueError:
                continue
            
            if game is None: continue
            
            moves = list(game.mainline_moves())
            if len(moves) < 7: continue
            
            # Sample position
            k = random.randint(max(5, len(moves) - 15), len(moves) - 1)
            
            # Reconstruct PGN str and Board
            board = game.board()
            pgn_parts = []
            for i in range(k):
                move = moves[i]
                move_san = board.san(move)
                board.push(move)
                if i % 2 == 0:
                    pgn_parts.append(f"{(i//2)+1}. {move_san}")
                else:
                    pgn_parts.append(f"{move_san}")
            
            pgn_str = " ".join(pgn_parts)
            
            # Ground Truth (Stockfish) - NOT the played move?
            # User said: "evaluate_accuracy but such that both models are evaluated on same random data"
            # evaluate_accuracy uses Stockfish as GT.
            gt_move = get_stockfish_best_move(board)
            if not gt_move: continue
            
            # Predictions
            preds_a = predict_with_model(model_a, pgn_str, board, vocab, stoi, k=7)
            preds_b = predict_with_model(model_b, pgn_str, board, vocab, stoi, k=7)
            
            if not preds_a or not preds_b: continue
            
            # Update Stats
            if preds_a[0] == gt_move: stats['A_top1'] += 1
            if gt_move in preds_a: stats['A_top7'] += 1
            
            if preds_b[0] == gt_move: stats['B_top1'] += 1
            if gt_move in preds_b: stats['B_top7'] += 1
            
            stats['total'] += 1
            pbar.update(1)
            pbar.set_postfix({
                f"A_1": f"{stats['A_top1']/stats['total']:.2%}",
                f"B_1": f"{stats['B_top1']/stats['total']:.2%}"
            })

    print("\nFinal Results:")
    print(f"Total Samples: {stats['total']}")
    print(f"Model A ({name_a}):")
    print(f"  Top-1: {stats['A_top1']}/{stats['total']} ({stats['A_top1']/stats['total']:.2%})")
    print(f"  Top-7: {stats['A_top7']}/{stats['total']} ({stats['A_top7']/stats['total']:.2%})")
    
    print(f"Model B ({name_b}):")
    print(f"  Top-1: {stats['B_top1']}/{stats['total']} ({stats['B_top1']/stats['total']:.2%})")
    print(f"  Top-7: {stats['B_top7']}/{stats['total']} ({stats['B_top7']/stats['total']:.2%})")

if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pgn_path = os.path.join(ROOT_DIR, "data", "weaker", "lichess_db_standard_rated_2013-09.pgn")
    
    model_dir = os.path.join(os.path.dirname(__file__))
    path_a = os.path.join(model_dir, "ultra_2o5.pt")
    path_b = os.path.join(model_dir, "checkpoints", "ultra_3o9.pt")
    
    if os.path.exists(pgn_path) and os.path.exists(path_a) and os.path.exists(path_b):
        compare_models(pgn_path, path_a, path_b)
    else:
        print("Missing files. Check paths.")
        print(f"PGN: {pgn_path} - {os.path.exists(pgn_path)}")
        print(f"Model A: {path_a} - {os.path.exists(path_a)}")
        print(f"Model B: {path_b} - {os.path.exists(path_b)}")
