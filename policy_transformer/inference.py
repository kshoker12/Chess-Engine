import torch
import torch.nn.functional as F
import chess
import chess.pgn 
import chess.engine
import io
import os
import pickle
import sys

# Ensure we can import the model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PolicyHead

# --- Configuration ---
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "ultra_3o9.pt")
VOCAB_PATH = os.path.join(os.path.dirname(__file__), "vocab.pkl")

# Hyperparameters (Must match training)
N_EMBD = 500
N_HEAD = 10
N_LAYER = 8
BLOCK_SIZE = 128
DROPOUT = 0.0 # No dropout during inference
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu' and torch.backends.mps.is_available():
    DEVICE = 'mps'

# --- Global State (Lazy Load) ---
_model = None
_vocab = None
_itos = None
_stoi = None

def load_resources():
    global _model, _vocab, _itos, _stoi
    if _model is not None:
        return

    print(f"Loading vocab from {VOCAB_PATH}...")
    with open(VOCAB_PATH, 'rb') as f:
        _vocab = pickle.load(f)
    
    _stoi = _vocab
    _itos = {i: s for s, i in _vocab.items()}
    vocab_size = len(_vocab)

    print(f"Loading model from {CHECKPOINT_PATH}...")
    _model = PolicyHead(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        block_size=BLOCK_SIZE,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
        device=DEVICE
    )
    # _model = torch.compile(_model, mode="default", dynamic=True)
    
    # Load weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    # Handle if checkpoint is wrapped in dict (e.g. {'model': ...}) or just state_dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    # Strip _orig_mod. prefix if checkpoint was saved from torch.compile() model
    if isinstance(state_dict, dict) and any(k.startswith('_orig_mod.') for k in state_dict):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    _model.load_state_dict(state_dict)
        
    _model.to(DEVICE)
    _model.eval()
    print("Model loaded successfully.")

# --- Helper Logic (Duplicated from preprocess.py for independence) ---
def mirror_uci_string(uci):
    # 'e2e4' -> 'e7e5' (flipped ranks)
    if not uci: return ""
    new_from = str(9 - int(uci[1]))
    new_to = str(9 - int(uci[3]))
    promo = uci[4:]
    return f"{uci[0]}{new_from}{uci[2]}{new_to}{promo}"

def get_token(uci_str):
    """
    Simple UCI tokenization: first 4 chars.
    e.g. "e2e4q" -> "e2e4"
    """
    return uci_str[:4]


# --- Main Inference Function ---
def predict_next_move(pgn_string, top_k = 9):
    # print("Utilizing Transformer for move prediction")
    """
    Takes a PGN string (e.g. "1. e4 e5 ..."), 
    parses it, handles parity (mirroring if Black to move),
    runs model, returns best UCI move using the policy.
    """
    load_resources()
    
    # 1. Parse Game
    pgn_io = io.StringIO(pgn_string)
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        # Empty game? Start position.
        board = chess.Board()
        # "1." is implicitly White to move.
    else:
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            
    # 2. Determine Turn & Mirroring
    should_mirror = (board.turn == chess.BLACK)
    
    # Replay for tokens
    token_ids = []
    
    moves_to_process = []
    if game:
        moves_to_process = list(game.mainline_moves())
    
    if len(moves_to_process) == 0:
        temp_board = chess.Board()
        token_ids = [0]
        # return [(0.0, "e2e4"), (0.0, "d2d4"), (0.0, "g1f3"), (0.0, "c2c4"), (0.0, "e2e3")]
    else: 
        # Replay to generate tokens
        # Replay to generate tokens
        temp_board = chess.Board()
        for move in moves_to_process:
            # piece = temp_board.piece_at(move.from_square) (Not needed for simple UCI)
            uci = move.uci()
            
            if should_mirror:
                uci = mirror_uci_string(uci)
                
            token = get_token(uci)
            if token in _stoi:
                token_ids.append(_stoi[token])
            else:
                token_ids.append(_stoi.get("|", 0))
            
            temp_board.push(move)

    # Convert to tensor and add batch dim
    idx = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    
    # 5. Generate Logits
    # We only need 1 new token
    with torch.no_grad():
        # Truncate if too long (BLOCK_SIZE)
        if idx.shape[1] > BLOCK_SIZE:
             idx = idx[:, -BLOCK_SIZE:]
             
        logits, _ = _model(idx)
        # Get last logits [Batch, VocabSize] -> [1, V]
        last_logits = logits[0, -1, :] # Shape [V]
        
    # 6. Legal Move Masking & Selection
    # Strategy: Iterate all legal moves, convert to tokens, get their logits, pick best.
    
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None # Checkmate or Stalemate
        
    move_candidates = []
    
    for move in legal_moves:
        uci = move.uci()
        
        # Apply parity mirroring if needed (Black to move -> Mirror to match White-only model)
        if should_mirror:
            search_uci = mirror_uci_string(uci)
        else:
            search_uci = uci
            
        token = get_token(search_uci)
        
        # Look up logit
        if token in _stoi:
            token_id = _stoi[token]
            score = last_logits[token_id].item()
            move_candidates.append((score, uci))
        else:
            # Token not in vocab? (Should act as -inf)
            move_candidates.append((-float('inf'), uci))
            
    # Sort by score descending (Transformer Probabilities)
    move_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # "Pick the best 5 from them but return the best one"
    top_5 = move_candidates[:top_k]
    return top_5 