import torch
import chess
import numpy as np
import os
import sys
import io
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from value_transformer.model import ChessFormer  # Imports your architecture

class ChessEvaluator:
    def __init__(self, model_path, device=None):
        # 1. Device Setup
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if self.device == 'cpu' and torch.backends.mps.is_available():
                self.device = 'mps'
        else:
            self.device = device
            
        print(f"Loading Value Model on {self.device}...")

        # 2. Initialize Model Architecture
        # MUST match your training config (30M params)
        if '1o3' in model_path: 
            self.model = ChessFormer(
                block_size=64,
                n_embed=390,
                n_heads=10,
                n_layers=10, 
                dropout=0.0 # No dropout during inference
            )
        else:
            self.model = ChessFormer(
                block_size=64,
                n_embed=384,
                n_heads=6,
                n_layers=6,
                dropout=0.0 # No dropout during inference
            )
        self.model = torch.compile(self.model, mode="default", dynamic=True)

        # 3. Load Weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        # Handle loading on CPU if trained on GPU
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval() # Critical: Disables dropout logic
        
        # 4. Define Mappings (Same as training)
        self.piece_map = {
            '.': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
        }
        print("Model Loaded Successfully.")

    def _get_tokens(self, board):
        """Convert board object to 64 tokens."""
        tokens = []
        for rank in range(7, -1, -1):
            for file in range(8):
                piece = board.piece_at(chess.square(file, rank))
                tokens.append(self.piece_map[piece.symbol()] if piece else 0)
        return np.array(tokens, dtype=np.int64)

    def _get_fen(self, pgn):
        pgn_io = io.StringIO(pgn)
        game = chess.pgn.read_game(pgn_io)
        board = game.end().board()
        return board.fen()

    def evaluate(self, fen):
        """
        Input: FEN string
        Output: Centipawn evaluation (float) from White's perspective.
                (+ means White is winning, - means Black is winning)
        """
        board = chess.Board(fen)
        
        # B. Tokenize
        tokens = self._get_tokens(board)
        
        # C. Prepare Tensor
        x = torch.from_numpy(tokens).unsqueeze(0).to(self.device) # Add batch dim (1, 64)
        
        # D. Inference
        with torch.no_grad():
            # Model outputs value between -1 and 1
            normalized_score = self.model(x).item()
            
        # E. De-Normalize (x 1500)
        cp_score = normalized_score
        return cp_score

    def batch_evaluate(self, fens: list) -> list:
        """
        Input: List of FEN strings
        Output: List of raw scores (floats) from White's perspective.
        """
        if not fens:
             return []
        
        all_tokens = []
        for fen in fens:
            board = chess.Board(fen)
            tokens = self._get_tokens(board)
            all_tokens.append(tokens)
            
        # Create tensor (B, 64)
        x = torch.from_numpy(np.array(all_tokens)).to(self.device)
        
        with torch.no_grad():
             # Model outputs value between -1 and 1
             # Output shape: (B, 1) -> squeeze -> (B,)
             scores = self.model(x).squeeze(1).cpu().numpy().tolist()
             
        return scores