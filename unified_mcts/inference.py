"""Inference helpers for JointChessNet (policy priors + value)."""
from __future__ import annotations

import io
import os
import pickle
import sys
from typing import List, Optional, Tuple

import chess
import chess.pgn
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_utils import (
    encode_board_stm_white,
    get_uci4_token,
    mirror_uci_string,
    pad_history,
)
from model import JointChessNet


class UnifiedInference:
    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu" and torch.backends.mps.is_available():
                device = "mps"
        self.device = device

        with open(vocab_path, "rb") as f:
            self.stoi = pickle.load(f)
        self.itos = {i: s for s, i in self.stoi.items()}
        self.pad_id = int(self.stoi["|"])
        vocab_size = len(self.stoi)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        self.model = JointChessNet(
            vocab_size=cfg.get("vocab_size", vocab_size),
            block_size=cfg.get("block_size", 128),
            d_b=cfg.get("d_b", 384),
            n_head_b=cfg.get("n_head_b", 6),
            n_layer_b=cfg.get("n_layer_b", 6),
            d=cfg.get("d", 500),
            n_head=cfg.get("n_head", 10),
            n_layer=cfg.get("n_layer", 8),
            d_c=cfg.get("d_c", cfg.get("d", 500)),
            dropout=0.0,
        ).to(device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.block_size = self.model.block_size

    def _encode_position(self, board: chess.Board, moves: List[chess.Move]):
        """Encode current board + move history for STM-white inference."""
        should_mirror = board.turn == chess.BLACK
        hist_ids = []
        for m in moves:
            uci = m.uci()
            tok = get_uci4_token(mirror_uci_string(uci) if should_mirror else uci)
            hist_ids.append(self.stoi.get(tok, self.pad_id))
        pieces, rights, _ = encode_board_stm_white(board)
        hist = pad_history(hist_ids, self.block_size, self.pad_id)
        return hist, pieces, rights, should_mirror

    def _history_and_board_from_pgn(self, pgn_string: str):
        pgn_io = io.StringIO(pgn_string)
        game = chess.pgn.read_game(pgn_io)
        board = chess.Board()
        moves: List[chess.Move] = []
        if game is not None:
            for m in game.mainline_moves():
                moves.append(m)
                board.push(m)
        hist, pieces, rights, should_mirror = self._encode_position(board, moves)
        return board, hist, pieces, rights, should_mirror

    @torch.no_grad()
    def predict_policy_board(
        self,
        board: chess.Board,
        moves: Optional[List[chess.Move]] = None,
        top_k: int = 10,
    ) -> List[Tuple[float, str]]:
        """Return [(logit, uci), ...] legal-masked from a live board + move list."""
        moves = moves or []
        legal = list(board.legal_moves)
        if not legal:
            return []
        hist, pieces, rights, should_mirror = self._encode_position(board, moves)
        hist_t = torch.tensor([hist], dtype=torch.long, device=self.device)
        board_t = torch.tensor([pieces], dtype=torch.long, device=self.device)
        rights_t = torch.tensor([rights], dtype=torch.long, device=self.device)
        out = self.model(hist_t, board_t, rights_t, compute_policy=True, compute_value=False)
        logits = out["logits"][0]

        cands = []
        for move in legal:
            uci = move.uci()
            search = mirror_uci_string(uci) if should_mirror else uci
            tok = get_uci4_token(search)
            if tok in self.stoi:
                score = float(logits[self.stoi[tok]].item())
            else:
                score = float("-inf")
            cands.append((score, uci))
        cands.sort(key=lambda x: x[0], reverse=True)
        return cands[:top_k]

    @torch.no_grad()
    def predict_policy(self, pgn_string: str, top_k: int = 10) -> List[Tuple[float, str]]:
        """Return [(logit, uci), ...] legal-masked, sorted descending — MCTS policy_func shape."""
        board, hist, pieces, rights, should_mirror = self._history_and_board_from_pgn(pgn_string)
        # Reuse board path
        moves = []
        temp = chess.Board()
        pgn_io = io.StringIO(pgn_string)
        game = chess.pgn.read_game(pgn_io)
        if game is not None:
            for m in game.mainline_moves():
                moves.append(m)
                temp.push(m)
        return self.predict_policy_board(board, moves, top_k=top_k)

    @torch.no_grad()
    def evaluate_board(self, board: chess.Board) -> float:
        pieces, rights, _ = encode_board_stm_white(board)
        hist = pad_history([], self.block_size, self.pad_id)
        hist_t = torch.tensor([hist], dtype=torch.long, device=self.device)
        board_t = torch.tensor([pieces], dtype=torch.long, device=self.device)
        rights_t = torch.tensor([rights], dtype=torch.long, device=self.device)
        out = self.model(hist_t, board_t, rights_t, compute_policy=False, compute_value=True)
        return float(out["v"][0].item())

    def evaluate_fen(self, fen: str) -> float:
        return self.evaluate_board(chess.Board(fen))

    @torch.no_grad()
    def choose_move(
        self,
        board: chess.Board,
        moves: Optional[List[chess.Move]] = None,
        top_k: int = 5,
        temperature: float = 0.0,
        dirichlet_alpha: float = 0.25,
        dirichlet_eps: float = 0.25,
    ) -> Optional[chess.Move]:
        """Sample or argmax a legal move from policy logits."""
        cands = self.predict_policy_board(board, moves, top_k=top_k)
        if not cands:
            return None
        ucis = [u for _, u in cands]
        logits = np.array([s for s, _ in cands], dtype=np.float64)
        if temperature <= 0:
            return chess.Move.from_uci(ucis[int(np.argmax(logits))])
        logits = logits / max(temperature, 1e-6)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / (probs.sum() + 1e-12)
        if dirichlet_eps > 0 and len(probs) > 1:
            noise = np.random.dirichlet([dirichlet_alpha] * len(probs))
            probs = (1 - dirichlet_eps) * probs + dirichlet_eps * noise
            probs = probs / (probs.sum() + 1e-12)
        pick = np.random.choice(ucis, p=probs)
        return chess.Move.from_uci(pick)
