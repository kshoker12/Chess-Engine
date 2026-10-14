"""
Thin adapter: build policy_func / value_func for engine.mcts.MCTS without editing engine/.

MCTS contract (see engine/mcts.py):
  - policy_func(pgn, top_k) -> [(logit, uci), ...]
  - value_func(fens: list[str]) -> list[float]
    FENs are *already* STM-white mirrored by MCTS before the call.
    Return STM advantage in (-1, 1). MCTS negates for the parent POV.

Usage:
  from unified_mcts.mcts_adapter import make_mcts_funcs
  from unified_mcts.inference import UnifiedInference
  inf = UnifiedInference(ckpt, vocab)
  policy_func, value_func = make_mcts_funcs(inf)
  mcts = MCTS(pgn, value_func, policy_func, num_simulations=400)
"""
from __future__ import annotations

import chess
import chess.pgn
import io
from typing import List, Tuple

from unified_mcts.inference import UnifiedInference


def make_mcts_funcs(inf: UnifiedInference):
    def policy_func(pgn: str, top_k: int = 10) -> List[Tuple[float, str]]:
        return inf.predict_policy(pgn, top_k=top_k)

    def value_func(fens: List[str]) -> List[float]:
        # Do not re-mirror: caller already mirrored Black-to-move positions.
        # evaluate_board only mirrors when turn==BLACK, so White-STM FENs pass through.
        return [inf.evaluate_board(chess.Board(fen)) for fen in fens]

    return policy_func, value_func


def board_from_pgn(pgn: str) -> chess.Board:
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return chess.Board()
    board = game.board()
    for m in game.mainline_moves():
        board.push(m)
    return board
