"""Board / rights tokenization and STM mirroring for unified_mcts."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import chess
import numpy as np

PIECE_MAP = {
    None: 0,
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
}

RIGHTS_DIM = 8
CP_SCALE = 400.0


def tanh_cp(cp: float) -> float:
    return math.tanh(cp / CP_SCALE)


def stm_value_from_white_pov(white_pov: float, mirrored: bool) -> float:
    """
    Convert absolute White-POV score to STM-white label space.

    After encode_board_stm_white, the board is always White-to-move in the
    encoded view. Labels must be advantage for that STM (matches Lichess
    mega / chess-ai-model §4.2). White-POV on the unmirrored board equals
    STM iff not mirrored; if mirrored, negate.
    """
    return -white_pov if mirrored else white_pov


def board_to_piece_tokens(board: chess.Board) -> np.ndarray:
    """64 uint8 tokens, rank 7→0, file 0→7 (matches value_transformer)."""
    tokens = np.zeros(64, dtype=np.uint8)
    idx = 0
    for rank in range(7, -1, -1):
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            tokens[idx] = PIECE_MAP[piece.symbol()] if piece else 0
            idx += 1
    return tokens


def rights_vector(board: chess.Board) -> np.ndarray:
    """
    K=8 rights vector for the given board (no mirroring applied here).
    Indices: W_OO, W_OOO, B_OO, B_OOO, ep_present, ep_file, 0, 0
    """
    r = np.zeros(RIGHTS_DIM, dtype=np.uint8)
    r[0] = int(board.has_kingside_castling_rights(chess.WHITE))
    r[1] = int(board.has_queenside_castling_rights(chess.WHITE))
    r[2] = int(board.has_kingside_castling_rights(chess.BLACK))
    r[3] = int(board.has_queenside_castling_rights(chess.BLACK))
    if board.ep_square is not None:
        r[4] = 1
        r[5] = chess.square_file(board.ep_square)
    return r


def mirror_rights_for_black_stm(r: np.ndarray) -> np.ndarray:
    """After vertical color-mirror, swap white/black castling; EP file unchanged."""
    out = r.copy()
    out[0], out[2] = r[2], r[0]
    out[1], out[3] = r[3], r[1]
    return out


def encode_board_stm_white(board: chess.Board) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Return (piece_tokens, rights, mirrored) with side-to-move always White.
    If Black to move: mirror board + swap castling colors.
    """
    mirrored = board.turn == chess.BLACK
    if mirrored:
        view = board.mirror()
        pieces = board_to_piece_tokens(view)
        rights = mirror_rights_for_black_stm(rights_vector(board))
        # EP file after vertical mirror is the same file index
        if board.ep_square is not None:
            rights[4] = 1
            rights[5] = chess.square_file(board.ep_square)
        else:
            rights[4] = 0
            rights[5] = 0
    else:
        pieces = board_to_piece_tokens(board)
        rights = rights_vector(board)
    return pieces, rights, mirrored


def castle_ep_ids(rights: np.ndarray) -> Tuple[int, int]:
    castle_id = int(rights[0] + 2 * rights[1] + 4 * rights[2] + 8 * rights[3])
    ep_id = 0 if int(rights[4]) == 0 else 1 + int(rights[5])
    return castle_id, ep_id


def get_uci4_token(uci: str) -> str:
    return uci[:4]


def mirror_uci_string(uci: str) -> str:
    if not uci or len(uci) < 4:
        return uci
    new_from = str(9 - int(uci[1]))
    new_to = str(9 - int(uci[3]))
    promo = uci[4:] if len(uci) > 4 else ""
    return f"{uci[0]}{new_from}{uci[2]}{new_to}{promo}"


def pad_history(token_ids: list, block_size: int, pad_id: int) -> np.ndarray:
    """Left-pad to block_size."""
    if len(token_ids) > block_size:
        token_ids = token_ids[-block_size:]
    out = np.full(block_size, pad_id, dtype=np.uint16)
    if token_ids:
        out[-len(token_ids) :] = np.asarray(token_ids, dtype=np.uint16)
    return out
