# features.py
import numpy as np
import chess

# 12 planes (6 piece types * 2 colors) * 64 = 768
# + side-to-move (1)
# + castling rights (4)
# + halfmove clock (1 scaled)
# + fullmove number (1 scaled)
# = 776 floats

PIECE_PLANES = {
    (chess.WHITE, chess.PAWN):   0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK):   3,
    (chess.WHITE, chess.QUEEN):  4,
    (chess.WHITE, chess.KING):   5,
    (chess.BLACK, chess.PAWN):   6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK):   9,
    (chess.BLACK, chess.QUEEN):  10,
    (chess.BLACK, chess.KING):   11,
}

def encode_features(board: chess.Board) -> np.ndarray:
    x = np.zeros(776, dtype=np.float32)

    # 12x64 planes (flattened in file order a1..h8)
    for color in (chess.WHITE, chess.BLACK):
        for ptype in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            idx = PIECE_PLANES[(color, ptype)]
            bb = board.pieces(ptype, color)
            # iterate over squares in the set
            for sq in bb:
                x[idx * 64 + sq] = 1.0

    base = 12 * 64

    # Side to move
    x[base + 0] = 1.0 if board.turn == chess.WHITE else 0.0

    # Castling rights (KQkq)
    x[base + 1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    x[base + 2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    x[base + 3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    x[base + 4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # Halfmove clock (scale to ~[0,1])
    x[base + 5] = min(board.halfmove_clock, 100) / 100.0

    # Fullmove number (scale to ~[0,1] assuming <= 200)
    x[base + 6] = min(board.fullmove_number, 200) / 200.0

    return x