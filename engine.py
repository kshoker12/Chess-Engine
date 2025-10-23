

from typing import Tuple, Optional, List
import math
from dataclasses import dataclass
import time 
import chess

# Lazy imports - only load when needed
def _get_eval_model():
    from eval_model import eval_cp
    return eval_cp

def _get_features():
    from features import encode_features
    return encode_features

INF = 10_000

@dataclass 
class EngineConfig: 
    max_depth: int = 5
    time_budget_ms: int = 900
    node_limit: int = 150_000
    blend_nn_with_material: float = 0.7
    enable_quiescence: bool = True
    use_null_move: bool = True           # simple null-move pruning
    null_move_R: int = 2                 # reduction for null move
    lmr_min_depth: int = 3               # start Late Move Reductions after this depth
    lmr_move_idx_threshold: int = 4      # reduce after the first N moves
    pv_max_len: int = 16                 # cap PV length to keep it tidy

# Transposition Table Entry
class TTEntry:
    __slots__ = ('z', 'depth', 'value', 'flag', 'move')

    def __init__(self, z: int, depth: int, value: int, flag: int, move: Optional[chess.Move] = None):
        self.z = z
        self.depth = depth
        self.value = value
        self.flag = flag
        self.move = move

class TranspositionTable:
    def __init__(self, size: int = 1_000_003):
        self.size = size
        self.table: List[Optional[TTEntry]] = [None] * size

    def probe(self, z: int, depth: int, alpha: int, beta: int):
        e = self.table[z % self.size]
        if e and e.z == z and e.depth >= depth:
            if e.flag == 0:  # EXACT
                return e.value, e.move, True
            if e.flag == 1 and e.value >= beta:    # LOWERBOUND (fail-high)
                return e.value, e.move, True
            if e.flag == 2 and e.value <= alpha:   # UPPERBOUND (fail-low)
                return e.value, e.move, True
        return None, None, False

    def store(self, z: int, depth: int, val: int, move: Optional[chess.Move], alpha: int, beta: int):
        if val <= alpha:
            flag = 2  # UPPERBOUND
        elif val >= beta:
            flag = 1  # LOWERBOUND
        else:
            flag = 0  # EXACT
        self.table[z % self.size] = TTEntry(z, depth, val, flag, move)

PIECE_VAL = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900
}

# Piece-square tables for better positional evaluation
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

KING_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

def material_eval(board) -> int:
    s = 0
    
    # Material evaluation
    for p, v in PIECE_VAL.items():
        s += v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK)))
    
    # Piece-square table evaluation
    piece_tables = {
        chess.PAWN: PAWN_TABLE,
        chess.KNIGHT: KNIGHT_TABLE,
        chess.BISHOP: BISHOP_TABLE,
        chess.ROOK: ROOK_TABLE,
        chess.QUEEN: QUEEN_TABLE,
        chess.KING: KING_TABLE
    }
    
    for piece_type, table in piece_tables.items():
        # White pieces
        for square in board.pieces(piece_type, chess.WHITE):
            s += table[square]
        
        # Black pieces (flip the table)
        for square in board.pieces(piece_type, chess.BLACK):
            s -= table[chess.square_mirror(square)]
    
    # score is always from side-to-move perspective
    return s if board.turn == chess.WHITE else -s

def static_eval(board, cfg: EngineConfig) -> int:
    # blend NN eval with material for stability
    encode_features = _get_features()
    eval_cp = _get_eval_model()
    
    f = encode_features(board)
    ml = eval_cp(f)
    mat = material_eval(board)
    cp = cfg.blend_nn_with_material * ml + (1.0 - cfg.blend_nn_with_material) * mat
    return int(round(max(min(cp, INF), -INF)))

# MVV-LVA-ish capture ordering plus TT move bonus
def ordered_moves(board, tt_move, killer_moves: List[Optional[chess.Move]], history: dict, depth: int):
    moves = list(board.legal_moves)
    
    def score(m):
        # 1. Transposition table move (highest priority)
        if tt_move and m == tt_move:
            return 10_000_000
        
        # 2. Captures (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
        if board.is_capture(m):
            victim = board.piece_type_at(m.to_square) or 0
            attacker = board.piece_type_at(m.from_square) or 0
            return 1_000_000 + victim * 10 - attacker
        
        # 3. Killer moves
        for killer in killer_moves:
            if killer and m == killer:
                return 500_000
        
        # 4. Promotions
        if m.promotion:
            return 100_000 + (m.promotion * 1000)
        
        # 5. History heuristic
        history_score = history.get(m, 0)
        if history_score > 0:
            return history_score
        
        # 6. Checks (more expensive, so lower priority)
        if board.gives_check(m):
            return 10_000
        
        return 0
    
    moves.sort(key=score, reverse=True)
    return moves


class Search:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.tt = TranspositionTable()
        self.nodes = 0
        self.start_time = 0.0
        self.stop = False
        self.best_pv: List[chess.Move] = []
        
        # Killer moves: moves that caused beta cutoffs at each depth
        self.killer_moves = [[None, None] for _ in range(64)]  # 2 killer moves per depth
        
        # History heuristic: track how often moves cause cutoffs
        self.history = {}  # move -> count of cutoffs

    def time_up(self) -> bool:
        if self.cfg.time_budget_ms <= 0:
            return False
        return (time.time() - self.start_time) * 1000.0 >= self.cfg.time_budget_ms

    # Quiescence: search only captures and checks to stabilize leaf evals
    def quiesce(self, board, alpha: int, beta: int) -> int:
        stand_pat = static_eval(board, self.cfg)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        # Get moves ordered by MVV-LVA, but include checks too
        moves = list(board.legal_moves)
        def qscore(m):
            if board.is_capture(m):
                victim = board.piece_type_at(m.to_square) or 0
                attacker = board.piece_type_at(m.from_square) or 0
                return 1_000_000 + victim * 10 - attacker
            elif board.gives_check(m):
                return 100_000  # Checks are important in quiescence
            return 0
        
        moves.sort(key=qscore, reverse=True)
        
        for m in moves:
            if not (board.is_capture(m) or board.gives_check(m)):
                continue
                
            board.push(m)
            self.nodes += 1
            score = -self.quiesce(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def alphabeta(self, board, depth: int, alpha: int, beta: int, pv) -> int:
        if self.stop or self.nodes >= self.cfg.node_limit or self.time_up():
            self.stop = True
            return static_eval(board, self.cfg)

        z = hash(board._transposition_key())
        val, tt_move, hit = self.tt.probe(z, depth, alpha, beta)
        if hit:
            return val

        if depth == 0:
            if self.cfg.enable_quiescence:
                return self.quiesce(board, alpha, beta)
            return static_eval(board, self.cfg)

        # Terminal nodes
        if board.is_checkmate():
            return -INF + int(board.fullmove_number)  # prefer faster mates
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or board.can_claim_threefold_repetition():
            return 0

        # Null-move pruning (optional, avoid when in check or low depth)
        if self.cfg.use_null_move and depth >= 3 and not board.is_check():
            board.push(chess.Move.null())
            self.nodes += 1
            score = -self.alphabeta(board, depth - 1 - self.cfg.null_move_R, -beta, -beta + 1, [])
            board.pop()
            if score >= beta:
                # fail-high – cut
                return beta

        best_move = None
        local_pv = []
        value = -INF
        move_idx = 0

        # Get killer moves for this depth
        killers = self.killer_moves[depth] if depth < len(self.killer_moves) else [None, None]
        
        for m in ordered_moves(board, tt_move, killers, self.history, depth):
            board.push(m)
            self.nodes += 1

            # Late Move Reductions (LMR): reduce depth for later, quiet moves
            next_depth = depth - 1
            if (depth >= self.cfg.lmr_min_depth and move_idx >= self.cfg.lmr_move_idx_threshold
                and not board.is_check() and not board.is_capture(m)):
                reduced = max(1, next_depth - 1)
                score = -self.alphabeta(board, reduced, -alpha - 1, -alpha, [])
                # Re-search at full window if it improves
                if score > alpha:
                    score = -self.alphabeta(board, next_depth, -beta, -alpha, [])
            else:
                score = -self.alphabeta(board, next_depth, -beta, -alpha, [])

            board.pop()

            if self.stop:
                return alpha  # return best-so-far

            if score > value:
                value = score
                best_move = m
                # principal variation handling
                local_pv = [m]  # children PV handled implicitly
            if value > alpha:
                alpha = value
                # record PV at root only (pv passed from root)
                if pv is not None:
                    pv.clear()
                    pv.extend(local_pv)
            if alpha >= beta:
                # Beta cutoff - update killer moves and history
                if not board.is_capture(m) and not board.gives_check(m):
                    # Add to killer moves
                    if killers[0] != m:
                        killers[1] = killers[0]
                        killers[0] = m
                    
                    # Update history heuristic
                    self.history[m] = self.history.get(m, 0) + depth * depth
                break

            move_idx += 1

        self.tt.store(z, depth, value, best_move, alpha, beta)
        return value

    def search(self, board) -> Tuple[str, int, List[str], int, int]:
        """Iterative deepening driver with aspiration windows. Returns (bestMoveUci, scoreCp, pvUci[], nodes, depthReached)"""
        self.start_time = time.time()
        self.stop = False
        self.nodes = 0
        best_move_uci = "0000"
        best_score = 0
        depth_reached = 0
        pv_moves = []
        
        # Aspiration window parameters
        aspiration_window = 50  # centipawns
        alpha = -INF
        beta = INF

        for depth in range(1, self.cfg.max_depth + 1):
            if self.stop:
                break
                
            pv = []
            
            # Use aspiration windows for depths > 1
            if depth > 1:
                alpha = max(-INF, best_score - aspiration_window)
                beta = min(INF, best_score + aspiration_window)
                
                score = self.alphabeta(board, depth, alpha, beta, pv)
                
                # If score is outside aspiration window, re-search with full window
                if score <= alpha or score >= beta:
                    alpha = -INF
                    beta = INF
                    score = self.alphabeta(board, depth, alpha, beta, pv)
            else:
                # First iteration: full window
                score = self.alphabeta(board, depth, -INF, INF, pv)
            
            depth_reached = depth
            if pv:
                pv_moves = pv[: self.cfg.pv_max_len]
                best_move_uci = pv_moves[0].uci()
                best_score = score

            # time or node cap check between iterations
            if self.time_up() or self.nodes >= self.cfg.node_limit:
                self.stop = True
                break

        pv_uci = [m.uci() for m in pv_moves]
        return best_move_uci, best_score, pv_uci, self.nodes, depth_reached

""" 
Find the best move for a given FEN string 
    Args:
        fen: The FEN string of the board
        max_depth: The maximum depth of the search
        time_budget_ms: The time budget in milliseconds
        node_limit: The node limit
        skill_level: The skill level
    Returns:
        - best_move: The best move as a string (e.g. "e2e4")
        - score_cp: The score in centipawns
        - pv: The principal variation as a list of moves
        - nodes_visited: The number of nodes visited
        - time_taken_ms: The time taken in milliseconds
"""
def find_best_move(fen: str, 
                    max_depth: int = 5, 
                    time_budget_ms: int = 900,
                    node_limit: int = 150_000,
                    skill_level: int = 10
                ) -> Tuple[str, int, List[str], int, int]:
    # Map skill -> Config knobs (simple presets)
    blend = 0.8 + 0.02 * min(max(skill_level, 1), 10)  # Higher model weight
    cfg = EngineConfig(
        max_depth=max_depth,
        time_budget_ms=time_budget_ms,
        node_limit=node_limit,
        blend_nn_with_material=min(blend, 0.95),
        enable_quiescence=True,
        use_null_move=skill_level >= 5, 
        null_move_R = 2,
        lmr_min_depth = 3, 
        lmr_move_idx_threshold = 3 if skill_level >= 7 else 5,
    )
    board = chess.Board(fen)
    engine = Search(cfg)
    move, score, pv, nodes, d = engine.search(board)
    return move, score, pv, nodes, d
    
    