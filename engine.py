

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
    enable_center_control: bool = True   # encourage center play in opening
    enable_blunder_prevention: bool = True  # prevent obvious tactical blunders

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

# Center control heatmap for opening phase (encourages central development)
CENTER_HEATMAP = [
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,  10,  15,  15,  10,   0,   0,
    0,   0,  15,  30,  30,  15,   0,   0,
    0,   0,  15,  30,  30,  15,   0,   0,
    0,   0,  10,  15,  15,  10,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0
]

# Opening phase threshold (first 12 full moves)
OPENING_PHASE_THRESHOLD = 12

def get_game_phase(board: chess.Board) -> str:
    """Determine the game phase based on fullmove number"""
    if board.fullmove_number <= OPENING_PHASE_THRESHOLD:
        return "opening"
    elif board.fullmove_number <= 30:
        return "middlegame"
    else:
        return "endgame"

def see_capture_value(board: chess.Board, move: chess.Move) -> int:
    """Simple SEE (Static Exchange Evaluation) for capture ordering"""
    if not board.is_capture(move):
        return 0
    
    victim = board.piece_type_at(move.to_square)
    attacker = board.piece_type_at(move.from_square)
    
    if victim is None or attacker is None:
        return 0
    
    # Value of captured piece minus value of attacking piece
    return PIECE_VAL.get(victim, 0) - PIECE_VAL.get(attacker, 0)

def full_see_evaluation(board: chess.Board, square: int) -> int:
    """
    Full Static Exchange Evaluation: compute net material from capturing on a square.
    Returns: net gain (positive) or loss (negative) from the exchange sequence.
    """
    if board.piece_type_at(square) is None:
        return 0
    
    # Find the least valuable attacker
    attackers = []
    for attacker in board.attackers(chess.WHITE, square):
        piece_type = board.piece_type_at(attacker)
        if piece_type:
            attackers.append((PIECE_VAL.get(piece_type, 0), attacker, chess.WHITE))
    
    for attacker in board.attackers(chess.BLACK, square):
        piece_type = board.piece_type_at(attacker)
        if piece_type:
            attackers.append((PIECE_VAL.get(piece_type, 0), attacker, chess.BLACK))
    
    if not attackers:
        return 0
    
    # Get the defender's piece value
    piece = board.piece_at(square)
    if piece is None:
        return 0
    defender_val = PIECE_VAL.get(piece.piece_type, 0)
    
    # Get attacker's color (the one who can initiate the capture)
    attackers.sort()  # Sort by value (least valuable first for MVV-LVA)
    
    # Simple approximation: attacker gain is victim - attacker cost
    best_attacker_val = attackers[0][0] if attackers else 0
    
    # If we capture, we gain victim and lose attacker
    return defender_val - best_attacker_val

def get_attackers_count(board: chess.Board, square: int) -> int:
    """Count how many pieces attack a square"""
    white_attackers = board.attackers(chess.WHITE, square)
    black_attackers = board.attackers(chess.BLACK, square)
    return len(white_attackers) + len(black_attackers)

def get_defenders_count(board: chess.Board, square: int) -> int:
    """Count how many pieces defend a square"""
    if board.piece_at(square) is None:
        return 0
    
    piece = board.piece_at(square)
    defenders = board.attackers(piece.color, square)
    return len(defenders)

def causes_material_loss(board: chess.Board, move: chess.Move) -> int:
    """
    Check if making a move causes immediate material loss.
    Returns the worst-case material delta (negative = losing material).
    This is a conservative 1-ply lookahead for opponent's best capture.
    """
    # Make the move
    board.push(move)
    
    # Track material before potential capture
    initial_material = sum(len(board.pieces(p, board.turn)) * PIECE_VAL.get(p, 0) 
                          for p in PIECE_VAL.keys())
    
    # Find opponent's best capture
    worst_delta = 0
    for opponent_move in board.legal_moves:
        if board.is_capture(opponent_move):
            board.push(opponent_move)
            
            # Calculate material after capture
            final_material = sum(len(board.pieces(p, board.turn)) * PIECE_VAL.get(p, 0) 
                               for p in PIECE_VAL.keys())
            delta = final_material - initial_material
            
            if delta < worst_delta:
                worst_delta = delta
            
            board.pop()
    
    board.pop()
    return worst_delta

def is_hanging_piece(board: chess.Board, move: chess.Move) -> bool:
    """
    Check if the moved piece hangs (is attacked and not adequately defended).
    Returns True if the piece is hanging after the move.
    """
    # Check if this is a piece move (not castling, promotion, etc.)
    piece = board.piece_at(move.from_square)
    if piece is None:
        return False
    
    # Make the move
    board.push(move)
    
    # Check if the destination square is attacked by opponent
    dest_square = move.to_square
    opponent_color = not board.turn
    
    # Find attackers
    attackers = list(board.attackers(opponent_color, dest_square))
    
    # Find defenders (check the piece's own color)
    piece_at_square = board.piece_at(dest_square)
    if piece_at_square is None:
        board.pop()
        return False
    
    defenders = list(board.attackers(piece_at_square.color, dest_square))
    
    board.pop()
    
    # If more attackers than defenders, piece is hanging
    return len(attackers) > len(defenders) and len(attackers) > 0

def center_control_bonus(board: chess.Board) -> int:
    """Reward moves that develop pieces toward the center in the opening"""
    bonus = 0
    phase = get_game_phase(board)
    
    if phase != "opening":
        return 0
    
    # Reward pieces in the center
    for square in range(64):
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        if piece.color == chess.WHITE:
            bonus += CENTER_HEATMAP[square]
        else:
            bonus -= CENTER_HEATMAP[chess.square_mirror(square)]
    
    return bonus if board.turn == chess.WHITE else -bonus

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
    
    # Add center control bonus in opening
    cp += center_control_bonus(board)
    
    return int(round(max(min(cp, INF), -INF)))

# MVV-LVA-ish capture ordering plus TT move bonus
def ordered_moves(board, tt_move, killer_moves: List[Optional[chess.Move]], history: dict, depth: int):
    moves = list(board.legal_moves)
    
    # Determine if we're in opening phase
    is_opening = board.fullmove_number <= OPENING_PHASE_THRESHOLD
    
    def score(m):
        # 1. Transposition table move (highest priority)
        if tt_move and m == tt_move:
            return 10_000_000
        
        # 2. Captures with SEE (Static Exchange Evaluation)
        if board.is_capture(m):
            see_val = see_capture_value(board, m)
            # Use SEE to prioritize profitable captures
            return 1_000_000 + see_val * 100 + (board.piece_type_at(m.to_square) or 0) * 10
        
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
        
        # 6. Opening development: reward center development in opening
        if is_opening:
            to_square = m.to_square
            center_bonus = CENTER_HEATMAP[to_square] if board.turn == chess.WHITE else CENTER_HEATMAP[chess.square_mirror(to_square)]
            if center_bonus > 0:
                return 50_000 + center_bonus * 10
        
        # 7. Checks (more expensive, so lower priority)
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
        self.current_depth = 0  # Track current search depth
        
        # Killer moves: moves that caused beta cutoffs at each depth
        self.killer_moves = [[None, None] for _ in range(64)]  # 2 killer moves per depth
        
        # History heuristic: track how often moves cause cutoffs
        self.history = {}  # move -> count of cutoffs

    def quick_blunder_check(self, board: chess.Board) -> int:
        """Check for immediate tactical blunders: captures, checks, and obvious threats"""
        # Check for immediate checkmate
        if board.is_checkmate():
            return -INF
        
        # Check if in check (punish being in check)
        if board.is_check():
            return -50
        
        # Quick material check after captures
        # Count legal moves as a safety measure
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 0 and not board.is_checkmate():
            return -INF  # Stalemate or other terminal
        
        return 0  # No immediate blunder detected

    def filter_blunders(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """
        Conservative filter: remove moves with obvious material loss at the root.
        This prevents the engine from making blatant tactical blunders.
        """
        if not self.cfg.enable_blunder_prevention:
            return moves
        
        safe_moves = []
        
        for move in moves:
            is_safe = True
            
            # Get the piece being moved
            piece = board.piece_at(move.from_square)
            
            # Test 1: Check if moving into an attacked square with insufficient defense
            # This catches hanging pieces
            if is_hanging_piece(board, move):
                if piece and piece.piece_type in PIECE_VAL:
                    piece_value = PIECE_VAL[piece.piece_type]
                    # Filter if losing a pawn or more
                    if piece_value >= 100:
                        is_safe = False
                        continue
            
            # Test 2: Check immediate material loss after move (1-ply lookahead)
            material_loss = causes_material_loss(board, move)
            # More aggressive threshold: filter losing more than 50 cp (half a pawn)
            if material_loss < -50:
                is_safe = False
                continue
            
            # Test 3: For captures, verify the SEE score
            if board.is_capture(move):
                see_score = see_capture_value(board, move)
                # Filter if capturing loses significant material
                if see_score < -100:
                    is_safe = False
                    continue
            
            # Test 4: Check piece value - don't move high-value pieces into attacks
            if piece and piece.piece_type in PIECE_VAL:
                piece_value = PIECE_VAL[piece.piece_type]
                # If moving a queen or rook, be extra careful
                if piece_value >= 500:  # Queen or Rook
                    # Make the move and check if it's attacked
                    board.push(move)
                    opp_attackers = board.attackers(not board.turn, move.to_square)
                    board.pop()
                    
                    # If the piece is under attack, filter unless there's compensation
                    if len(list(opp_attackers)) > 0 and material_loss < -200:
                        is_safe = False
                        continue
            
            if is_safe:
                safe_moves.append(move)
        
        # Fallback: if all moves filtered, return original (avoid no-move scenario)
        return safe_moves if safe_moves else moves

    def time_up(self) -> bool:
        if self.cfg.time_budget_ms <= 0:
            return False
        return (time.time() - self.start_time) * 1000.0 >= self.cfg.time_budget_ms

    # Quiescence: search only captures and checks to stabilize leaf evals
    def quiesce(self, board, alpha: int, beta: int) -> int:
        stand_pat = static_eval(board, self.cfg)
        
        # Blunder check in quiescence - don't allow obvious bad moves
        if board.is_checkmate():
            return -INF + 100  # Mate is very bad
        if board.is_check():
            stand_pat -= 30  # Penalty for being in check
        
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

    def alphabeta(self, board, depth: int, alpha: int, beta: int, pv, is_root: bool = False) -> int:
        if self.stop or self.nodes >= self.cfg.node_limit or self.time_up():
            self.stop = True
            return static_eval(board, self.cfg)

        z = hash(board._transposition_key())
        val, tt_move, hit = self.tt.probe(z, depth, alpha, beta)
        if hit:
            return val

        # Quick blunder check at all depths for immediate tactical issues
        if depth <= 2:  # Apply at shallow depths where tactics matter most
            blunder_penalty = self.quick_blunder_check(board)
            if abs(blunder_penalty) > 100:  # Significant blunder
                return blunder_penalty

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
            score = -self.alphabeta(board, depth - 1 - self.cfg.null_move_R, -beta, -beta + 1, [], False)
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
        
        # Get moves and filter blunders at root level
        moves = ordered_moves(board, tt_move, killers, self.history, depth)
        
        # Apply blunder filter at root (only at the actual root depth being searched)
        if is_root and self.cfg.enable_blunder_prevention and depth == self.current_depth:
            moves = self.filter_blunders(board, moves)
        
        for m in moves:
            board.push(m)
            self.nodes += 1

            # Late Move Reductions (LMR): reduce depth for later, quiet moves
            next_depth = depth - 1
            if (depth >= self.cfg.lmr_min_depth and move_idx >= self.cfg.lmr_move_idx_threshold
                and not board.is_check() and not board.is_capture(m)):
                reduced = max(1, next_depth - 1)
                score = -self.alphabeta(board, reduced, -alpha - 1, -alpha, [], False)
                # Re-search at full window if it improves
                if score > alpha:
                    score = -self.alphabeta(board, next_depth, -beta, -alpha, [], False)
            else:
                score = -self.alphabeta(board, next_depth, -beta, -alpha, [], False)

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
        
        # Pre-filter candidate moves at root to prevent blunders
        if self.cfg.enable_blunder_prevention:
            # Filter will be applied during search iterations
            pass
        
        # Aspiration window parameters
        aspiration_window = 50  # centipawns
        alpha = -INF
        beta = INF

        for depth in range(1, self.cfg.max_depth + 1):
            if self.stop:
                break
            
            # Set current depth so blunder filter knows we're at root
            self.current_depth = depth
                
            pv = []
            
            # Use aspiration windows for depths > 1
            if depth > 1:
                alpha = max(-INF, best_score - aspiration_window)
                beta = min(INF, best_score + aspiration_window)
                
                score = self.alphabeta(board, depth, alpha, beta, pv, is_root=True)
                
                # If score is outside aspiration window, re-search with full window
                if score <= alpha or score >= beta:
                    alpha = -INF
                    beta = INF
                    score = self.alphabeta(board, depth, alpha, beta, pv, is_root=True)
            else:
                # First iteration: full window
                score = self.alphabeta(board, depth, -INF, INF, pv, is_root=True)
            
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
    
    