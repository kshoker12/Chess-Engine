import chess
import io

'''
Alpha-Beta Pruning algorithm for chess going up until given depth
- pgn: string of pgn
- board: board object from chess library
- value_func: function that returns the value of a board
- policy_func: function that returns the policy of a board
- max_depth: max depth to search
'''
class AlphaBeta:
    def __init__(self, pgn, value_func, policy_func, depth):
        self._init_root(pgn)
        self.value_func = value_func
        self.policy_func = policy_func
        self.max_depth = depth
        self.k = 9
        self.global_expansion = 0
        self.global_eval = 0

    def _init_root(self, pgn):
        pgn_io = io.StringIO(pgn)
        game = chess.pgn.read_game(pgn_io)
        if game:
             root_board = game.end().board()
        else:
             root_board = chess.Board()
        self.root = root_board

    def search(self):
        best_move = None
        best_score = -float('inf')

        top_moves = self._reorder_moves(self.root, self._expand(self.root, self.max_depth))
        for i, (score, move) in enumerate(top_moves):
            temp_board = self.root.copy()
            temp_board.push_uci(move)
            value = self._alpha_beta(temp_board, self.max_depth - 1, -float('inf'), float('inf'))
            if self.root.turn == chess.BLACK: value = -value
            if value > best_score:
                best_score = value
                best_move = move
        print('expansions', self.global_expansion)
        print('evals', self.global_eval)
        return best_move, best_score

    def _alpha_beta(self, board, depth, alpha, beta):
        if depth == 0 or board.is_game_over():
            return self._evaluate(board)
        
        top_moves = self._reorder_moves(board, self._expand(board, depth))

        for i, (score, move) in enumerate(top_moves):
            temp_board = board.copy()
            temp_board.push_uci(move)
            
            value = self._alpha_beta(temp_board, depth - 1, alpha, beta)
            if board.turn == chess.WHITE:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)

            if beta <= alpha:
                break
        
        return alpha if board.turn == chess.WHITE else beta

    def _reorder_moves(self, board, moves):
        return moves
    
    def _expand(self, board, depth):
        pgn = str(chess.pgn.Game.from_board(board))
        self.global_expansion += 1
        return self.policy_func(pgn, top_k=self.k)

    def _evaluate(self, board):
        self.global_eval += 1
        if board.is_game_over():
            if board.is_checkmate():
                return float('inf') if board.turn == chess.BLACK else -float('inf')
            return 0
        if board.turn == chess.BLACK:
            return -self.value_func(board.mirror().fen())
        else:
            return self.value_func(board.fen())