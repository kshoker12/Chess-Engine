import chess
import chess.pgn
from typing import Optional
import torch
import torch.nn.functional as F
import io

'''
Node for MCTS
- board: current board state
- pgn:   cached PGN string (built once, inherited by children)
- parent: parent node
- children: dictionary of child nodes
- N: number of visits
- W: total value
- P: prior probability
'''
class MCTSNode:
    def __init__(self, board: chess.Board, pgn: str = "", parent: Optional['MCTSNode'] = None):
        self.board = board
        self.pgn = pgn          # cached — never recomputed
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.P = 0.0

class MCTS:
    def __init__(self, pgn, value_func, policy_func, num_simulations: int = 150, c_punc: float = 1.00):
        # Build PGN once for the root board at construction time
        self._init_root(pgn)
        self.num_simulations = num_simulations
        self.c_punc = c_punc
        self.value_func = value_func
        self.policy_func = policy_func
        self.k = 12
        self.tau = 8
        self.global_eval = 0
        self.global_expansions = 0 

    def _init_root(self, pgn):
        pgn_io = io.StringIO(pgn)
        game = chess.pgn.read_game(pgn_io)
        if game:
             root_board = game.end().board()
        else:
             root_board = chess.Board()
        self.root = MCTSNode(root_board, pgn=pgn)

    def search(self):
        for i in range(self.num_simulations):
            if i == 1:
                self.k = 7
                self.tau = 4.0
            if i == self.num_simulations // 10:
                self.tau = 2.5
                self.k = 5
            node = self._select(self.root)
            value = self._evaluate(node)
            if i < self.num_simulations - 1 and not node.board.is_game_over():
                self._expand(node)
            self._backpropagate(node, value)

        best_move = None
        best_score = -float('inf')
        logs = []
        for key, child in self.root.children.items():
            score = child.N
            logs.append((key, score, child.W / child.N if child.N > 0 else child.W, child.P))
            if score > best_score:
                best_score = score
                best_move = key
        logs.sort(key=lambda x: x[1], reverse=True)
        for log in logs:
            print(log)
        print(f"Evaluations: {self.global_eval}")
        print(f"Expansions: {self.global_expansions}")
        return best_move, best_score

    def _expand(self, node: MCTSNode):
        # Use cached PGN — no PGN reconstruction on every call
        moves = self.policy_func(node.pgn, top_k=self.k)
        if not moves:
            return
        self.global_expansions += 1

        logits = torch.tensor([s for s, _ in moves], dtype=torch.float32)
        probs = F.softmax(logits / self.tau, dim=0).tolist()

        fens = []
        child_moves = []
        for prob, (_, move) in zip(probs, moves):
            child_board = node.board.copy()
            child_board.push_uci(move)

            # Build child PGN cheaply by appending the move to the parent's PGN
            child_pgn = node.pgn.rstrip()
            move_num = child_board.fullmove_number
            if node.board.turn == chess.WHITE:
                child_pgn += f" {move_num}. {node.board.san(chess.Move.from_uci(move))}"
            else:
                child_pgn += f" {node.board.san(chess.Move.from_uci(move))}"

            fen = child_board.mirror().fen() if child_board.turn == chess.BLACK else child_board.fen()
            fens.append(fen)
            child_moves.append((move, prob, child_board, child_pgn))

        if not child_moves:
            return

        # Single batched value call for all new children
        values = self.value_func(fens)

        for i, (move, prob, child_board, child_pgn) in enumerate(child_moves):
            child_node = MCTSNode(child_board, pgn=child_pgn, parent=node)
            child_node.P = prob
            child_node.W = -values[i]
            node.children[move] = child_node

    def _select(self, node: MCTSNode):
        c = self.c_punc
        while node.children:
            best_score = -float('inf')
            best_child = None
            sqrt_N = node.N ** 0.5
            for child in node.children.values():
                Q = child.W / child.N if child.N > 0 else child.W
                U = c * child.P * sqrt_N / (child.N + 1)
                score = Q + U
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child
        return node

    def _evaluate(self, node: MCTSNode):
        self.global_eval += 1
        if node.board.is_game_over():
            return 1.02 if node.board.is_checkmate() else 0
        return node.W

    def _backpropagate(self, node: MCTSNode, value: float):
        while node:
            node.N += 1
            node.W += value
            value = -value
            node = node.parent