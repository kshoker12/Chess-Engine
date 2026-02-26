
import chess
import chess.pgn
import io
import torch
import random
import math
from typing import List, Tuple, Optional

# Import inference components
from policy_transformer.inference import predict_next_move
from value_transformer.inference import ChessEvaluator


class SmartSearch:
    def __init__(self, evaluator):
         self.value_model = evaluator
         
    def evaluate_board(self, board: chess.Board) -> float:
        """
        Evaluate board from the perspective of the side to move using ValueTransformer.
        ValueTransformer returns score for 'White-like' perspective of the input.
        We mirror if Black, so the input is always 'Player to Move'.
        The output is 'Advantage for Player to Move'.
        """
        if board.is_checkmate():
            return -20000.0 # We have been checkmated (current side to move has no moves and in check)
            
        view = board.mirror() if board.turn == chess.BLACK else board
        # Evaluate returns centipawns
        score = self.value_model.evaluate(view.fen())
        return score

    def get_ranked_candidates(self, pgn: str, board: chess.Board, top_k: int = 9) -> List[Tuple[str, float]]:
        """
        Get candidates from policy, evaluate them with value network, and return sorted list.
        Returns: List of (move_uci, value_score)
        value_score is "My Advantage after making this move" = - "Opponent Advantage".
        """
        # 1. Get Policy Candidates
        # predict_next_move returns [(policy_score, move_uci), ...]
        policy_candidates = predict_next_move(pgn, top_k=top_k)
        
        if not policy_candidates:
            return []
            
        moves_to_eval = []
        fens_to_eval = []
        
        # Prepare batch
        for p_score, move_uci in policy_candidates:
            # Make move on temp board
            temp_board = board.copy()
            try:
                temp_board.push_uci(move_uci)
            except ValueError:
                continue
                
            # Check for game over
            if temp_board.is_game_over():
                 if temp_board.is_checkmate():
                     # I checkmated them!
                     my_advantage = 20000.0
                 else:
                     # Draw
                     my_advantage = 0.0
                 moves_to_eval.append((move_uci, my_advantage, True)) # is_final_score = True
            else:
                # Need to evaluate Opponent's Turn
                # Mirror if Black to generate White-perspective FEN for model
                view = temp_board.mirror() if temp_board.turn == chess.BLACK else temp_board
                fens_to_eval.append(view.fen())
                moves_to_eval.append((move_uci, 0.0, False)) # Placeholders
                
        # Batch Evaluate
        scores = []
        if fens_to_eval:
             scores = self.value_model.batch_evaluate(fens_to_eval)
             
        # Reconstruct Results
        ranked_moves = []
        score_idx = 0
        
        for move_uci, pre_calc_score, is_final in moves_to_eval:
             if is_final:
                 my_advantage = pre_calc_score
             else:
                 # Model returns Opponent's Advantage (from their perspective)
                 opp_advantage = scores[score_idx]
                 score_idx += 1
                 my_advantage = -opp_advantage
                 
             ranked_moves.append((move_uci, my_advantage))
            
        # Sort by My Advantage (Descending)
        ranked_moves.sort(key=lambda x: x[1], reverse=True)
        return ranked_moves

    def rollout(self, pgn: str, board: chess.Board, depth: int) -> float:
        """
        Greedy rollout: pick BEST move at each step and continue.
        Returns the terminal value (from root perspective? No, from current leaf perspective).
        Actually, we want to return the value back up.
        At depth 0, returns current board evaluation.
        At depth > 0, makes Best Move, returns -rollout(child).
        """
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
            
        # Get candidates and pick ONE best
        ranked = self.get_ranked_candidates(pgn, board, top_k=16)
        
        if not ranked:
            if board.is_game_over():
                 return self.evaluate_board(board)
            return -20000.0 
            
        best_move, best_val = ranked[0] # Greedy choice
        
        # Recurse
        board.push_uci(best_move)
        # Generate valid PGN (SAN) from board state
        game = chess.pgn.Game.from_board(board)
        new_pgn = str(game)

        return -self.rollout(new_pgn, board, depth - 1)

    def search(self, pgn: str, depth: int) -> Tuple[str, float, List[dict]]:
        """
        Root search.
        Evaluates top K candidates, then rolls out each one to 'depth'.
        Returns (best_move, best_score, details).
        """
        # Parse Root Board
        pgn_io = io.StringIO(pgn)
        game = chess.pgn.read_game(pgn_io)
        if game:
             root_board = game.end().board()
        else:
             root_board = chess.Board() # Assume start pos if empty PGN
        
        candidates = self.get_ranked_candidates(pgn, root_board, top_k=8)
        
        if not candidates:
            return "", 0.0, []
            
        results = []
        best_move = None
        best_final_score = -float('inf')
        
        for move_uci, immediate_score in candidates:
            # Setup rollout
            temp_board = root_board.copy()
            try:
                temp_board.push_uci(move_uci)
            except ValueError:
                continue
            
            # Game Over check
            if temp_board.is_game_over():
                final_score = immediate_score # Should be 20000 or 0
                results.append({
                    "move": move_uci,
                    "immediate_score": immediate_score,
                    "rollout_score": final_score
                })
                if final_score > best_final_score:
                    best_final_score = final_score
                    best_move = move_uci
                continue

            game = chess.pgn.Game.from_board(temp_board)
            new_pgn = str(game)
            
            # Rollout (returns score from Opponent perspective after move)
            # So My Value = -Rollout Value
            rollout_val = self.rollout(new_pgn, temp_board, depth - 1)
            final_score = -rollout_val
            
            results.append({
                "move": move_uci,
                "immediate_score": immediate_score,
                "rollout_score": final_score
            })
            
            if final_score > best_final_score:
                best_final_score = final_score
                best_move = move_uci
                
        # Sort details
        results.sort(key=lambda x: x['rollout_score'], reverse=True)
        
        return best_move, best_final_score, results


import random
import numpy as np
import torch
import torch.nn.functional as F
from value_transformer.approximate import approximate_cp

class DepthSmartSearch:
    def __init__(self, evaluator):
        self.value_model = evaluator
        self.breadth = 4
        self.candidate_limit = 8

    def _get_scored_moves(self, pgn: str, board: chess.Board) -> List[Tuple[str, float, float]]:
        """
        Get candidates using Policy Transformer.
        Evaluate each candidate using Value Transformer + Approximate CP.
        Combine scores using: (1-lam)*val + lam*prob.
        Returns list of (move_uci, combined_score, raw_value_for_leaf_avg).
        Note: The score returned here is used for pruning.
        The raw_value is used for leaf averaging.
        """
        candidates = predict_next_move(pgn, top_k=self.candidate_limit)
        if not candidates:
            return []

        # Calculate probabilities
        logits = torch.tensor([s for s, m in candidates])
        probs = F.softmax(logits, dim=0).tolist()

        scored_candidates = []

        for i, (policy_score, move_uci) in enumerate(candidates):
            prob = probs[i]
            
            temp_board = board.copy()
            try:
                temp_board.push_uci(move_uci)
            except ValueError:
                continue

            if temp_board.is_game_over():
                if temp_board.is_checkmate():
                    # Mate for side to move (board.turn).
                    # So +Inf for me. 
                    val = 20000.0
                else:
                    val = 0.0
                
                # Combine
                # For consistency with app.py logic:
                # combined = (1-lam)*val + lam*prob 
                # (Ignoring approx for terminal)
                lam = random.uniform(0, 0.05)
                combined_score = (1-lam)*val + lam*prob
                scored_candidates.append((move_uci, combined_score, val, temp_board))
                continue

            # Evaluate Position
            # Logic from app.py:
            # view = board.mirror() if black else board
            # val = -evaluator(view)
            # approx = -tanh(approx(view)/400)
            
            black_turn = temp_board.turn == chess.BLACK
            view = temp_board.mirror() if black_turn else temp_board
            
            # Helper for single eval (can optimize to batch later if needed, but strict logic match first)
            # app.py uses: cp_score = evaluator.evaluate(view.fen())
            # but wait, app.py calls find_best_move_transformer recursively?
            # No, looking at get_transformer_move logic which calls find_best_move_transformer.
            # find_best_move_transformer logic:
            # second_cp = -evaluator.evaluate(view.fen())
            
            raw_eval = self.value_model.evaluate(view.fen())
            second_cp = -raw_eval
            
            approx_val = approximate_cp(view)
            approx_cp = -np.tanh(approx_val / 400)
            
            lam1 = random.uniform(0, 0.1)
            combined_value = (1- lam1) * second_cp + lam1 * approx_cp
            
            lam = random.uniform(0, 0.05)
            combined_score = (1- lam) * combined_value + lam * prob
            
            scored_candidates.append((move_uci, combined_score, combined_value, temp_board))

        # Sort by combined_score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates

    def _collect_leaves(self, pgn: str, board: chess.Board, depth_left: int) -> List[float]:
        """
        Recursively collect leaf values.
        Returns list of values (from Root Perspective).
        """
        # Base Case: Leaf (Depth 0 or Game Over)
        if depth_left == 0 or board.is_game_over():
            # If game over, return score.
            # If depth 0, evaluate position (using Value + Approx).
            # Note: The strict logic uses 'combined_value' (Value + Approx) for evaluation.
            # We need standard evaluation of this board.
            
            if board.is_game_over():
                 if board.is_checkmate():
                     # Check whose turn it is.
                     # If turn is SAME as Root, then Root lost?
                     # Wait. Logic is relative.
                     # Let's say we are evaluating a Leaf.
                     # We want the value from the perspective of the ROOT PLAYER.
                     # The path led here.
                     # But simple approach: Just return the static evaluation of this board 
                     # relative to Side-To-Move, and propagate up?
                     # No, we want Average of Leaves. Implies all leaves are evaluated 
                     # from same perspective.
                     
                     # Let's assume the value returned by `combined_value` above 
                     # is "Advantage for Player who just moved" (because of the negative sign).
                     # Correct.
                     
                     # But here we are at a leaf. We didn't just move. 
                     # We need to evaluate the state.
                     pass 
                 else:
                     return [0.0]

            # Evaluate state
            black_turn = board.turn == chess.BLACK
            view = board.mirror() if black_turn else board
            
            # Raw eval (Advantage for View-Side i.e. White-in-view i.e. Side-To-Move)
            raw_eval = self.value_model.evaluate(view.fen())
            
            # We want value from perspective of Player who made the previous move?
            # Or just Side-To-Move?
            # In `_get_scored_moves`, we calculated `second_cp = -raw_eval`.
            # This is "Advantage for Player who just moved to reach this state".
            # So let's stick to that.
            
            approx_val = approximate_cp(view)
            # Note: approximate_cp returns Centipawns for White? Or Side-to-move?
            # Usually stockfish-like is side-to-move or white? 
            # approximate.py check needed. Assuming side-to-move or white.
            # logic in app.py: approx_cp = -np.tanh(approx(view)/400).
            # This implies approx(view) is for side-to-move?
            # If -tanh is used to flip it, it means approx was for side-to-move and we want neg?
            # Yes, standard minimax/negamax logic.
            
            # So: Leaf Value = - (Value + Approx) for Side-To-Move
            # Which equals Value for Previous Mover.
            
            combined_value = (1.0) * raw_eval + (0.0) * (np.tanh(approx_val/400)) 
            # Wait, sticking to strict logic.
            # `find_best_move_transformer` calculates `combined_value` as the metric.
            # `combined_value = (1- lam1) * second_cp + lam1 * approx_cp`
            # where `second_cp = -raw_eval`.
            # So `combined_value` is effectively negative.
            
            second_cp = -raw_eval
            approx_cp = -np.tanh(approx_val / 400)
            lam1 = random.uniform(0, 0.1)
            
            leaf_val = (1 - lam1) * second_cp + lam1 * approx_cp
            

            if board.is_checkmate():
                # If Checkmate and I am to move, I lost.
                # If turn == root_turn, Root lost. -> -20000.
                pass # Handled by is_checkmate check above?
                # Re-check.
                # If board.is_checkmate():
                #     if board.turn == self.root_turn: return [-20000.0]
                #     else: return [20000.0]
            
            return [leaf_val]


        # Recursive Step
        scored = self._get_scored_moves(pgn, board)
        # scored: [(move, combined_score, raw_val, next_board)]
        
        # Prune to Top K
        best_kids = scored[:self.breadth]
        
        leaves = []
        for _, _, _, next_board in best_kids:
            # Generate PGN
            g = chess.pgn.Game.from_board(next_board)
            next_pgn = str(g)
            
            child_leaves = self._collect_leaves(next_pgn, next_board, depth_left - 1)
            
            # Do we negate child leaves?
            # No, `collect_leaves` returns values from Root Perspective (intended).
            # But we need to ensure `collect_leaves` does that.
            # My logic above for leaf evaluation needs to be robust.
            # "Leaf Value from Root Perspective".
            
            # Let's adjust `collect_leaves` signature or logic to ensure Perspective.
            # Actually, just pass `root_turn` to `collect_leaves`?
            pass
            leaves.extend(child_leaves)
            
        return leaves

    def param_search(self, pgn: str, depth: int):
        # 1. Root
        pgn_io = io.StringIO(pgn)
        game = chess.pgn.read_game(pgn_io)
        if game:
             root_board = game.end().board()
        else:
             root_board = chess.Board()
        
        self.root_turn = root_board.turn
        
        # Root Expansion (Ply 0)
        scored_roots = self._get_scored_moves(pgn, root_board)
        # Top 4
        candidates = scored_roots[:self.breadth]
        
        best_move = None
        best_avg = -float('inf')
        results = []
        
        total_plies = depth * 2
        
        for move_uci, _, val, next_board in candidates:
            # We are at Ply 1 (State after Root Move).
            # We need to go `total_plies - 1` more steps.
            
            g = chess.pgn.Game.from_board(next_board)
            next_pgn = str(g)
            
            leaves = self._collect_leaves_recursive(next_pgn, next_board, total_plies - 1)
            
            if leaves:
                avg = sum(leaves) / len(leaves)
            else:
                avg = -20000.0 # Should not happen unless stalemate immediately
            
            results.append({
                "move": move_uci,
                "score": avg,
                "leaf_count": len(leaves)
            })
            
            if avg > best_avg:
                best_avg = avg
                best_move = move_uci
                
        results.sort(key=lambda x: x["score"], reverse=True)
        return best_move, best_avg, results

    def _collect_leaves_recursive(self, pgn, board, depth_left):
        if depth_left == 0 or board.is_game_over():
            return [self._evaluate_leaf(board)]

        scored = self._get_scored_moves(pgn, board)
        top_k = scored[:self.breadth]
        
        all_leaves = []
        for _, _, _, next_bd in top_k:
            g = chess.pgn.Game.from_board(next_bd)
            n_pgn = str(g)
            all_leaves.extend(self._collect_leaves_recursive(n_pgn, next_bd, depth_left - 1))
            
        return all_leaves

    def _evaluate_leaf(self, board):
        if board.is_game_over():
            if board.is_checkmate():
                # If current turn is Root's turn, Root is mated -> Loss.
                if board.turn == self.root_turn:
                    return -20000.0
                else:
                    return 20000.0
            return 0.0 # Draw
            
        # Evaluate
        black_turn = board.turn == chess.BLACK
        view = board.mirror() if black_turn else board
        
        raw_eval = self.value_model.evaluate(view.fen())
        approx_val = approximate_cp(view)
        
        # Calculate 'combined_value' (Advantage for Previous Mover i.e. NOT board.turn)
        second_cp = -raw_eval
        approx_cp = -np.tanh(approx_val / 400)
        lam1 = random.uniform(0, 0.1)
        
        combined_value = (1 - lam1) * second_cp + lam1 * approx_cp
        
        # combined_value is Advantage for Player who just moved.
        # Who moved? The Opponent of board.turn.
        # If board.turn == root_turn, then Opponent of Root moved.
        # So combined_value is Advantage for Opponent.
        # So Return -combined_value.
        
        if board.turn == self.root_turn:
            return -combined_value
        else:
            return combined_value


class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, pgn="", prior_prob=0.0):
        self.board = board
        self.parent = parent
        self.pgn = pgn # Store PGN string enabling history for Transformer
        
        # Maps move_uci -> child_node
        self.children = {}
        
        # Action Stats
        # N(s, a)
        self.visit_count = {} 
        # W(s, a)
        self.total_action_value = {}
        # Q(s, a)
        self.mean_action_value = {}
        # P(s, a)
        self.prior_prob = {}
        
        self.is_expanded = False
        
        # Store value for this node (from perspective of player to move at this node)
        # Note: Backprop handles sign flipping
        self.value = None

class AlphaMCTS:
    def __init__(self, evaluator, simulations=80, c_puct=1.5):
        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.top_k = 9
        
    def _get_policy(self, pgn: str, root_turn: bool):
        """Returns list of (move_uci, prob)."""
        candidates = predict_next_move(pgn, top_k=self.top_k if root_turn else 7)
        if not candidates:
            return []
            
        logits = torch.tensor([s for s, m in candidates])
        probs = F.softmax(logits, dim=0).tolist()
        
        return list(zip([m for s, m in candidates], probs))

    def _evaluate_board(self, board: chess.Board) -> float:
        """
        Returns value [-1, 1] from perspective of SIDE TO MOVE.
        Using strictly mirrored logic from app.py:
        val = (1-lam)*eval + lam*approx
        """
        if board.is_game_over():
            if board.is_checkmate():
                 # I am mated. Bad for Side To Move.
                 return -1.0 
            return 0.0 # Draw
            
        black_turn = board.turn == chess.BLACK
        view = board.mirror() if black_turn else board
        
        raw_eval = self.evaluator.evaluate(view.fen())
        approx_val = approximate_cp(view)
        
        # Advantage for Previous Mover (based on DepthSmartSearch analysis)
        if black_turn:
            second_cp = -raw_eval
            approx_cp = -np.tanh(approx_val / 400)
        else:
            second_cp = raw_eval
            approx_cp = np.tanh(approx_val / 400)
        lam1 = 0.0
        
        combined_val_prev_mover = (1 - lam1) * second_cp + lam1 * approx_cp
        
        # Value for Side To Move
        val_stm = combined_val_prev_mover
        
        # Clamp to [-1, 1] just in case, though tanh and model output roughly in range
        return max(-1.0, min(1.0, val_stm))

    def search(self, pgn: str):
        pgn_io = io.StringIO(pgn)
        game = chess.pgn.read_game(pgn_io)
        if game:
             root_board = game.end().board()
        else:
             root_board = chess.Board()
             
        root = MCTSNode(root_board, pgn=pgn)
        
        # Expansion for Root (Initialize Stats)
        policy = self._get_policy(pgn, True)
        
        # If no policy moves (mate/stalemate), return
        if not policy:
             return "", 0.0, []
             
        for mv, prob in policy:
            root.visit_count[mv] = 0
            root.total_action_value[mv] = 0.0
            root.mean_action_value[mv] = 0.0
            root.prior_prob[mv] = prob
        root.is_expanded = True
        
        # Simulation Loop
        for _ in range(self.simulations):
            node = root
            path = [] # List of (node, action_taken_to_get_child)
            
            # 1. SELECT
            while node.is_expanded and not node.board.is_game_over():
                # Select best child based on PUCT
                best_uct = -float('inf')
                best_move = None
                
                sum_visit = sum(node.visit_count.values())
                sqrt_sum_visit = np.sqrt(sum_visit) if sum_visit > 0 else 0
                
                found_move = False
                for move_uci in node.prior_prob.keys():
                    q = node.mean_action_value[move_uci]
                    p = node.prior_prob[move_uci]
                    n = node.visit_count[move_uci]
                    
                    # U(s, a) = c * P * sqrt(Sum N) / (1 + N)
                    u = self.c_puct * p * sqrt_sum_visit / (1 + n)
                    score = q + u
                    
                    if score > best_uct:
                        best_uct = score
                        best_move = move_uci
                        found_move = True
                
                if not found_move:
                    break # Should not happen unless no legal moves
                
                path.append((node, best_move))
                
                # Traverse
                if best_move in node.children:
                    node = node.children[best_move]
                else:
                    # Create child
                    new_board = node.board.copy()
                    try:
                        # Append Move to PGN for correct history in child
                        move_obj = chess.Move.from_uci(best_move)
                        san_move = node.board.san(move_obj)
                        child_pgn = node.pgn + " " + san_move
                        
                        new_board.push_uci(best_move)
                    except:
                        break # Invalid?
                        
                    child_node = MCTSNode(new_board, parent=node, pgn=child_pgn)
                    node.children[best_move] = child_node
                    node = child_node
                    # Break to expand/eval this new node
                    break 
            
            # node is now the leaf
            leaf = node
            
            # 2. EVALUATE
            # Value for Side To Move at Leaf
            value = self._evaluate_board(leaf.board)
            
            # 3. EXPAND
            if not leaf.is_expanded and not leaf.board.is_game_over():
                # Use stored PGN with history
                l_pgn = leaf.pgn
                
                policy = self._get_policy(l_pgn, False)
                for mv, prob in policy:
                    leaf.visit_count[mv] = 0
                    leaf.total_action_value[mv] = 0.0
                    leaf.mean_action_value[mv] = 0.0
                    leaf.prior_prob[mv] = prob
                leaf.is_expanded = True
                
            # 4. BACKUP (Negamax)
            leaf_turn = leaf.board.turn
            
            for node_s, action_a in reversed(path):
                # Value for node_s.turn
                if node_s.board.turn == leaf_turn:
                    v_for_node = value
                else:
                    v_for_node = -value
                    
                node_s.visit_count[action_a] += 1
                node_s.total_action_value[action_a] += v_for_node
                node_s.mean_action_value[action_a] = node_s.total_action_value[action_a] / node_s.visit_count[action_a]
        
        # Result
        # Return most visited move (robust)
        best_visited = -1
        best_move = None
        
        details = []
        for mv, n in root.visit_count.items():
            q = root.mean_action_value[mv]
            details.append({
                "move": mv,
                "visits": n,
                "q_value": q,
                "prior": root.prior_prob[mv]
            })
            if n > best_visited:
                best_visited = n
                best_move = mv
                
        # Sort details
        details.sort(key=lambda x: x['visits'], reverse=True)
        
        # Find score of best move
        best_q = root.mean_action_value.get(best_move, 0.0)
        
        return best_move, best_q, details


class AlphaMCTS2:
    def __init__(self, simulations=400, agent=0):
        self.simulations = simulations
        self.agent = agent
        # Config for internal evaluation - defaulting to generally smart settings
        from engine import ordered_moves, _get_eval_model
        
        self.ordered_moves = ordered_moves
        self._get_eval_model = _get_eval_model
        
        self.c_puct = 2.5
        self.top_k = 10 

    def _get_root_policy(self, pgn: str):
        """Use Policy Transformer for Root Priors"""
        candidates = predict_next_move(pgn, top_k=self.top_k)
        if not candidates:
            return []
        
        # Softmax the logits/scores
        logits = torch.tensor([s for s, m in candidates])
        probs = F.softmax(logits, dim=0).tolist()
        return list(zip([m for s, m in candidates], probs))

    def _get_internal_policy(self, board: chess.Board):
        """Use Heuristic Ordering (engine.py) for Internal Priors"""
        # We use ordered_moves to get a ranked list of legal moves
        moves = self.ordered_moves(board, None, [None, None], {}, 0)
        
        # Assign priors based on rank: P(rank) ~ 1 / (rank + k) or exp decay
        # Let's use simple exponential decay
        prior_forcing = []
        total_p = 0.0
        
        for i, move in enumerate(moves):
            # Higher prior for top ranked moves
            # e.g. 1.0, 0.8, 0.64 ...
            p = 0.8 ** i
            prior_forcing.append((move.uci(), p))
            total_p += p
            
        # Normalize
        normalized = [(m, p/total_p) for m, p in prior_forcing]
        return normalized

    def _evaluate_leaf(self, board: chess.Board) -> float:
        """
        Evaluate leaf using RAW Agent0/Agent1 model.
        Bypasses static_eval's material blending and center control.
        """
        if board.is_game_over():
             if board.is_checkmate():
                 return -1.0 # Mated
             return 0.0
             
        # Get the active model (swapped by context manager in app.py)
        model = self._get_eval_model(use_transformer=True)
        
        black_turn = board.turn == chess.BLACK
        view = board.mirror() if black_turn else board
        
        # Evaluate returns raw score (tanh-like [-1, 1])
        val = model.evaluate(view.fen())
        
        if black_turn:
            val = -val
            
        return max(-1.0, min(1.0, val))

    def search(self, pgn: str):
        pgn_io = io.StringIO(pgn)
        game = chess.pgn.read_game(pgn_io)
        if game:
             root_board = game.end().board()
        else:
             root_board = chess.Board()
        
        # Root Node
        root = MCTSNode(root_board, pgn=pgn)
        
        # 1. Expand Root with Policy Network
        policy = self._get_root_policy(pgn)
        if not policy:
             return "", 0.0, []
             
        for mv, prob in policy:
            root.visit_count[mv] = 0
            root.total_action_value[mv] = 0.0
            root.mean_action_value[mv] = 0.0
            root.prior_prob[mv] = prob
        root.is_expanded = True
        
        for _ in range(self.simulations):
            node = root
            path = []
            
            # SELECT
            while node.is_expanded and not node.board.is_game_over():
                best_uct = -float('inf')
                best_move = None
                
                sum_visit = sum(node.visit_count.values())
                sqrt_sum_visit = math.sqrt(sum_visit) if sum_visit > 0 else 0
                
                found = False
                for move_uci in node.prior_prob.keys():
                    q = node.mean_action_value[move_uci]
                    p = node.prior_prob[move_uci]
                    n = node.visit_count[move_uci]
                    
                    u = self.c_puct * p * sqrt_sum_visit / (1 + n)
                    score = q + u
                    
                    if score > best_uct:
                        best_uct = score
                        best_move = move_uci
                        found = True
                        
                if not found: break
                
                path.append((node, best_move))
                
                if best_move in node.children:
                    node = node.children[best_move]
                else:
                    # Leaf reached (virtual)
                    # Create child
                    new_board = node.board.copy()
                    try:
                        move_obj = chess.Move.from_uci(best_move)
                        san_move = node.board.san(move_obj)
                        child_pgn = node.pgn + " " + san_move
                        new_board.push(move_obj)
                    except: break
                    
                    child_node = MCTSNode(new_board, parent=node, pgn=child_pgn)
                    node.children[best_move] = child_node
                    node = child_node
                    break
            
            leaf = node
            
            # EVALUATE
            value = self._evaluate_leaf(leaf.board)
            
            # EXPAND (Internal)
            if not leaf.is_expanded and not leaf.board.is_game_over():
                # Use Heuristic Policy (engine.py ordering)
                policy = self._get_internal_policy(leaf.board)
                for mv, prob in policy:
                    leaf.visit_count[mv] = 0
                    leaf.total_action_value[mv] = 0.0
                    leaf.mean_action_value[mv] = 0.0
                    leaf.prior_prob[mv] = prob
                leaf.is_expanded = True
                
            # BACKUP
            leaf_turn = leaf.board.turn
            for node_s, action_a in reversed(path):
                if node_s.board.turn == leaf_turn:
                    v_node = value
                else:
                    v_node = -value
                    
                node_s.visit_count[action_a] += 1
                node_s.total_action_value[action_a] += v_node
                node_s.mean_action_value[action_a] = node_s.total_action_value[action_a] / node_s.visit_count[action_a]

        # Select Best
        best_visited = -1
        best_move = None
        details = []
        for mv, n in root.visit_count.items():
            q = root.mean_action_value[mv]
            details.append({
                "move": mv, "visits": n, "q_value": q, "prior": root.prior_prob[mv]
            })
            if n > best_visited:
                best_visited = n
                best_move = mv
        
        details.sort(key=lambda x: x['visits'], reverse=True)
        best_q = root.mean_action_value.get(best_move, 0.0)
        
        return best_move, best_q, details