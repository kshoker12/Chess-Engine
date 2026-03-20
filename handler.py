import runpod
import chess
import chess.pgn
import io
import torch
import torch.nn.functional as F
import numpy as np
import random
import traceback
from policy_transformer.inference import predict_next_move
from value_transformer.inference import ChessEvaluator
from value_transformer.approximate import approximate_cp

# Warm-start: Load model at module level
print("Loading Agent 0 (4o1 model)...")
try:
    Agent0 = ChessEvaluator('value_transformer/mini_value_6o4.pt')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    Agent0 = None


def find_best_move_transformer(pgn: str, evaluator: ChessEvaluator = None):
    """
    Helper function to find best move using transformer policy and value network.
    Reused from app.py logic.
    """
    if evaluator is None:
        evaluator = Agent0
    
    # Get Candidates
    top_moves = predict_next_move(pgn, top_k=9)
    if not top_moves:
        return None, None

    # Get Root Board
    pgn_io = io.StringIO(pgn)
    game = chess.pgn.read_game(pgn_io)
    if game:
        root_board = game.end().board()
    else:
        root_board = chess.Board()
        
    best_score = -float('inf')
    best_val = -float('inf')
    best_move = None
    
    logits = torch.tensor([s for s, m in top_moves])
    probs = F.softmax(logits, dim=0).tolist()

    for i, (score, move) in enumerate(top_moves):
        prob = probs[i]
        
        board = root_board.copy()
        try:
            board.push_uci(move)
        except ValueError:
            continue
            
        if board.is_checkmate():
            return move, float('inf')

        black_turn = board.turn == chess.BLACK
        view = board.mirror() if black_turn else board 
        
        second_cp = -evaluator.evaluate(view.fen())
        approx_cp = -np.tanh(approximate_cp(view) / 400)

        lam = random.uniform(0, 0.0)
        lam1 = random.uniform(0, 0.1)
        combined_value = (1 - lam1) * second_cp + lam1 * approx_cp
        combined_score = (1 - lam) * combined_value + lam * prob
        
        if combined_score > best_score:
            best_score = combined_score
            best_move = move
            best_val = combined_value

    if best_move:
        return best_move, best_val 
    else:
        return None, None


def handle_alphabeta_eval(input_data):
    """Handle alphabeta-eval endpoint."""
    try:
        from engine.alphabeta import AlphaBeta
        
        # Extract parameters
        pgn = input_data.get('pgn')
        if not pgn:
            return {
                "best_move": "",
                "score": 0.0,
                "details": [],
                "error": "Missing required parameter: pgn"
            }
        
        depth = input_data.get('depth', 3)
        
        # Create searcher
        searcher = AlphaBeta(
            pgn, 
            value_func=Agent0.evaluate, 
            policy_func=predict_next_move, 
            depth=depth
        )
        best_move, score = searcher.search()
        
        # Clamp score for JSON serialization
        safe_score = max(-1e6, min(1e6, float(score)))
        
        return {
            "best_move": best_move if best_move else "",
            "score": safe_score,
            "details": [],
            "error": None
        }
        
    except Exception as e:
        traceback.print_exc()
        return {
            "best_move": "",
            "score": 0.0,
            "details": [],
            "error": str(e)
        }


def handle_mcts3(input_data):
    """Handle mcts-3 endpoint."""
    try:
        from engine.mcts import MCTS
        
        # Extract parameters
        pgn = input_data.get('pgn')
        if not pgn:
            return {
                "best_move": "",
                "score": 0.0,
                "details": [],
                "error": "Missing required parameter: pgn"
            }
        
        simulations = input_data.get('simulations', 800)
        
        # Create searcher
        searcher = MCTS(
            pgn, 
            value_func=Agent0.batch_evaluate, 
            policy_func=predict_next_move,
            num_simulations=simulations
        )
        best_move, score = searcher.search()
        
        return {
            "best_move": best_move if best_move else "",
            "score": float(score),
            "details": [],
            "error": None
        }
        
    except Exception as e:
        traceback.print_exc()
        return {
            "best_move": "",
            "score": 0.0,
            "details": [],
            "error": str(e)
        }


def handle_transformer_move(input_data):
    """Handle transformer-move endpoint."""
    try:
        # Extract parameters
        pgn = input_data.get('pgn')
        if not pgn:
            return {
                "best_move": "",
                "error": "Missing required parameter: pgn"
            }
        
        # Parse PGN and get root board
        pgn_io = io.StringIO(pgn)
        game = chess.pgn.read_game(pgn_io)
        if game:
            root_board = game.end().board()
        else:
            root_board = chess.Board()
            
        top_moves = predict_next_move(pgn, top_k=12)
        if not top_moves:
            return {"best_move": "", "error": "No moves found"}

        # Calculate probabilities
        logits = torch.tensor([s for s, m in top_moves])
        probs = F.softmax(logits, dim=0).tolist()
        
        best_move = None
        best_score = -float('inf')

        logs = []
        
        for i, (score, move) in enumerate(top_moves):
            prob = probs[i]
            board = root_board.copy()
            
            # Fix: move is a UCI string, use push_uci
            try:
                board.push_uci(move) 
            except ValueError:
                continue
                
            new_pgn_game = chess.pgn.Game.from_board(board)
            new_pgn_str = str(new_pgn_game)
            print(f"Checking move {move}")
            opp_move, opp_score = find_best_move_transformer(new_pgn_str, evaluator=Agent0)
            if opp_score is None:
                if board.is_checkmate():
                     opp_score = float('inf')
                else:
                     opp_score = 0.0 # Stalemate/Draw
            else:
                opp_score = -opp_score 
        
            lam = random.uniform(0, 0.05)
            combined_score = (1 - lam) * opp_score + lam * prob 
        
            if combined_score > best_score:
                best_score = combined_score
                best_move = move

            logs.append({
                "move": move,
                "prob": prob,
                "val": opp_score,
                "score": combined_score
            })

        logs.sort(key=lambda x: x["score"], reverse=True)
    
        for log in logs:
            print(f"Move: {log['move']} | Prob: {log['prob']:.3f} | Val: {log['val']:.2f} | Score: {log['score']:.4f}")
        
        print(f"Best Move: {best_move} | Best Score: {best_score:.4f}")

        if best_move: 
            return {"best_move": best_move, "error": None}
        else:
            return {"best_move": "", "error": "No valid moves found"}
            
    except Exception as e:
        traceback.print_exc()
        return {"best_move": "", "error": str(e)}


def handler(event):
    """
    Main RunPod handler function.
    Routes requests to appropriate endpoint handlers based on 'endpoint' field in input.
    """
    try:
        # Check if model is loaded
        if Agent0 is None:
            return {
                "error": "Model not loaded. Check container logs for initialization errors."
            }
        
        # Extract input data
        input_data = event.get('input', {})
        
        # Get endpoint type
        endpoint = input_data.get('endpoint')
        
        if not endpoint:
            return {
                "error": "Missing required parameter: endpoint. Must be one of: 'alphabeta-eval', 'mcts-3', 'transformer-move'"
            }
        
        # Route to appropriate handler
        if endpoint == 'alphabeta-eval':
            return handle_alphabeta_eval(input_data)
        elif endpoint == 'mcts-3':
            return handle_mcts3(input_data)
        elif endpoint == 'transformer-move':
            return handle_transformer_move(input_data)
        else:
            return {
                "error": f"Unknown endpoint: {endpoint}. Must be one of: 'alphabeta-eval', 'mcts-3', 'transformer-move'"
            }
            
    except Exception as e:
        traceback.print_exc()
        return {
            "error": f"Handler error: {str(e)}"
        }


# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
