from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from mangum import Mangum
import numpy as np
import torch
import torch.nn.functional as F
import chess
import chess.pgn
import io 
import random
from value_transformer.approximate import approximate_cp
from policy_transformer.inference import predict_next_move
from value_transformer.inference import ChessEvaluator

from contextlib import contextmanager

# Lazy import - only load when needed
def get_engine():
    from engine import find_best_move
    return find_best_move

# Initialize Agents
print("Loading Agent 0...")
Agent0 = ChessEvaluator('value_transformer/mini_value_6o4.pt')
print("Loading Agent 1...")
Agent1 = ChessEvaluator('value_transformer/mini_value_6o4.pt')

# Default to Agent0 for backward compatibility if needed, though we will select explicitly
ValueTransformer = Agent0 

@contextmanager
def swap_engine_evaluator(agent_id: int):
    """
    Context manager to temporarily swap the evaluator used by the engine.
    This patches engine._eval_model_cache[(False, True)] which corresponds to 
    use_pytorch=False, use_transformer=True.
    """
    import engine
    
    # 1. Select the evaluator
    selected_evaluator = Agent1 if agent_id == 1 else Agent0
    
    # 2. Key for transformer model in engine's cache
    # Key structure in engine.py: (use_pytorch, use_transformer)
    cache_key = (False, True)
    
    # 3. Store original evaluator (if any) to restore later
    original_evaluator = engine._eval_model_cache.get(cache_key)
    
    try:
        # 4. Patch the cache
        engine._eval_model_cache[cache_key] = selected_evaluator
        yield selected_evaluator
    finally:
        # 5. Restore original evaluator
        if original_evaluator is not None:
             engine._eval_model_cache[cache_key] = original_evaluator
        else:
             # If it wasn't there before, remove it (though strictly engine might have loaded it)
             # Safer to just leave it or restore strictly. 
             # Given engine.py logic, it loads on demand. unique swap is fine.
             pass

def find_best_move_transformer(pgn: str, evaluator: ChessEvaluator = None):
    
    # Use provided evaluator or default to Agent0
    if evaluator is None:
        evaluator = Agent0

    # 1. Get Candidates
    top_moves = predict_next_move(pgn, top_k = 9)
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

    # print(f"Turn: {root_board.turn} (Evaluating as Active Player)")

    for i, (score, move) in enumerate(top_moves):
        prob = probs[i]
        
        board = root_board.copy()
        try:
            board.push_uci(move)
        except ValueError:
            continue
            
        if board.is_checkmate():
            return move, float('inf')

        # view = board 
        # cp_score = evaluator.evaluate(view.fen())

        # if board.turn == chess.WHITE:
        #     cp_score = -cp_score

        black_turn = board.turn == chess.BLACK
        
        view = board.mirror() if black_turn else board 
        
        second_cp = -evaluator.evaluate(view.fen())
        approx_cp = -np.tanh(approximate_cp(view) / 400)

        print(f"flipped_score {second_cp}", f"move {move}")
        
        lam = random.uniform(0, 0.0)
        lam1 = random.uniform(0, 0.1)
        combined_value = (1- lam1) * second_cp + lam1 * approx_cp
        combined_score = (1- lam) * combined_value + lam * prob
        
        if combined_score > best_score:
            best_score = combined_score
            best_move = move
            best_val = combined_value
        
        # print(f"Move: {move} | Prob: {prob:.3f} | Val: {cp_score:.2f} | Score: {combined_score:.4f}") 

    if best_move:
       #  print(f"Selected: {best_move} ({best_score:.2f})")
        return best_move, best_val 
    else:
        return None, None

# Create a FastAPI instance
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Move Request type 
class MoveRequest(BaseModel):
    fen: str
    max_depth: int = 4
    difficulty: str = "medium"  # "easy", "medium", "hard"
    agent: int = 0

class MoveResponse(BaseModel):
    best_move: str
    score_cp: int

class TransformerEvalRequest(BaseModel):
    fen: str

class TransformerEvalResponse(BaseModel):
    score_cp: float
    error: Optional[str] = None 

class TransformerMoveRequest(BaseModel):
    pgn: str
    agent: int = 0

class TransformerMoveResponse(BaseModel):
    best_move: str
    error: Optional[str] = None

class CustomSearchRequest(BaseModel):
    pgn: str
    depth: int = 2
    agent: int = 0

class CustomSearchResponse(BaseModel):
    best_move: str
    score: float
    details: list
    error: Optional[str] = None

class SmartSearchRequest(BaseModel):
    pgn: str
    depth: int = 8

class SmartSearchResponse(BaseModel):
    best_move: str
    score: float
    details: list
    error: Optional[str] = None

class DepthSmartSearchRequest(BaseModel):
    pgn: str
    depth: int = 2
    agent: int = 0

class DepthSmartSearchResponse(BaseModel):
    best_move: str
    score: float
    details: list
    error: Optional[str] = None


@app.get("/healthz")
def health_check():
    return {"status" : "ok"}

@app.post("/v1/api/move", response_model=MoveResponse)
def get_ai_move(req: MoveRequest):   
    try:
        # Lazy load engine only when needed
        find_best_move = get_engine()
        
        # Swap evaluator based on agent ID
        with swap_engine_evaluator(req.agent):
            move, score, pv, nodes, depth = find_best_move(
                req.fen, 
                max_depth=req.max_depth,
                difficulty=req.difficulty
            )
        return {"best_move": str(move), "score_cp": int(score)}
    except Exception as e:
        # Fallback to a simple move if engine fails
        import chess
        board = chess.Board(req.fen)
        legal_moves = list(board.legal_moves)
        if legal_moves:
            fallback_move = str(legal_moves[0])
            return {"best_move": fallback_move, "score_cp": 0}
        else:
            return {"best_move": "e2e4", "score_cp": 0}


@app.post("/v1/api/transformer-move", response_model=TransformerMoveResponse)
def get_transformer_move(req: TransformerMoveRequest):
    # move, score = find_best_move_transformer(req.pgn)
    # return {"best_move": move, "error": ""}
    try:
        from policy_transformer.inference import predict_next_move
        from value_transformer.inference import ChessEvaluator
        
        pgn_io = io.StringIO(req.pgn)
        game = chess.pgn.read_game(pgn_io)
        if game:
            root_board = game.end().board()
        else:
            root_board = chess.Board()
            
        top_moves = predict_next_move(req.pgn, top_k = 12)
        if not top_moves:
             return {"best_move": "", "error": "No moves found"}

        # Select Evaluator
        evaluator = Agent1 if req.agent == 1 else Agent0

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
            opp_move, opp_score = find_best_move_transformer(new_pgn_str, evaluator=evaluator)
            if opp_score is None:
                if board.is_checkmate():
                     opp_score = float('inf')
                else:
                     opp_score = 0.0 # Stalemate/Draw
            else:
                opp_score = -opp_score 
        
            lam = random.uniform(0, 0.1)
            combined_score = (1- lam) * opp_score + lam * prob 
        
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
            return {"best_move": best_move}
        else:
            return {"best_move": "", "error": "No valid moves found"}
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"best_move": "", "error": str(e)}

class MCTSEvalRequest(BaseModel):
    pgn: str
    simulations: int = 400
    agent: int = 0

class MCTSEvalResponse(BaseModel):
    best_move: str
    score: float # Q-value
    details: list
    error: Optional[str] = None

class AlphaBetaEvalRequest(BaseModel):
    pgn: str
    depth: int = 3
    agent: int = 0

class AlphaBetaEvalResponse(BaseModel):
    best_move: str
    score: float # Q-value
    details: list
    error: Optional[str] = None



@app.post("/v1/api/alphabeta-eval", response_model=AlphaBetaEvalResponse)
def alphabeta_eval(req: AlphaBetaEvalRequest):
    try:
        from engine.alphabeta import AlphaBeta
        
        # Select Evaluator
        evaluator = Agent1 if req.agent == 1 else Agent0
        
        searcher = AlphaBeta(req.pgn, value_func=evaluator.evaluate, policy_func=predict_next_move, depth=req.depth)
        best_move, score = searcher.search()
        
        # Clamp score: inf/-inf (checkmate) is not JSON-serializable
        safe_score = max(-1e6, min(1e6, float(score)))
        return {
            "best_move": best_move if best_move else "",
            "score": safe_score,
            "details": [],
            "error": None
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "best_move": "",
            "score": 0.0,
            "details": [],
            "error": str(e)
        }

@app.post("/v1/api/mcts-3", response_model=MCTSEvalResponse)
def mcts_3(req: MCTSEvalRequest):
    try:
        from engine.mcts import MCTS
        
        print('evaluating mcts-3')
        searcher = MCTS(req.pgn, value_func=Agent0.batch_evaluate, policy_func=predict_next_move, num_simulations=req.simulations)
        best_move, score = searcher.search()
        
        return {
            "best_move": best_move if best_move else "",
            "score": float(score),
            "details": [],
            "error": None
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "best_move": "",
            "score": 0.0,
            "details": [],
            "error": str(e)
        }


# Lambda handler
handler = Mangum(app)