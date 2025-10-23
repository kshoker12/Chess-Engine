from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum

# Lazy import - only load when needed
def get_engine():
    from engine import find_best_move
    return find_best_move

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

class MoveResponse(BaseModel):
    best_move: str
    score_cp: int 

@app.get("/healthz")
def health_check():
    return {"status" : "ok"}

@app.get("/test-model")
def test_model():
    """Test endpoint to verify model is working"""
    try:
        from eval_model import eval_cp
        from features import encode_features
        import chess
        
        board = chess.Board()
        features = encode_features(board)
        score = eval_cp(features)
        
        return {
            "model_working": True,
            "features_shape": features.shape,
            "model_score": score,
            "board_fen": board.fen()
        }
    except Exception as e:
        return {
            "model_working": False,
            "error": str(e)
        }

@app.get("/test-static-eval")
def test_static_eval():
    """Test endpoint to verify static evaluation is working"""
    try:
        import chess
        from engine import static_eval, EngineConfig
        
        board = chess.Board()
        cfg = EngineConfig(blend_nn_with_material=0.8)
        score = static_eval(board, cfg)
        
        return {
            "static_eval_working": True,
            "score": score,
            "board_fen": board.fen()
        }
    except Exception as e:
        return {
            "static_eval_working": False,
            "error": str(e)
        }

@app.post("/v1/api/move", response_model=MoveResponse)
def get_ai_move(req: MoveRequest):   
    try:
        # Lazy load engine only when needed
        find_best_move = get_engine()
        move, score, pv, nodes, depth = find_best_move(req.fen, req.max_depth)
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

# Lambda handler
handler = Mangum(app)