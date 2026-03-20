# Chess AI Service

This repo implements a chess move generation service that combines:
- A transformer policy model to propose candidate moves.
- A transformer value model to evaluate positions.
- Search algorithms (Alpha-Beta and MCTS) to pick a final move.

The service is exposed primarily via a RunPod serverless handler (`handler.py`). There is also a FastAPI app (`app.py`) that exposes a compatible set of endpoints.

## Core components

### Policy transformer (candidate moves)
`policy_transformer/inference.py` provides:
- `predict_next_move(pgn_string, top_k=...)`

Given a PGN string, it reconstructs the board, handles turn parity by mirroring when needed, runs the policy transformer, and returns a ranked list of candidate UCI moves:
- `[(policy_score, move_uci), ...]`

### Value transformer (position evaluation)
`value_transformer/inference.py` provides `ChessEvaluator`:
- `ChessEvaluator(model_path)`
- `evaluate(fen) -> float` (centipawn-like score from White's perspective)
- `batch_evaluate(fens: list[str]) -> list[float]`

### Search algorithms
Search lives in `engine/` and is written to plug in the policy and value functions above:
- `engine/alphabeta.py`: Alpha-Beta pruning using `policy_func(pgn, top_k=...)` and `value_func(board)` at leaf nodes.
- `engine/mcts.py`: Monte-Carlo Tree Search using policy priors and batched value evaluation during expansion.

The key design is that both searchers are model-agnostic: they don't hardcode how moves are generated or how values are computed. They call the policy/value functions passed in.

## Runtime entrypoints

### RunPod serverless handler (main)
`handler.py` is the primary entrypoint. It:
- Loads the value model at import time (`Agent0 = ChessEvaluator('value_transformer/mini_value_6o4.pt')`).
- Routes requests by `event['input']['endpoint']`.
- Dispatches to one of:
  - `alphabeta-eval`
  - `mcts-3`
  - `transformer-move`

### FastAPI app (secondary compatibility)
`app.py` defines a FastAPI application (wrapped for AWS Lambda via `Mangum`), with endpoints that mirror the same search capabilities and uses the same policy/value components.

## API endpoints

### RunPod handler: `input.endpoint`
`handler.py` expects the RunPod event shape:
- `event['input']` contains the request fields.

Supported endpoints:

1. `alphabeta-eval`
   - Input:
     - `pgn` (string, required)
     - `depth` (int, optional; default in code: `3`)
   - Output:
     - `best_move` (string)
     - `score` (float)
     - `error` (string or null)

2. `mcts-3`
   - Input:
     - `pgn` (string, required)
     - `simulations` (int, optional; default in code: `800`)
   - Output:
     - `best_move` (string)
     - `score` (float)
     - `error` (string or null)

3. `transformer-move`
   - Input:
     - `pgn` (string, required)
   - Output:
     - `best_move` (string)
     - `error` (string or null)

### FastAPI routes (in `app.py`)
Implemented routes in `app.py`:
- `GET /healthz`
  - Returns `{"status": "ok"}`
- `POST /v1/api/alphabeta-eval`
  - Request model includes `pgn`, `depth`, and `agent`
- `POST /v1/api/mcts-3`
  - Request model includes `pgn`, `simulations`, and `agent`
- `POST /v1/api/transformer-move`
  - Request model includes `pgn` and `agent`

Note: `app.py` also contains additional code paths, but the routes above are the ones directly implemented with the transformer + search flow found in `engine/`.

## Training / evaluation scripts

The repo also includes training and analysis code under:
- `policy_transformer/train.py` and related scripts
- `value_transformer/train.py` and related scripts/
