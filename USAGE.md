# Updated pgn_to_dataset.py - Now with Stockfish Evaluation

## What Changed

The `pgn_to_dataset.py` script now evaluates positions with Stockfish instead of using placeholder zeros.

## Key Changes

1. **Added Stockfish Engine Integration**
   - Imports `chess.engine` for Stockfish interface
   - Configurable Stockfish path via `STOCKFISH_PATH` env variable
   - Default: `/opt/homebrew/bin/stockfish`

2. **Added Evaluation Functions**
   - `clamp_cp()`: Converts Stockfish scores to centipawns from side-to-move perspective
   - `ensure_engine()`: Initializes and configures Stockfish
   - Handles mate scores and clamps to ±1000 centipawns

3. **Updated Evaluation Logic**
   - Previously: `y_list.append(0)`  # Placeholder
   - Now: Uses `engine.analyse()` to get real evaluations
   - Includes error handling with fallback to 0

## Usage

```bash
# Set Stockfish path (if not at default location)
export STOCKFISH_PATH=/path/to/stockfish

# Process PGN files
python pgn_to_dataset.py --pgn games/

# This will create:
# - dataset_X_best.npy (features)
# - dataset_y_best.npy (Stockfish evaluations, not zeros!)
```

## Configuration

Edit these constants in `pgn_to_dataset.py`:
- `ENGINE_TIME_MS = 300`  # Time per position evaluation
- `CP_CLAMP = 1000`       # Max/min centipawn values

## Testing

```bash
# Test Stockfish integration
python test_stockfish_integration.py

# Test the generated dataset
python test_pgn_processing.py
```

## Next Steps

After regenerating the dataset with Stockfish evaluations:

```bash
# Train the model on real evaluations
python train_eval.py --prefix dataset_best --out sk_eval.joblib
```

The model will now learn from real Stockfish evaluations instead of zeros!
