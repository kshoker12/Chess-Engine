# Neural Chess Engine with AWS Lambda Deployment

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![AWS Lambda](https://img.shields.io/badge/AWS-Lambda-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green.svg)
![Docker](https://img.shields.io/badge/Docker-Container-blue.svg)

A production-grade chess engine with **three evaluation models** (PyTorch, scikit-learn, Stockfish) combined with classical alpha-beta search algorithms, deployed as a serverless API on AWS Lambda with architecture optimization.

## Architecture Overview

This project implements a hybrid chess engine that leverages machine learning for position evaluation while maintaining the tactical precision of classical search algorithms. The system follows a three-tier architecture:

```
Data Generation → Model Training → Inference Engine
     ↓                ↓                ↓
Stockfish Labels → ML Pipeline → Alpha-Beta Search
```

The engine uses a **hybrid evaluation approach** that blends neural network predictions (80%) with classical material and positional heuristics (20%), providing both strategic depth and tactical accuracy.

## Technical Deep Dive

### Data Generation Pipeline

The training data is generated using Stockfish integration and real game analysis for high-quality position labeling:

- **Dataset Scale**: 1,000,000+ positions from diverse game sources
- **Data Sources**: Kaggle datasets + PGN analysis from Grandmaster games
- **Feature Encoding**: 776-dimensional vectors representing board state (12 pieces × 64 squares + castling, move clocks)
- **Label Generation**: Stockfish depth-15 evaluations converted to centipawn scores
- **Game Analysis**: Includes games from Carlsen, Caruana, Nakamura, Fischer, Anand

### Evaluation Models

The engine supports **three evaluation models** selectable via the `difficulty` parameter:

#### 1. Easy Mode - PyTorch Neural Network
- **Framework**: PyTorch with batch normalization and dropout
- **Architecture**: MLP with hidden layers [512, 264, 64, 32]
- **Features**: GPU acceleration support, flexible training
- **Speed**: Fastest inference
- **Use Case**: Quick games, rapid responses

#### 2. Medium Mode - scikit-learn Neural Network (Default)
- **Framework**: scikit-learn MLPRegressor
- **Architecture**: MLP with hidden layers [512, 264, 64, 32]
- **Preprocessing**: StandardScaler for feature normalization
- **Target Transformation**: TransformedTargetRegressor for centipawn prediction
- **Training Data**: 1,000,000 positions with 80/10/10 train/test split
- **Optimization**: Early stopping, Adam solver, L2 regularization (alpha=5e-5)
- **Speed**: Medium, balanced performance
- **Use Case**: Production default, balanced accuracy/speed

#### 3. Hard Mode - Stockfish Engine
- **Engine**: Stockfish 16 compiled from source
- **Evaluation**: Direct engine analysis (depth 8)
- **Speed**: Slowest but most accurate
- **Use Case**: Best move analysis, critical positions

**Model Selection:**
```python
# All models use the same 776-dimensional input features
find_best_move(fen, difficulty="easy")    # PyTorch
find_best_move(fen, difficulty="medium")  # scikit-learn (default)
find_best_move(fen, difficulty="hard")    # Stockfish
```



```python
# ML Pipeline Architecture
pipe = Pipeline([
    ("xscaler", StandardScaler()),
    ("mlp", TransformedTargetRegressor(
        regressor=MLPRegressor(
            activation="relu",
            solver="adam",
            early_stopping=True
        ),
        transformer=StandardScaler()
    ))
])
```

### Search Engine Implementation

The core engine implements a highly optimized alpha-beta pruning algorithm with multiple enhancements:

#### Core Search Features
- **Transposition Tables**: Position caching for repeated sub-trees
- **Iterative Deepening**: Progressive depth search with aspiration windows
- **Null-Move Pruning**: Efficient pruning for non-tactical positions
- **Late Move Reductions (LMR)**: Reduced search depth for unlikely moves
- **Quiescence Search**: Tactical stability through capture-only extensions
- **Blunder Prevention**: Advanced filtering to prevent queen sacrifices and material blunders
- **Center Control**: Opening phase encouragement for strategic development

#### Move Ordering Heuristics
- **Killer Move Heuristic**: Prioritize moves that caused beta cutoffs
- **History Heuristic**: Track move success rates across positions
- **MVV-LVA Ordering**: Most Valuable Victim - Least Valuable Attacker
- **Static Exchange Evaluation (SEE)**: Prune obviously bad captures and hanging pieces
- **Tactical Blunder Filtering**: Prevent queen/rook sacrifices and material losses

#### Hybrid Evaluation Strategy
```python
# Blended evaluation: 80% neural network + 20% classical
blend = 0.8 + 0.02 * skill_level
cp = blend * neural_eval + (1.0 - blend) * material_eval

# Enhanced with blunder prevention
if material_loss < -50:  # Filter obvious blunders
    prune_move()
```

#### Blunder Prevention System
The engine includes sophisticated tactical safeguards:
- **Hanging Piece Detection**: Identify pieces moving into unprotected squares
- **Material Loss Analysis**: 1-ply lookahead to detect worst-case material loss
- **Static Exchange Evaluation**: Prune bad captures before search
- **High-Value Piece Protection**: Special handling for Queen/Rook sacrifices
- **Conservative Filtering**: Only filter moves losing >50 centipawns to preserve legitimate sacrifices

### Production Optimizations

#### Lambda-Specific Optimizations
- **Lazy Loading**: Deferred imports reduce cold start times by ~60%
- **Memory Efficiency**: Model caching prevents repeated loading
- **ARM64 Architecture**: Native Lambda performance optimization
- **Joblib Configuration**: Multiprocessing disabled for Lambda compatibility

#### Performance Enhancements
- **Containerized Deployment**: Docker-based dependency management
- **Legacy Docker Builder**: Ensures Lambda-compatible manifest format
- **Model Flexibility**: Three evaluation models with automatic fallback mechanisms
- **Resource Optimization**: Memory and timeout tuning for serverless constraints

## AWS Deployment Architecture

### Infrastructure Stack

```
Internet → API Gateway → Lambda (ARM64) → ECR Container
```

- **API Gateway**: RESTful endpoint management and request routing
- **AWS Lambda**: Serverless compute with ARM64 Graviton2 processors
- **ECR (Elastic Container Registry)**: Container image storage and versioning
- **FastAPI + Mangum**: ASGI adapter for Lambda integration

### Deployment Strategy

## API Interface

### RESTful Endpoints

#### Health Check
```bash
GET /healthz
# Response: {"status": "ok"}
```

#### Move Generation
```bash
POST /v1/api/move
Content-Type: application/json

{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "max_depth": 4,
  "difficulty": "medium"
}

# Response:
{
  "best_move": "e2e4",
  "score_cp": 25
}
```

**Difficulty Levels:**
- **`easy`**: Uses PyTorch neural network (fastest, good for quick games)
- **`medium`**: Uses scikit-learn model (balanced, recommended default)
- **`hard`**: Uses Stockfish evaluation (most accurate, slowest)

## Project Structure

```
├── data_gen.py          # Stockfish-based dataset generation
├── features.py          # 776-dim feature encoding
├── train_eval.py        # ML model training pipeline (scikit-learn)
├── train_pytorch.py     # PyTorch model training
├── pytorch_model.py     # PyTorch model inference
├── eval_model.py        # Lazy-loaded scikit-learn model
├── engine.py            # Alpha-beta search with optimizations
├── app.py               # FastAPI + Lambda handler
├── Dockerfile           # ARM64 Lambda container
├── requirements.txt     # Python dependencies
├── sk_eval.joblib       # scikit-learn model (medium)
├── chess_eval_pytorch.pt # PyTorch model (easy)
├── dataset_*.npy        # Training data arrays (1M positions)
├── games/               # PGN files from top GMs
└── dataset_meta.json    # Dataset metadata
```

## Key Technical Highlights

### Innovation Areas
- **Triple-Model Architecture**: PyTorch, scikit-learn, and Stockfish evaluation models with seamless switching
- **Hybrid AI Architecture**: Combines neural evaluation with classical search algorithms
- **Large-Scale Training**: 1,000,000+ position dataset from diverse sources
- **Blunder Prevention**: Advanced tactical safeguards prevent queen sacrifices and material blunders
- **Model Flexibility**: Difficulty-based model selection with automatic fallback mechanisms
- **Serverless Optimization**: Production-grade optimizations for AWS Lambda deployment
- **Scalable Design**: Efficient handling of cold starts and resource constraints
- **Professional API**: RESTful design with comprehensive error handling

### Engineering Excellence
- **Production Deployment**: Containerized serverless architecture
- **Performance Optimization**: ARM64 native execution with lazy loading
- **Code Quality**: Modular design with clear separation of concerns
- **Monitoring**: Health checks and debugging endpoints for operational visibility
- **Data Quality**: 1M+ positions from real GM games and Stockfish analysis

### Technical Depth
- **Advanced Search**: Multi-layered alpha-beta optimizations with blunder prevention
- **Multiple ML Models**: PyTorch, scikit-learn, and Stockfish evaluation engines
- **Machine Learning**: Custom feature engineering on 776-dimensional state encoding
- **Model Swapping**: Runtime model selection via difficulty parameter
- **Tactical Safety**: SEE, hanging piece detection, and material loss analysis
- **Cloud Architecture**: AWS-native deployment with multi-stage Docker builds
- **DevOps Integration**: Docker-based CI/CD with ECR integration and ARM64 optimization

## Usage

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install PyTorch for PyTorch model support
pip install torch

# Run FastAPI server
uvicorn app:app --reload

# Test endpoints with difficulty levels
curl -X POST "http://localhost:8000/v1/api/move" \
     -H "Content-Type: application/json" \
     -d '{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "difficulty": "medium"}'

# Test different difficulty levels
curl -X POST "http://localhost:8000/v1/api/move" \
     -H "Content-Type: application/json" \
     -d '{"fen": "...", "difficulty": "easy"}'   # PyTorch
curl -X POST "http://localhost:8000/v1/api/move" \
     -H "Content-Type: application/json" \
     -d '{"fen": "...", "difficulty": "medium"}'  # scikit-learn (default)
curl -X POST "http://localhost:8000/v1/api/move" \
     -H "Content-Type: application/json" \
     -d '{"fen": "...", "difficulty": "hard"}'    # Stockfish
```

### AWS Deployment

The Dockerfile includes all three models:

```bash
# Build ARM64 container (includes Stockfish compilation from source)
docker build --platform linux/arm64 -t chess-ai-engine .

# The build process:
# 1. Builder stage: Compiles Stockfish 16 from source on Amazon Linux 2023
# 2. Final stage: Copies Stockfish binary and includes:
#    - sk_eval.joblib (medium difficulty)
#    - chess_eval_pytorch.pt (easy difficulty - optional)
#    - /usr/local/bin/stockfish (hard difficulty)

# Push to ECR
docker push <account>.dkr.ecr.<region>.amazonaws.com/chess-ai-backend:latest

# Deploy to Lambda
aws lambda update-function-code \
  --function-name chess-ai-engine \
  --image-uri <account>.dkr.ecr.<region>.amazonaws.com/chess-ai-backend:latest
```

**Fallback Behavior:**
- If PyTorch model missing → falls back to scikit-learn
- If Stockfish not found → falls back to scikit-learn
- Ensures engine always has at least one working model

---

*This project demonstrates expertise in machine learning, classical AI algorithms, cloud architecture, and production engineering - showcasing the intersection of modern ML techniques with traditional computer science fundamentals.*