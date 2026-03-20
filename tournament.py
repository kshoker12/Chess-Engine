"""
Single-endpoint Elo limiter test.

Plays a model (your FastAPI endpoint) against Stockfish with strength limiting:
  UCI_LimitStrength=true
  UCI_Elo=<target>

For each game, the model alternates colors (white/black). The selected endpoint is used
to pick the model move on model turns.

Usage example:
  python3 tournament_limited_elo_endpoint.py --difficulty transformer-move --uci-elo 2400 --games 20
"""

import math
import time
from typing import List, Optional

import chess
import chess.engine
import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8001"
DEFAULT_SF_PATH = "/opt/homebrew/bin/stockfish"

# =========================
# Hardcoded test constants
# =========================
# Pick exactly one endpoint at a time:
#   "transformer-move" | "alphabeta-eval" | "mcts-3"
DIFFICULTY = "transformer-move"

# Stockfish limiter target Elo:
UCI_ELO = 3190

# Number of games to play (alternating colors each game).
GAMES = 10

# FastAPI endpoint base URL.
BASE_URL = DEFAULT_BASE_URL

# Stockfish binary.
SF_PATH = DEFAULT_SF_PATH

# Which agent id to send to your evaluator (app.py uses Agent0/Agent1).
AGENT_ID = 0

# Move time for Stockfish (when it's playing).
TIME_PER_MOVE = 0.4

# API request timeout (seconds).
API_TIMEOUT_S = 600

# Print a heartbeat every N plies.
HEARTBEAT_EVERY_PLIES = 10

# Endpoint-specific overrides:
ALPHABETA_DEPTH = 4
MCTS_SIMULATIONS = 400

# Print extra stockfish move logging.
VERBOSE = False


def _moves_to_pgn(move_stack_uci: List[str]) -> str:
    """
    Convert a UCI move list into the SAN-based PGN string format your FastAPI endpoints parse.
    """
    if not move_stack_uci:
        return ""
    board = chess.Board()
    parts: List[str] = []
    for uci in move_stack_uci:
        move = chess.Move.from_uci(uci)
        san = board.san(move)
        if board.turn == chess.WHITE:
            parts.append(f"{board.fullmove_number}. {san}")
        else:
            parts.append(san)
        board.push(move)
    return " ".join(parts)


def _coerce_uci_to_legal(board: chess.Board, uci: str) -> str:
    """
    If API returns empty/invalid/illegal move, fall back to a legal move.

    Also handles the common promotion format mismatch:
    - returned `e7e8` (len=4) -> try `e7e8q` if legal.
    """
    legal = [m.uci() for m in board.legal_moves]
    if not legal:
        return ""

    if isinstance(uci, str) and uci in legal:
        return uci

    if isinstance(uci, str) and len(uci) == 4:
        uci_q = uci + "q"
        if uci_q in legal:
            return uci_q

    if not uci:
        print("[tournament] Empty move from API; using fallback legal move.")
    else:
        print(f"[tournament] Illegal/invalid move from API: {uci!r}; using fallback legal move.")
    return legal[0]


def _post_best_move(base_url: str, difficulty: str, payload: dict, timeout_s: float) -> str:
    endpoints = {
        "transformer-move": f"{base_url}/v1/api/transformer-move",
        "alphabeta-eval": f"{base_url}/v1/api/alphabeta-eval",
        "mcts-3": f"{base_url}/v1/api/mcts-3",
    }
    if difficulty not in endpoints:
        raise ValueError(f"Unknown difficulty {difficulty}. Expected one of: {list(endpoints.keys())}")

    t0 = time.time()
    resp = requests.post(endpoints[difficulty], json=payload, timeout=timeout_s)
    dt = time.time() - t0
    data = resp.json()

    best_move = data.get("best_move", "")
    err = data.get("error")
    if err:
        print(f"[{difficulty}] endpoint error: {err}")
    print(f"[{difficulty}] API move in {dt:.2f}s -> {best_move}")
    return best_move


def _model_is_to_play(board: chess.Board, model_is_white: bool) -> bool:
    return (board.turn == chess.WHITE) == model_is_white


def _approx_elo_from_score(stockfish_elo: float, score: float) -> float:
    """
    Approximate model Elo from score vs a known opponent Elo.
    score is interpreted as expected score in [0,1]:
      score = (wins + 0.5*draws)/N

    Uses:
      model_elo = X + 400*log10(score/(1-score))

    Clamp to avoid log10(0).
    """
    eps = 1e-6
    s = min(1.0 - eps, max(eps, score))
    return stockfish_elo + 400.0 * math.log10(s / (1.0 - s))


def run() -> None:
    games = GAMES
    sf_elo = float(UCI_ELO)
    model_agent = int(AGENT_ID)

    print(f"Difficulty: {DIFFICULTY}")
    print(f"Games: {games}")
    print(f"Stockfish strength limit: UCI_Elo={sf_elo} (UCI_LimitStrength=true)")
    print(f"Base URL: {BASE_URL}")
    print(f"Stockfish path: {SF_PATH}")
    print(f"Time per move: {TIME_PER_MOVE}s")
    if VERBOSE:
        print("Verbosity: on")

    wins = 0
    draws = 0
    losses = 0

    for g in range(games):
        # Alternate model colors.
        model_is_white = (g % 2 == 0)

        board = chess.Board()
        move_stack_uci: List[str] = []
        pgn_parts: List[str] = []

        game_start_t = time.time()

        print(f"\n[{DIFFICULTY}] Game {g+1}/{games} start | model_color={'W' if model_is_white else 'B'}")

        plies = 0
        heartbeat_next = HEARTBEAT_EVERY_PLIES

        with chess.engine.SimpleEngine.popen_uci(SF_PATH) as engine:
            # UCI strength limiting.
            engine.configure({"UCI_LimitStrength": "true", "UCI_Elo": str(int(sf_elo))})

            while not board.is_game_over():
                plies = len(board.move_stack) + 1

                if _model_is_to_play(board, model_is_white):
                    pgn = " ".join(pgn_parts)

                    if DIFFICULTY == "transformer-move":
                        payload = {"pgn": pgn, "agent": model_agent}
                    elif DIFFICULTY == "alphabeta-eval":
                        payload = {"pgn": pgn, "depth": ALPHABETA_DEPTH, "agent": model_agent}
                    elif DIFFICULTY == "mcts-3":
                        payload = {"pgn": pgn, "simulations": MCTS_SIMULATIONS, "agent": model_agent}
                    else:
                        payload = {"pgn": pgn, "agent": model_agent}

                    # API returns UCI string.
                    move_uci = _post_best_move(
                        base_url=BASE_URL,
                        difficulty=DIFFICULTY,
                        payload=payload,
                        timeout_s=API_TIMEOUT_S,
                    )
                    move_uci = _coerce_uci_to_legal(board, move_uci)
                    if not move_uci:
                        break
                    move = chess.Move.from_uci(move_uci)
                else:
                    t0 = time.time()
                    result = engine.play(board, chess.engine.Limit(time=TIME_PER_MOVE))
                    dt = time.time() - t0
                    move = result.move
                    if VERBOSE:
                        mv = move.uci() if move else "None"
                        print(f"[stockfish] move in {dt:.2f}s -> {mv}")

                san = board.san(move)

                # Update SAN-part PGN string in the same format your app parses.
                # `board.turn` is the mover's turn before pushing.
                if board.turn == chess.WHITE:
                    pgn_parts.append(f"{board.fullmove_number}. {san}")
                else:
                    pgn_parts.append(san)

                board.push(move)
                move_stack_uci.append(move.uci())

                if HEARTBEAT_EVERY_PLIES > 0 and len(board.move_stack) >= heartbeat_next:
                    fullmove = board.fullmove_number
                    turn = "W" if board.turn == chess.WHITE else "B"
                    print(
                        f"[{DIFFICULTY}] progress: game {g+1}/{games} "
                        f"plies={len(board.move_stack)} turn={turn} fullmove={fullmove}"
                    )
                    heartbeat_next += HEARTBEAT_EVERY_PLIES

        outcome = board.outcome()
        result = outcome.result() if outcome else "1/2-1/2"

        dt_game = time.time() - game_start_t

        # Score accounting from model perspective.
        if result == "1-0":
            model_won = model_is_white
        elif result == "0-1":
            model_won = not model_is_white
        else:
            model_won = False

        if result == "1-0" or result == "0-1":
            if model_won:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1

        final_plies = len(board.move_stack)
        print(f"[{DIFFICULTY}] Game {g+1}/{games} end: {result} | model_color={'W' if model_is_white else 'B'} | plies={final_plies} | {dt_game:.2f}s")

    score = (wins + 0.5 * draws) / games
    model_elo = _approx_elo_from_score(sf_elo, score)

    print("\n=== Summary ===")
    print(f"Difficulty: {DIFFICULTY}")
    print(f"Stockfish Elo target: {sf_elo}")
    print(f"Model score: {score:.3f} (W={wins} D={draws} L={losses})")
    print(f"Approx model Elo: {model_elo:.0f}")


def main() -> None:
    # Minimal validation so failures are obvious.
    if DIFFICULTY not in ("transformer-move", "alphabeta-eval", "mcts-3"):
        raise ValueError(f"Invalid DIFFICULTY: {DIFFICULTY}")
    run()


if __name__ == "__main__":
    main()

