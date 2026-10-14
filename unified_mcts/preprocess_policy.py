#!/usr/bin/env python3
"""
Phase B preprocess: PGN games → single-target policy windows + boards.

Original PGN framing (transformer-dev: should_mirror = result == "0-1"):
  - 1-0  → keep game as-is; train winner (White) plies only
  - 0-1  → mirror entire game so winner becomes White; train those winner plies only
  - draw → keep game as-is; both sides (no winner); each row still STM-white
  - Every network input: side-to-move is White (board + history + target)

Stockfish labeling is OFF by default (Phase B ignores policy_yv).

Example (all games in each dir — omit --max-games-per-source):
  python preprocess_policy.py \\
    --pgn-dirs full:../data/games elite:../data/elitegames weak:../data/weaker \\
    --vocab ../policy_transformer/vocab1.pkl \\
    --out ./data_export/policy
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import chess
import chess.engine
import chess.pgn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_utils import (
    RIGHTS_DIM,
    encode_board_stm_white,
    get_uci4_token,
    mirror_uci_string,
    pad_history,
    stm_value_from_white_pov,
    tanh_cp,
)
from dataset import StreamingPolicyWriter


def list_pgn_files(root: str) -> List[str]:
    p = Path(root)
    if p.is_file() and p.suffix.lower() == ".pgn":
        return [str(p)]
    return sorted(str(x) for x in p.rglob("*.pgn"))


def parse_sources(specs: List[str]) -> List[Tuple[str, str]]:
    out = []
    for s in specs:
        if ":" not in s:
            raise ValueError(f"Expected name:path, got {s}")
        name, path = s.split(":", 1)
        out.append((name, path))
    return out


def eval_sf(board: chess.Board, engine: chess.engine.SimpleEngine, depth: int) -> float:
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white()
    if score.is_mate():
        m = score.mate()
        cp = 10000.0 if m and m > 0 else -10000.0
    else:
        cp = float(score.score(mate_score=10000))
    return tanh_cp(cp)


def history_ids_for_stm(
    prior_moves: List[chess.Move],
    mirrored: bool,
    stoi: dict,
    pad_id: int,
) -> List[int]:
    """Remirror all prior moves with the current STM (matches inference)."""
    hist_ids: List[int] = []
    for prev in prior_moves:
        uci = prev.uci()
        tok = get_uci4_token(mirror_uci_string(uci) if mirrored else uci)
        hist_ids.append(int(stoi[tok]) if tok in stoi else pad_id)
    return hist_ids


def winner_is_white_from_result(result: str) -> Optional[bool]:
    """Return True if White won, False if Black won, None if draw/unknown."""
    result = (result or "*").strip()
    if result == "1-0":
        return True
    if result == "0-1":
        return False
    return None


def process_game(
    game: chess.pgn.Game,
    stoi: dict,
    pad_id: int,
    block_size: int,
    engine: Optional[chess.engine.SimpleEngine],
    sf_depth: int,
    sf_every: int,
    max_plies: Optional[int],
    winner_only: bool = True,
    skip_draws: bool = False,
) -> List[dict]:
    """
    Original framing:
      1-0  → keep as-is; winner_only → White plies
      0-1  → mirror entire game (winner→White); winner_only → those plies
      draw → keep as-is; both sides (STM-white each ply)

    STM-white on a Black-win winner ply == tokens from mirroring the whole game
    then taking White's moves in that frame (verified vs transformer-dev rule).
    """
    result = game.headers.get("Result", "*")
    winner_white = winner_is_white_from_result(result)  # True / False / None(draw)
    # Original: should_mirror = (result == "0-1")
    if winner_only and winner_white is None and skip_draws:
        return []

    rows = []
    board = game.board()
    prior_moves: List[chess.Move] = []
    ply = 0
    for move in game.mainline_moves():
        if max_plies is not None and ply >= max_plies:
            break

        is_white_ply = ply % 2 == 0
        if winner_only and winner_white is not None:
            # Decisive games: loss only on winner. Black win → odd plies (then STM-mirror).
            keep = is_white_ply if winner_white else (not is_white_ply)
        else:
            # Draws (or --all-plies): keep game as-is, every ply
            keep = True

        if keep:
            pieces, rights, mirrored = encode_board_stm_white(board)
            uci = move.uci()
            token = get_uci4_token(mirror_uci_string(uci) if mirrored else uci)
            hist_ids = history_ids_for_stm(prior_moves, mirrored, stoi, pad_id)

            if token in stoi:
                y_pi = stoi[token]
                y_v = np.float16(0.0)
                do_sf = engine is not None and (ply % sf_every == 0)
                if do_sf:
                    try:
                        y_white = eval_sf(board, engine, sf_depth)
                        y_v = np.float16(stm_value_from_white_pov(y_white, mirrored))
                    except Exception:
                        y_v = np.float16(0.0)

                rows.append(
                    {
                        "hist": pad_history(hist_ids, block_size, pad_id),
                        "board": pieces,
                        "rights": rights,
                        "y": np.uint16(y_pi),
                        "yv": y_v,
                    }
                )

        prior_moves.append(move)
        board.push(move)
        ply += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pgn-dirs",
        nargs="+",
        required=True,
        help="name:path entries, e.g. full:../data/games elite:../data/elitegames",
    )
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--stockfish",
        default=None,
        help="Optional: path to stockfish. If omitted, policy_yv is zeros (Phase B default).",
    )
    ap.add_argument("--sf-depth", type=int, default=12)
    ap.add_argument("--sf-every", type=int, default=1, help="Label every k-th ply (only with --stockfish)")
    ap.add_argument("--max-games-per-source", type=int, default=None)
    ap.add_argument("--max-plies-per-game", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument(
        "--write-yv",
        action="store_true",
        help="Always write policy_yv.f16.memmap (zeros if no SF).",
    )
    ap.add_argument(
        "--all-plies",
        action="store_true",
        help="DEPRECATED escape hatch: emit both sides (not original convention).",
    )
    ap.add_argument(
        "--skip-draws",
        action="store_true",
        help="Drop draws/unknown results. Default: keep draws as-is (both sides, STM-white).",
    )
    args = ap.parse_args()

    winner_only = not args.all_plies
    skip_draws = bool(args.skip_draws)

    with open(args.vocab, "rb") as f:
        stoi = pickle.load(f)
    pad_id = int(stoi["|"])
    os.makedirs(args.out, exist_ok=True)

    engine = None
    if args.stockfish:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        print(f"Stockfish: {args.stockfish} depth={args.sf_depth} every={args.sf_every}")
    else:
        print("No --stockfish: writing hist/board/rights/y only")

    print(
        f"Policy convention: winner_only={winner_only} skip_draws={skip_draws} "
        f"(winner framed as White / STM-white)"
    )

    writer = StreamingPolicyWriter(args.out, args.block_size, RIGHTS_DIM, flush_every=50_000)
    sources_meta = []
    games_kept = games_skipped = 0

    try:
        for name, root in parse_sources(args.pgn_dirs):
            files = list_pgn_files(root)
            print(f"Source {name}: {len(files)} pgn files under {root}", flush=True)
            games_done = 0
            src_start = writer.n
            for fp in files:
                if args.max_games_per_source is not None and games_done >= args.max_games_per_source:
                    break
                with open(fp, encoding="utf-8", errors="ignore") as f:
                    while True:
                        if args.max_games_per_source is not None and games_done >= args.max_games_per_source:
                            break
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break
                        rows = process_game(
                            game,
                            stoi,
                            pad_id,
                            args.block_size,
                            engine,
                            args.sf_depth,
                            args.sf_every,
                            args.max_plies_per_game,
                            winner_only=winner_only,
                            skip_draws=skip_draws,
                        )
                        if not rows:
                            games_skipped += 1
                        else:
                            games_kept += 1
                        for row in rows:
                            writer.add(row["hist"], row["board"], row["rights"], row["y"], row["yv"])
                        games_done += 1
                        if games_done % 1000 == 0:
                            print(
                                f"  {name}: games={games_done} rows={writer.n + len(writer._y)} "
                                f"kept={games_kept} skipped={games_skipped}",
                                flush=True,
                            )
            writer.flush()
            sources_meta.append({"name": name, "start": src_start, "end": writer.n})
            print(f"Finished {name}: rows [{src_start}, {writer.n})", flush=True)
    finally:
        if engine is not None:
            engine.quit()

    n = writer.close()
    if n == 0:
        raise SystemExit("No rows produced (check Result headers / --include-draws)")

    # If no SF and user did not request yv, still keep zeros file (Phase B ignores it).
    meta = {
        "N": n,
        "K": RIGHTS_DIM,
        "block_size": args.block_size,
        "vocab": os.path.abspath(args.vocab),
        "pad_id": pad_id,
        "vocab_size": len(stoi),
        "sf_depth": args.sf_depth if engine is not None else None,
        "sf_every": args.sf_every if engine is not None else None,
        "sf_labeled": engine is not None,
        "has_policy_yv": True,
        "winner_only": winner_only,
        "skip_draws": skip_draws,
        "convention": "winner_as_white_stm",
        "sources": sources_meta,
        "games_kept": games_kept,
        "games_skipped": games_skipped,
        "y_space": "tanh(cp/400)" if engine is not None else "unused",
    }
    with open(os.path.join(args.out, "policy_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done → {args.out} N={n} kept_games={games_kept} skipped={games_skipped}", flush=True)
    print("sources:", sources_meta, flush=True)


if __name__ == "__main__":
    main()
