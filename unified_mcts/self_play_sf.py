#!/usr/bin/env python3
"""
Collect value + policy FT shards: unified vs Stockfish.

Unified: policy top-k, then Bernoulli P_OPTIMAL value-rerank vs temperature sample.
Stockfish: MultiPV softmax(-cp_loss / T_SF) — mostly optimal with temperature.

Value rows: |v_θ − y_SF| ≥ err_thresh on unified candidate children (+ random floor).
Policy rows: every SF ply (STM-white, single-target) plus optional SF-best at error boards.

Writes:
  <out>/value_ft/{value_X,value_R,value_Y,value_meta.json}
  <out>/policy_ft/{policy_hist,policy_board,policy_R,policy_y,policy_yv,policy_meta.json}

Example:
  python self_play_sf.py \\
    --ckpt ../ckpts/unified_phase_b_best.pt \\
    --vocab ../policy_transformer/vocab1.pkl \\
    --stockfish /opt/homebrew/bin/stockfish \\
    --out ./data_export/sp_shard_0 \\
    --shard-id 0 --games 200 --cores 4
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.engine
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm

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
from dataset import write_memmap
from inference import UnifiedInference

# Populated in worker_init (spawn-safe).
_G: Dict[str, Any] = {}


def eval_sf_white_pov(board: chess.Board, engine: chess.engine.SimpleEngine, depth: int) -> float:
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white()
    if score.is_mate():
        m = score.mate()
        cp = 10000.0 if m and m > 0 else -10000.0
    else:
        cp = float(score.score(mate_score=10000) or 0)
    return tanh_cp(cp)


def eval_sf_stm(board: chess.Board, engine: chess.engine.SimpleEngine, depth: int, mirrored: bool) -> float:
    return stm_value_from_white_pov(eval_sf_white_pov(board, engine, depth), mirrored)


def _relative_cp(info: dict) -> float:
    score = info["score"].relative
    if score.is_mate():
        m = score.mate()
        return 10000.0 if m and m > 0 else -10000.0
    return float(score.score(mate_score=10000) or 0)


def sf_sample_move(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int,
    multipv: int,
    t_sf: float,
    rng: np.random.RandomState,
) -> chess.Move:
    """Sample among MultiPV lines with softmax(-cp_loss / T_SF)."""
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    if not isinstance(infos, list):
        infos = [infos]
    moves, cps = [], []
    for inf in infos:
        pv = inf.get("pv") or []
        if not pv:
            continue
        mv = pv[0]
        if mv not in board.legal_moves:
            continue
        moves.append(mv)
        cps.append(_relative_cp(inf))
    if not moves:
        result = engine.play(board, chess.engine.Limit(depth=depth))
        return result.move
    cps = np.asarray(cps, dtype=np.float64)
    best = cps.max()
    loss = best - cps
    logits = -loss / max(float(t_sf), 1e-6)
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / (probs.sum() + 1e-12)
    return moves[int(rng.choice(len(moves), p=probs))]


def history_ids(prior: List[chess.Move], mirrored: bool, stoi: dict, pad_id: int) -> List[int]:
    ids = []
    for m in prior:
        tok = get_uci4_token(mirror_uci_string(m.uci()) if mirrored else m.uci())
        ids.append(int(stoi[tok]) if tok in stoi else pad_id)
    return ids


def policy_row(
    board: chess.Board,
    prior: List[chess.Move],
    move: chess.Move,
    stoi: dict,
    pad_id: int,
    block_size: int,
) -> Optional[dict]:
    pieces, rights, mirrored = encode_board_stm_white(board)
    token = get_uci4_token(mirror_uci_string(move.uci()) if mirrored else move.uci())
    if token not in stoi:
        return None
    hist = pad_history(history_ids(prior, mirrored, stoi, pad_id), block_size, pad_id)
    return {
        "hist": hist,
        "board": pieces,
        "rights": rights,
        "y": np.uint16(stoi[token]),
    }


def unified_pick(
    inf: UnifiedInference,
    board: chess.Board,
    moves: List[chess.Move],
    cands: List[Tuple[float, str]],
    p_optimal: float,
    temperature: float,
    rng: np.random.RandomState,
) -> chess.Move:
    if not cands:
        return list(board.legal_moves)[0]
    if rng.rand() < p_optimal:
        best_uci, best_s = cands[0][1], -1e9
        for _, uci in cands:
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                continue
            board.push(mv)
            s = -float(inf.evaluate_board(board))
            board.pop()
            if s > best_s:
                best_s, best_uci = s, uci
        return chess.Move.from_uci(best_uci)
    mv = inf.choose_move(
        board,
        moves,
        top_k=len(cands),
        temperature=max(temperature, 0.3),
        dirichlet_eps=0.0,
    )
    return mv or chess.Move.from_uci(cands[0][1])


def play_one_game(game_id: int) -> dict:
    inf: UnifiedInference = _G["inf"]
    engine: chess.engine.SimpleEngine = _G["engine"]
    stoi = _G["stoi"]
    pad_id = _G["pad_id"]
    cfg = _G["cfg"]
    rng = np.random.RandomState(int(cfg["seed"]) + int(game_id) * 10007)

    model_is_white = (game_id + int(cfg["shard_id"])) % 2 == 0
    board = chess.Board()
    moves: List[chess.Move] = []
    value_rows: List[Tuple[np.ndarray, np.ndarray, float]] = []
    policy_rows: List[dict] = []

    for _ply in range(int(cfg["max_plies"])):
        if board.is_game_over():
            break
        white_to_move = board.turn == chess.WHITE
        model_turn = (white_to_move and model_is_white) or ((not white_to_move) and (not model_is_white))

        if model_turn:
            cands = inf.predict_policy_board(board, moves, top_k=int(cfg["top_k"]))
            if not cands:
                break
            mined_here = False
            for _, uci in cands:
                mv = chess.Move.from_uci(uci)
                if mv not in board.legal_moves:
                    continue
                board.push(mv)
                pieces, rights, mirrored = encode_board_stm_white(board)
                v_theta = float(inf.evaluate_board(board))
                try:
                    y_sf = float(eval_sf_stm(board, engine, int(cfg["sf_depth"]), mirrored))
                except Exception:
                    y_sf = 0.0
                err = abs(v_theta - y_sf)
                keep = err >= float(cfg["err_thresh"]) or rng.rand() < float(cfg["floor_frac"])
                if keep:
                    value_rows.append((pieces, rights, y_sf))
                    mined_here = mined_here or err >= float(cfg["err_thresh"])
                board.pop()

            if cfg["policy_from_errors"] and mined_here:
                try:
                    sf_best = engine.play(board, chess.engine.Limit(depth=int(cfg["sf_depth"]))).move
                    if sf_best is not None:
                        row = policy_row(board, moves, sf_best, stoi, pad_id, int(cfg["block_size"]))
                        if row:
                            policy_rows.append(row)
                except Exception:
                    pass

            mv = unified_pick(
                inf, board, moves, cands, float(cfg["p_optimal"]), float(cfg["temperature"]), rng
            )
        else:
            try:
                mv = sf_sample_move(
                    board,
                    engine,
                    int(cfg["sf_depth"]),
                    int(cfg["multipv"]),
                    float(cfg["t_sf"]),
                    rng,
                )
            except Exception:
                break
            if mv is None:
                break
            row = policy_row(board, moves, mv, stoi, pad_id, int(cfg["block_size"]))
            if row:
                policy_rows.append(row)

        if mv not in board.legal_moves:
            break
        board.push(mv)
        moves.append(mv)

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        result = "draw"
    elif (outcome.winner == chess.WHITE) == model_is_white:
        result = "model"
    else:
        result = "sf"
    return {
        "value": value_rows,
        "policy": policy_rows,
        "result": result,
        "model_white": model_is_white,
        "plies": len(moves),
    }


def worker_init(ckpt: str, vocab: str, stockfish: str, cfg: dict, device: str):
    inf = UnifiedInference(ckpt, vocab, device=device)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    engine.configure({"Threads": 1})
    if cfg.get("sf_skill") is not None:
        try:
            engine.configure({"Skill Level": int(cfg["sf_skill"])})
        except Exception:
            pass
    with open(vocab, "rb") as f:
        stoi = pickle.load(f)
    _G["inf"] = inf
    _G["engine"] = engine
    _G["stoi"] = stoi
    _G["pad_id"] = int(stoi["|"])
    _G["cfg"] = cfg
    import atexit
    atexit.register(lambda: engine.quit() if engine else None)


def worker_play(game_id: int) -> dict:
    try:
        return play_one_game(game_id)
    except Exception as e:
        return {"value": [], "policy": [], "result": "error", "error": str(e), "plies": 0}


def dump_value(out_dir: str, rows: List[Tuple[np.ndarray, np.ndarray, float]], meta_extra: dict):
    os.makedirs(out_dir, exist_ok=True)
    n = len(rows)
    if n == 0:
        raise SystemExit(f"No value rows collected → {out_dir}")
    X = np.stack([r[0] for r in rows]).astype(np.uint8)
    R = np.stack([r[1] for r in rows]).astype(np.uint8)
    Y = np.asarray([r[2] for r in rows], dtype=np.float16)
    write_memmap(os.path.join(out_dir, "value_X.u8.memmap"), X)
    write_memmap(os.path.join(out_dir, "value_R.u8.memmap"), R)
    write_memmap(os.path.join(out_dir, "value_Y.f16.memmap"), Y)
    meta = {
        "N": n,
        "K": RIGHTS_DIM,
        "y_space": "tanh(cp/400)",
        "source": "self_play_sf",
        **meta_extra,
    }
    with open(os.path.join(out_dir, "value_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"value_ft N={n} → {out_dir}")


def dump_policy(out_dir: str, rows: List[dict], vocab: str, pad_id: int, meta_extra: dict):
    os.makedirs(out_dir, exist_ok=True)
    n = len(rows)
    if n == 0:
        raise SystemExit(f"No policy rows collected → {out_dir}")
    hist = np.stack([r["hist"] for r in rows]).astype(np.uint16)
    board = np.stack([r["board"] for r in rows]).astype(np.uint8)
    rights = np.stack([r["rights"] for r in rows]).astype(np.uint8)
    y = np.asarray([r["y"] for r in rows], dtype=np.uint16)
    yv = np.zeros(n, dtype=np.float16)
    write_memmap(os.path.join(out_dir, "policy_hist.u16.memmap"), hist)
    write_memmap(os.path.join(out_dir, "policy_board.u8.memmap"), board)
    write_memmap(os.path.join(out_dir, "policy_R.u8.memmap"), rights)
    write_memmap(os.path.join(out_dir, "policy_y.u16.memmap"), y)
    write_memmap(os.path.join(out_dir, "policy_yv.f16.memmap"), yv)
    meta = {
        "N": n,
        "K": RIGHTS_DIM,
        "block_size": int(hist.shape[1]),
        "vocab": os.path.abspath(vocab),
        "pad_id": pad_id,
        "has_policy_yv": True,
        "sf_labeled": True,
        "source": "self_play_sf",
        "sources": [{"name": "self_play", "start": 0, "end": n}],
        **meta_extra,
    }
    with open(os.path.join(out_dir, "policy_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"policy_ft N={n} → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--out", required=True, help="Shard root (writes value_ft/ and policy_ft/)")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None, help="Default 1000+shard_id")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--sf-depth", type=int, default=12)
    ap.add_argument("--sf-skill", type=int, default=None, help="Optional UCI Skill Level (0-20)")
    ap.add_argument("--multipv", type=int, default=4)
    ap.add_argument("--t-sf", type=float, default=60.0, help="SF MultiPV temperature in cp")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--p-optimal", type=float, default=0.7)
    ap.add_argument("--temperature", type=float, default=0.8, help="Unified explore temperature")
    ap.add_argument("--err-thresh", type=float, default=0.15)
    ap.add_argument("--floor-frac", type=float, default=0.10)
    ap.add_argument("--policy-from-errors", action="store_true", default=True)
    ap.add_argument("--no-policy-from-errors", action="store_false", dest="policy_from_errors")
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else 1000 + int(args.shard_id)
    random.seed(seed)
    np.random.seed(seed)

    with open(args.vocab, "rb") as f:
        stoi = pickle.load(f)
    pad_id = int(stoi["|"])

    cfg = {
        "shard_id": int(args.shard_id),
        "seed": int(seed),
        "max_plies": int(args.max_plies),
        "sf_depth": int(args.sf_depth),
        "sf_skill": args.sf_skill,
        "multipv": int(args.multipv),
        "t_sf": float(args.t_sf),
        "top_k": int(args.top_k),
        "p_optimal": float(args.p_optimal),
        "temperature": float(args.temperature),
        "err_thresh": float(args.err_thresh),
        "floor_frac": float(args.floor_frac),
        "policy_from_errors": bool(args.policy_from_errors),
        "block_size": int(args.block_size),
    }

    os.makedirs(args.out, exist_ok=True)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    value_all: List[Tuple[np.ndarray, np.ndarray, float]] = []
    policy_all: List[dict] = []
    wins_sf = wins_model = draws = errors = 0

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=int(args.cores),
        initializer=worker_init,
        initargs=(args.ckpt, args.vocab, args.stockfish, cfg, args.device),
    ) as pool:
        for res in tqdm(
            pool.imap_unordered(worker_play, range(int(args.games))),
            total=int(args.games),
            desc=f"sp_shard_{args.shard_id}",
        ):
            value_all.extend(res.get("value") or [])
            policy_all.extend(res.get("policy") or [])
            r = res.get("result")
            if r == "sf":
                wins_sf += 1
            elif r == "model":
                wins_model += 1
            elif r == "draw":
                draws += 1
            else:
                errors += 1

    extra = {
        "sf_depth": args.sf_depth,
        "games": args.games,
        "shard_id": args.shard_id,
        "seed": seed,
        "ckpt": os.path.abspath(args.ckpt),
        "wins_sf": wins_sf,
        "wins_model": wins_model,
        "draws": draws,
        "errors": errors,
        "p_optimal": args.p_optimal,
        "err_thresh": args.err_thresh,
        "t_sf": args.t_sf,
        "multipv": args.multipv,
    }
    dump_value(os.path.join(args.out, "value_ft"), value_all, extra)
    dump_policy(os.path.join(args.out, "policy_ft"), policy_all, args.vocab, pad_id, extra)
    print(
        f"Done shard={args.shard_id} games={args.games} "
        f"SF={wins_sf} model={wins_model} draws={draws} errors={errors}"
    )


if __name__ == "__main__":
    main()
