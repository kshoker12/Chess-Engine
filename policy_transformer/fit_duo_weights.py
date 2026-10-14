"""
Fit the base + sidekick temperatures on held-out validation positions, as in
Asymmetric Duos (Zhou et al., 2025, Sec. 3):

    f_duo(x) = T_large * f_base(x) + T_small * f_sidekick(x)

(T_large, T_small) are chosen to minimise negative log likelihood on a validation
set, with L-BFGS. NLL is convex in the temperatures, so the optimum is global.
Here the "classes" are the legal moves, the label is Stockfish's best move (0.1 s),
and the softmax is over legal moves only. As in the paper, the base model alone is
also temperature-scaled (T_small = 0) as the single-model reference; a single
temperature never changes its ranking, so its top-k is unaffected.

Run from the repo root (only inference and a cached Stockfish labelling are needed):

    python policy_transformer/fit_duo_weights.py

The temperatures are saved to OUT_PATH for the top-k comparison script.
"""
import json
import os
import sys

import chess
import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import eval_topk_stockfish as ev

# ── Validation data (a month not used in training and not the test month) ──
VAL_PGN = os.path.join(ROOT_DIR, "data", "eval_holdout", "lichess_elite_2023-03.pgn")
VAL_POSITIONS = os.path.join(ROOT_DIR, "data", "eval_holdout", "val_positions_10k_sf0.1.jsonl")
VAL_LOGITS = os.path.join(ROOT_DIR, "data", "eval_holdout", "val_logits_10k.pt")
OUT_PATH = os.path.join(ROOT_DIR, "data", "eval_holdout", "duo_weights.json")
N_VAL = 10_000
VAL_SEED = 456          # differs from the test seed (123); the PGN differs too
TOPK = (1, 3, 5, 7, 9)


# ── Step 1: legal-move logits of both models on the validation positions ──
def collect_logits(positions):
    """Padded [N, M] logits for base and sidekick over the legal moves, a validity
    mask, and the index of Stockfish's move in each position's legal-move list."""
    device = ev.pick_device()
    base, base_stoi = ev.load_model("base", device)
    side, side_stoi = ev.load_model("sidekick", device)

    rows_b, rows_s, ys = [], [], []
    for pos in tqdm(positions, desc="Model logits on validation"):
        board = chess.Board()
        for u in pos["moves"]:
            board.push_uci(u)
        ucis, lb = ev.legal_move_logits(base, base_stoi, device, pos["moves"], board)
        ucis_s, ls = ev.legal_move_logits(side, side_stoi, device, pos["moves"], board)
        assert ucis == ucis_s, "legal-move order differs between models"
        rows_b.append(lb)
        rows_s.append(ls)
        ys.append(ucis.index(pos["best"]))

    n, m = len(positions), max(len(r) for r in rows_b)
    Lb, Ls = torch.zeros(n, m), torch.zeros(n, m)
    mask = torch.zeros(n, m, dtype=torch.bool)
    for i, (rb, rs) in enumerate(zip(rows_b, rows_s)):
        k = len(rb)
        Lb[i, :k], Ls[i, :k], mask[i, :k] = torch.tensor(rb), torch.tensor(rs), True
    for name, L in (("base", Lb), ("sidekick", Ls)):
        bad = int(torch.isinf(L).sum())
        if bad:
            print(f"WARNING: {bad} legal moves have no token in the {name} vocab; clamped to -1e4")
            L.clamp_(min=-1e4)
    return dict(base=Lb, sidekick=Ls, mask=mask, y=torch.tensor(ys))


def load_or_collect_logits(positions):
    if os.path.exists(VAL_LOGITS):
        data = torch.load(VAL_LOGITS)
        if len(data["y"]) == len(positions):
            print(f"Loaded cached logits from {VAL_LOGITS}")
            return data
    data = collect_logits(positions)
    torch.save(data, VAL_LOGITS)
    print(f"Saved logits to {VAL_LOGITS}")
    return data


# ── Step 2: metrics and fits on the cached logits ──
def masked(C, mask):
    return C.masked_fill(~mask, float("-inf"))


def per_position_nll(C, mask, y):
    return F.cross_entropy(masked(C, mask), y, reduction="none")


def topk_hits(C, mask, y, ks=TOPK):
    """Top-k hit rates with the same tie-break as the eval script (earlier legal move first)."""
    C = masked(C, mask)
    cy = C.gather(1, y[:, None])
    idx = torch.arange(C.shape[1])[None, :]
    rank = (C > cy).sum(1) + ((C == cy) & (idx < y[:, None])).sum(1)
    return {k: (rank < k).double().mean().item() for k in ks}


def fit_temperatures(Lb, Ls, mask, y, base_only=False):
    """Minimise validation NLL over (T_large, T_small), or over T_large alone
    (T_small = 0) for the temperature-scaled base reference."""
    p = torch.tensor([1.0, 0.1], dtype=torch.float64, requires_grad=True)
    combine = (lambda: p[0] * Lb) if base_only else (lambda: p[0] * Lb + p[1] * Ls)
    opt = torch.optim.LBFGS([p], lr=1.0, max_iter=500, tolerance_grad=1e-10,
                            tolerance_change=1e-14, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = per_position_nll(combine(), mask, y).mean()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        C = combine()
        t_large, t_small = p[0].item(), (0.0 if base_only else p[1].item())
        return t_large, t_small, per_position_nll(C, mask, y).mean().item(), topk_hits(C, mask, y)


def main():
    positions = ev.load_or_build_positions(VAL_PGN, VAL_POSITIONS, N_VAL, VAL_SEED)
    d = load_or_collect_logits(positions)
    Lb, Ls, mask, y = d["base"].double(), d["sidekick"].double(), d["mask"], d["y"]
    print(f"\n{len(y)} validation positions (Stockfish {ev.STOCKFISH_TIME}s best move)\n")

    raw_nll = per_position_nll(Lb, mask, y).mean().item()
    tb, _, nll_b, hits_b = fit_temperatures(Lb, Ls, mask, y, base_only=True)
    tl, ts, nll_d, hits_d = fit_temperatures(Lb, Ls, mask, y)

    print(f"{'':32}{'T_large':>9}{'T_small':>9}{'val NLL':>9}" + "".join(f"{'top-' + str(k):>8}" for k in TOPK))
    for name, a, b, nll, h in (
        ("base, raw logits", 1.0, 0.0, raw_nll, None),
        ("base, temperature-scaled", tb, 0.0, nll_b, hits_b),
        ("base + sidekick (Duo)", tl, ts, nll_d, hits_d),
    ):
        tops = "".join(f"{100 * h[k]:>7.2f}%" for k in TOPK) if h else " " * 8 * len(TOPK)
        print(f"{name:32}{a:>9.4f}{b:>9.4f}{nll:>9.4f}{tops}")
    print(f"\nT_small / T_large = {ts / tl:.3f}")

    out = dict(
        n_val=len(y), val_pgn=os.path.basename(VAL_PGN), stockfish_time=ev.STOCKFISH_TIME,
        duo=dict(T_large=tl, T_small=ts, val_nll=nll_d, val_topk=hits_d),
        base_scaled=dict(T_large=tb, val_nll=nll_b, val_topk=hits_b),
        base_raw_nll=raw_nll,
    )
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
