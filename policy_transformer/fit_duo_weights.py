"""
Fit the base + sidekick logit weights on held-out validation positions, the way
Asymmetric Duos does (Zhou et al., 2025, Sec. 3):

    f_duo(x) = T_large * f_base(x) + T_small * f_sidekick(x)

with the two temperatures chosen to minimise negative log likelihood on a
validation set. Here the "classes" are the legal moves, the label is Stockfish's
best move (0.1 s), and the softmax is over legal moves only.

Two things are reported:

1. The Duos fit: (T_large, T_small) minimising validation NLL directly. NLL is
   convex in the temperatures, so L-BFGS finds the global optimum.
2. A grid over the mixing weight w in {0, 0.1, ..., 1}, logits = (1-w)*base + w*sidekick.
   Top-k depends only on w (a common scale never changes the ranking), so this
   grid is exactly the family of rankings the Duo can produce. For each w the
   overall scale is fitted by NLL, so NLL values are comparable across w. The
   bootstrap shows how stable the NLL-optimal w is across resampled validation sets.

Run from the repo root (only inference and a cached Stockfish labelling are needed):

    python policy_transformer/fit_duo_weights.py

The weights are saved to OUT_PATH for the top-k comparison script.
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
GRID = [i / 10 for i in range(11)]   # w = 0 is base only, w = 1 is sidekick only
TOPK = (1, 3, 5, 7, 9)
N_BOOT = 1000
BOOT_SEED = 0


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


def fit_temperatures(Lb, Ls, mask, y, fixed_w=None):
    """Minimise validation NLL. fixed_w=None fits (T_large, T_small) freely, as in Duos.
    With fixed_w, only a common scale s is fitted: T_large = s(1-w), T_small = s w."""
    if fixed_w is None:
        p = torch.tensor([1.0, 0.1], dtype=torch.float64, requires_grad=True)
        combine = lambda: p[0] * Lb + p[1] * Ls
    else:
        p = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        combine = lambda: p[0] * ((1 - fixed_w) * Lb + fixed_w * Ls)
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
        return p.detach().tolist(), per_position_nll(C, mask, y), C


def main():
    positions = ev.load_or_build_positions(VAL_PGN, VAL_POSITIONS, N_VAL, VAL_SEED)
    d = load_or_collect_logits(positions)
    Lb, Ls, mask, y = d["base"].double(), d["sidekick"].double(), d["mask"], d["y"]
    n = len(y)
    print(f"\n{n} validation positions (Stockfish {ev.STOCKFISH_TIME}s best move)")

    # Reference points: uncalibrated base, and base with its own temperature fitted.
    raw_nll = per_position_nll(Lb, mask, y).mean().item()
    print(f"Base alone, raw logits (T=1):      NLL {raw_nll:.4f}")

    # Grid over w, with the scale fitted by NLL for each w.
    rows, losses = [], []
    for w in GRID:
        (s,), nll_i, C = fit_temperatures(Lb, Ls, mask, y, fixed_w=w)
        hits = topk_hits(C, mask, y)
        rows.append(dict(w=w, scale=s, T_large=s * (1 - w), T_small=s * w,
                         nll=nll_i.mean().item(), **{f"top{k}": hits[k] for k in TOPK}))
        losses.append(nll_i)
    L = torch.stack(losses, dim=1)   # [N, len(GRID)] per-position NLL at each w's fitted scale

    print("\nGrid over w  (logits = scale * ((1-w) * base + w * sidekick); scale fitted by NLL)")
    print(f"{'w':>4} {'scale':>7} {'T_large':>8} {'T_small':>8} {'NLL':>8}"
          + "".join(f"{'top-' + str(k):>8}" for k in TOPK))
    best = min(range(len(GRID)), key=lambda i: rows[i]["nll"])
    for i, r in enumerate(rows):
        tag = {0: "  <- base only", len(GRID) - 1: "  <- sidekick only"}.get(i, "")
        mark = "  <- best NLL" if i == best else ""
        print(f"{r['w']:>4.1f} {r['scale']:>7.3f} {r['T_large']:>8.3f} {r['T_small']:>8.3f} {r['nll']:>8.4f}"
              + "".join(f"{100 * r[f'top{k}']:>7.2f}%" for k in TOPK) + tag + mark)

    # Duos fit: both temperatures free.
    (tl, ts), nll_i, C = fit_temperatures(Lb, Ls, mask, y)
    w_duo = ts / (tl + ts)
    hits = topk_hits(C, mask, y)
    print(f"\nDuos fit (T_large, T_small free): T_large {tl:.4f}, T_small {ts:.4f}, "
          f"implied w = T_small/(T_large+T_small) = {w_duo:.3f}, NLL {nll_i.mean().item():.4f}")
    print("  top-k at the fitted weights: " + ", ".join(f"top-{k} {100 * hits[k]:.2f}%" for k in TOPK))

    # Stability of the NLL-optimal grid weight under resampling of the validation set.
    g = torch.Generator().manual_seed(BOOT_SEED)
    counts = torch.zeros(len(GRID), dtype=torch.long)
    for _ in range(N_BOOT):
        b = torch.randint(0, n, (n,), generator=g)
        counts[L[b].mean(0).argmin()] += 1
    print(f"\nBootstrap ({N_BOOT} resamples): how often each w has the lowest NLL")
    print("  " + "  ".join(f"w={w:.1f}: {100 * c / N_BOOT:.0f}%" for w, c in zip(GRID, counts.tolist()) if c))

    out = dict(
        n_val=n, val_pgn=os.path.basename(VAL_PGN), stockfish_time=ev.STOCKFISH_TIME,
        chosen=dict(w=rows[best]["w"], T_large=rows[best]["T_large"], T_small=rows[best]["T_small"],
                    rule="grid w with lowest validation NLL (scale fitted per w)"),
        duo_fit=dict(T_large=tl, T_small=ts, implied_w=w_duo, nll=nll_i.mean().item()),
        base_raw_nll=raw_nll, grid=rows,
        bootstrap={f"{w:.1f}": c / N_BOOT for w, c in zip(GRID, counts.tolist())},
    )
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nChosen w = {rows[best]['w']:.1f}. Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
