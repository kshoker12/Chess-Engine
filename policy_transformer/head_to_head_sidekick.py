"""
Head-to-head MCTS: production search vs sidekick-guided search.

Both engines use the same value network and the production MCTS (engine/mcts.py:
PUCT with c = 1.10, top-10 policy priors, softmax tau = 1, most-visited move,
checkmate value 1.03).

    prod      base policy at the root and at every expansion, 400 simulations
    sidekick  base policy at the root only, sidekick policy at every other
              expansion, simulations set by --matchups (below)

Search is deterministic for a given position, so one game per colour from the start
position is a sample of 2. To get more games, each game starts from a different book
opening (16 lines, 6 to 8 plies) and every opening is played twice with colours swapped.

Fair compute. The sidekick engine is cheaper per simulation, so two matchups are played:
    equal_time   the sidekick's simulation count is calibrated so that its mean seconds per
                 move matches prod's on the same machine (the count is then fixed, so games
                 are reproducible)
    equal_sims   400 vs 400, which isolates move quality from speed

    python policy_transformer/head_to_head_sidekick.py                      # 16 openings, both matchups
    python policy_transformer/head_to_head_sidekick.py --start-only         # 2 games per matchup
    python policy_transformer/head_to_head_sidekick.py --openings 4 --matchups equal_time

Results (JSON per matchup, PGN, summary) go to OUT_DIR after every opening, so a stopped run keeps what finished.
"""
import argparse
import json
import math
import os
import pickle
import sys
import time

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from policy_transformer.model import PolicyHead
from value_transformer.model import ChessFormer

# ── Checkpoints ──
BASE_CKPT = os.path.join(HERE, "ultra_3o9.pt")
SIDEKICK_CKPT = os.path.join(HERE, "best_sidekick_v1.pt")
VALUE_CKPT = os.path.join(ROOT, "value_transformer", "mini_value_6o4.pt")
BASE_VOCAB = os.path.join(HERE, "vocab.pkl")
SIDEKICK_VOCAB = os.path.join(HERE, "vocab1.pkl")
OUT_DIR = os.path.join(ROOT, "data", "h2h_results")

# ── Architectures (must match the checkpoints) ──
BASE_CFG = dict(n_embd=500, n_head=10, n_layer=8)
SIDEKICK_CFG = dict(n_embd=256, n_head=8, n_layer=4)
BLOCK_SIZE = 128

# ── Production search settings (engine/mcts.py defaults; app.py /v1/api/mcts-3, hard mode) ──
PROD_SIMS = 400
C_PUNC = 1.10
TOP_K = 10
TAU = 1.0

# ── Experiment ──
MAX_PLIES = 300         # longer games are adjudicated as draws
CALIB_POSITIONS = 12    # positions used to measure time per move
MAX_SIM_MULT = 4.0      # cap on the calibrated simulation count (x PROD_SIMS)

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

BOOK = [
    "e4 e5 Nf3 Nc6 Bb5 a6",             # Ruy Lopez
    "e4 e5 Nf3 Nc6 Bc4 Bc5",            # Italian
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6",    # Sicilian, Najdorf move order
    "e4 e6 d4 d5 Nc3 Nf6",              # French
    "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5",    # Caro-Kann
    "d4 d5 c4 e6 Nc3 Nf6",              # Queen's Gambit Declined
    "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6",       # King's Indian
    "d4 Nf6 c4 e6 Nc3 Bb4",             # Nimzo-Indian
    "c4 e5 Nc3 Nf6 Nf3 Nc6",            # English
    "Nf3 d5 g3 Nf6 Bg2 e6",             # Reti
    "d4 d5 Nf3 Nf6 Bf4 e6",             # London
    "e4 d5 exd5 Qxd5 Nc3 Qa5",          # Scandinavian
    "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4",   # Petrov
    "d4 Nf6 c4 e6 Nf3 b6",              # Queen's Indian
    "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4",     # Slav
    "e4 g6 d4 Bg7 Nc3 d6",              # Modern / Pirc
]


# ── Models and the two functions the search calls ──
def strip_compile_prefix(sd):
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def load_policy(ckpt_path, vocab_path, cfg):
    with open(vocab_path, "rb") as f:
        stoi = pickle.load(f)
    model = PolicyHead(vocab_size=len(stoi), block_size=BLOCK_SIZE, dropout=0.0, device=DEVICE, **cfg)
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(strip_compile_prefix(sd))
    model.to(DEVICE).eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"policy {os.path.basename(ckpt_path)}: {n:.2f}M params, vocab {len(stoi)}")
    return model, stoi


def mirror_uci(uci):
    # 'e2e4' -> 'e7e5' (ranks flipped); same as policy_transformer/inference.py
    return f"{uci[0]}{9 - int(uci[1])}{uci[2]}{9 - int(uci[3])}{uci[4:]}"


class PolicyFn:
    """Same output as policy_transformer.inference.predict_next_move: the top_k legal moves as
    (logit, uci), sorted by logit, with the history mirrored when Black is to move. The history
    comes in as a UCI move list instead of a PGN string (no re-parsing), and the logits of the
    legal moves are read in one transfer. `calls` counts forward passes."""

    def __init__(self, model, stoi, name):
        self.model, self.stoi, self.name, self.calls = model, stoi, name, 0
        self.pad = stoi.get("|", 0)

    @torch.no_grad()
    def __call__(self, moves, board, top_k):
        self.calls += 1
        mirror = board.turn == chess.BLACK
        if len(moves) == 0:
            ids = [0]
        else:
            ids = [self.stoi.get((mirror_uci(u) if mirror else u)[:4], self.pad) for u in moves]
        idx = torch.tensor([ids[-BLOCK_SIZE:]], dtype=torch.long, device=DEVICE)
        last = self.model(idx)[0][0, -1, :]

        legal = list(board.legal_moves)
        if not legal:
            return []
        ucis, tok_ids = [], []
        for m in legal:
            u = m.uci()
            ucis.append(u)
            tok_ids.append(self.stoi.get((mirror_uci(u) if mirror else u)[:4], -1))
        sel = torch.tensor([max(t, 0) for t in tok_ids], dtype=torch.long, device=DEVICE)
        scores = last.index_select(0, sel).float().cpu().tolist()
        cand = [(s if t >= 0 else float("-inf"), u) for s, t, u in zip(scores, tok_ids, ucis)]
        cand.sort(key=lambda x: x[0], reverse=True)
        return cand[:top_k]


class ValueFn:
    """Same as value_transformer.inference.ChessEvaluator.batch_evaluate (FEN list -> values)."""
    PIECE_MAP = {'.': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                 'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}

    def __init__(self, ckpt_path):
        # mini_value_6o4: 384 embd, 6 heads, 6 layers (production inference.py config)
        self.model = ChessFormer(block_size=64, n_embed=384, n_heads=6, n_layers=6, dropout=0.0)
        sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        self.model.load_state_dict(strip_compile_prefix(sd))
        self.model.to(DEVICE).eval()
        self.calls = 0
        n = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"value {os.path.basename(ckpt_path)}: {n:.2f}M params")

    def _tokens(self, board):
        toks = []
        for rank in range(7, -1, -1):
            for file in range(8):
                piece = board.piece_at(chess.square(file, rank))
                toks.append(self.PIECE_MAP[piece.symbol()] if piece else 0)
        return np.array(toks, dtype=np.int64)

    @torch.no_grad()
    def __call__(self, fens):
        if not fens:
            return []
        self.calls += 1
        x = torch.from_numpy(np.array([self._tokens(chess.Board(f)) for f in fens])).to(DEVICE)
        return self.model(x).squeeze(1).cpu().numpy().tolist()


# ── MCTS: engine/mcts.py, with the policy chosen per node (root vs deeper) ──
class Node:
    __slots__ = ("board", "moves", "parent", "children", "N", "W", "P")

    def __init__(self, board, moves, parent=None):
        self.board, self.moves, self.parent = board, moves, parent
        self.children = {}
        self.N, self.W, self.P = 0, 0.0, 0.0


class MCTS:
    """Line-for-line the production search (selection, expansion, evaluation, backup, move choice).
    The only change: expansion asks `root_policy` when the node is the root and `deep_policy`
    otherwise. With both equal to the base policy this is the production engine."""

    def __init__(self, moves, board, value_fn, root_policy, deep_policy, sims,
                 c_punc=C_PUNC, k=TOP_K, tau=TAU):
        self.root = Node(board.copy(), tuple(moves))
        self.value_fn, self.root_policy, self.deep_policy = value_fn, root_policy, deep_policy
        self.sims, self.c_punc, self.k, self.tau = sims, c_punc, k, tau
        self.expansions = 0
        # draw avoidance: root FEN and the last 4 (fen before the move, move) pairs
        self.root_fen = board.fen()
        replay, pairs = chess.Board(), []
        for i, u in enumerate(moves):
            if i >= len(moves) - 4:
                pairs.append((replay.fen(), u))
            replay.push_uci(u)
        self.recent_pairs = pairs

    def search(self):
        for i in range(self.sims):
            node = self._select(self.root)
            value = self._evaluate(node)
            if i < self.sims - 1 and not node.board.is_game_over():
                self._expand(node)
            self._backpropagate(node, value)

        best_move, best_score = None, -float("inf")
        for key, child in self.root.children.items():
            score = child.N
            if (self.root_fen, key) in self.recent_pairs:
                score *= 0.95
            if score > best_score:
                best_score, best_move = score, key
        return best_move, best_score

    def _expand(self, node):
        policy = self.root_policy if node is self.root else self.deep_policy
        cand = policy(node.moves, node.board, self.k)
        if not cand:
            return
        self.expansions += 1
        logits = torch.tensor([s for s, _ in cand], dtype=torch.float32)
        probs = F.softmax(logits / self.tau, dim=0).tolist()

        fens, kids = [], []
        for prob, (_, mv) in zip(probs, cand):
            cb = node.board.copy()
            cb.push_uci(mv)
            fens.append(cb.mirror().fen() if cb.turn == chess.BLACK else cb.fen())
            kids.append((mv, prob, cb))
        values = self.value_fn(fens)
        for (mv, prob, cb), v in zip(kids, values):
            child = Node(cb, node.moves + (mv,), parent=node)
            child.P = prob
            child.W = -v
            node.children[mv] = child

    def _select(self, node):
        c = self.c_punc
        while node.children:
            best_score, best_child = -float("inf"), None
            sqrt_N = node.N ** 0.5
            for child in node.children.values():
                Q = child.W / child.N if child.N > 0 else child.W
                U = c * child.P * sqrt_N / (child.N + 1)
                if Q + U > best_score:
                    best_score, best_child = Q + U, child
            node = best_child
        return node

    def _evaluate(self, node):
        if node.board.is_game_over():
            return 1.03 if node.board.is_checkmate() else 0
        return node.W

    def _backpropagate(self, node, value):
        while node:
            node.N += 1
            node.W += value
            value = -value
            node = node.parent


class Engine:
    def __init__(self, name, sims, root_policy, deep_policy, value_fn):
        self.name, self.sims, self.value_fn = name, sims, value_fn
        self.root_policy, self.deep_policy = root_policy, deep_policy
        self.policies = {root_policy, deep_policy}

    def choose(self, moves, board, sims=None):
        """Returns (uci move, stats). Timing covers the whole search."""
        before = {p.name: p.calls for p in self.policies}
        v0 = self.value_fn.calls
        t0 = time.perf_counter()
        mcts = MCTS(moves, board, self.value_fn, self.root_policy, self.deep_policy, sims or self.sims)
        move, _ = mcts.search()
        dt = time.perf_counter() - t0
        calls = {p.name: p.calls - before[p.name] for p in self.policies}
        return move, dict(sec=dt, base_calls=calls.get("base", 0), side_calls=calls.get("sidekick", 0),
                          value_calls=self.value_fn.calls - v0, expansions=mcts.expansions)


# ── Openings ──
def opening_board(line):
    board, moves = chess.Board(), []
    for san in line.split():
        moves.append(board.push_san(san).uci())
    return board, tuple(moves)


# ── Checks before spending hours ──
def selftest(prod, sidekick, base_policy, side_policy, value_fn):
    board, moves = opening_board(BOOK[0])
    for fn in (base_policy, side_policy):
        out = fn(moves, board, TOP_K)
        assert len(out) == TOP_K and all(a[0] >= b[0] for a, b in zip(out, out[1:]))
        assert all(chess.Move.from_uci(u) in board.legal_moves for _, u in out)
        print(f"  {fn.name:9s} top-3 after '{BOOK[0]}':", [(round(s, 2), u) for s, u in out[:3]])

    m1, _ = prod.choose(moves, board, sims=60)
    m2, _ = prod.choose(moves, board, sims=60)
    assert m1 == m2, "search is not deterministic on this machine"          # 1. determinism

    same = Engine("hybrid-with-base", 60, base_policy, base_policy, value_fn)
    assert same.choose(moves, board)[0] == m1                                # 2. hybrid path == prod

    _, st = sidekick.choose(moves, board, sims=60)                           # 3. large policy once per move
    assert st["base_calls"] == 1 and st["side_calls"] == st["expansions"] - 1, st
    print("  sidekick engine, 60 sims:", st)
    print("  self-test OK")


# ── Calibration: sidekick simulations so its time per move matches prod's ──
def mean_time(engine, positions, sims):
    return float(np.mean([engine.choose(m, b, sims=sims)[1]["sec"] for b, m in positions]))


def calibrate(prod, sidekick, prod_sims):
    positions = [(chess.Board(), ())] + [opening_board(l) for l in BOOK[:CALIB_POSITIONS - 1]]
    mean_time(prod, positions[:2], prod_sims)         # warm-up
    mean_time(sidekick, positions[:2], prod_sims)
    t_prod = mean_time(prod, positions, prod_sims)
    t_same = mean_time(sidekick, positions, prod_sims)
    print(f"  prod     @ {prod_sims} sims: {t_prod:.3f} s/move")
    print(f"  sidekick @ {prod_sims} sims: {t_same:.3f} s/move  (x{t_prod / t_same:.2f} faster)")
    cap = MAX_SIM_MULT * prod_sims
    sims = int(round(min(prod_sims * t_prod / t_same, cap)))
    for _ in range(2):                                # cost is close to, not exactly, linear in sims
        t = mean_time(sidekick, positions, sims)
        print(f"  sidekick @ {sims} sims: {t:.3f} s/move")
        sims = int(round(min(sims * t_prod / t, cap)))
    print(f"  => equal-time sidekick simulations: {sims} (prod: {prod_sims})")
    return sims, t_prod


# ── Play ──
def play_game(line, white, black):
    board, moves = opening_board(line)
    moves = list(moves)
    n_book = len(moves)
    per_engine = {white.name: [], black.name: []}
    while not board.is_game_over(claim_draw=True) and len(moves) < MAX_PLIES:
        eng = white if board.turn == chess.WHITE else black
        mv, st = eng.choose(tuple(moves), board)
        if mv is None:
            break
        per_engine[eng.name].append(st)
        board.push_uci(mv)
        moves.append(mv)
    over = board.is_game_over(claim_draw=True)
    result = board.result(claim_draw=True) if over else "1/2-1/2"
    pgn = chess.pgn.Game.from_board(board)
    pgn.headers.update(White=white.name, Black=black.name, Result=result,
                       Event="sidekick head-to-head", Opening=line or "start")
    if not over:
        pgn.headers["Termination"] = "adjudicated: ply cap"
    return dict(opening=line, white=white.name, black=black.name, result=result, plies=len(moves),
                book_plies=n_book, per_engine=per_engine, pgn=str(pgn), adjudicated=not over)


def run_matchup(name, prod, side, openings, prod_sims):
    games = []
    bar = tqdm(total=2 * len(openings), desc=name)
    for line in openings:
        for white, black in ((side, prod), (prod, side)):
            g = play_game(line, white, black)
            games.append(g)
            bar.update(1)
            bar.set_postfix(last=f"{g['white']}-{g['black']} {g['result']}", plies=g["plies"])
        with open(os.path.join(OUT_DIR, f"h2h_{name}.json"), "w") as f:
            json.dump(dict(name=name, side_sims=side.sims, prod_sims=prod_sims, games=games), f)
        with open(os.path.join(OUT_DIR, f"h2h_{name}.pgn"), "w") as f:
            f.write("\n\n".join(g["pgn"] for g in games) + "\n")
    bar.close()
    return games


# ── Report ──
def score_for(g, who):
    if g["result"] == "1/2-1/2":
        return 0.5
    return float((g["result"] == "1-0") == (g["white"] == who))


def elo(s):
    s = min(max(s, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / s - 1)


def report(results, prod_sims):
    rng = np.random.default_rng(0)
    summary = {}
    for name, (side_sims, games) in results.items():
        sc = np.array([score_for(g, "sidekick") for g in games])
        w, d, l = int((sc == 1).sum()), int((sc == 0.5).sum()), int((sc == 0).sum())
        pair = np.array([sc[i:i + 2].mean() for i in range(0, len(sc), 2)])   # one value per opening
        boots = np.array([rng.choice(pair, len(pair)).mean() for _ in range(5000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"\n=== {name}: sidekick @ {side_sims} sims vs prod @ {prod_sims} sims ({len(games)} games) ===")
        print(f"sidekick: +{w} ={d} -{l}   score {sc.mean():.3f}  (95% CI {lo:.3f} to {hi:.3f}, bootstrap over openings)")
        print(f"approx Elo difference (sidekick - prod): {elo(sc.mean()):+.0f}  (CI {elo(lo):+.0f} to {elo(hi):+.0f})")
        print(f"adjudicated at ply cap: {sum(g['adjudicated'] for g in games)}; "
              f"mean plies: {np.mean([g['plies'] for g in games]):.0f}")
        row = {}
        for who in ("prod", "sidekick"):
            st = [s for g in games for s in g["per_engine"][who]]
            row[who] = dict(sec_per_move=float(np.mean([s["sec"] for s in st])),
                            base_calls=float(np.mean([s["base_calls"] for s in st])),
                            side_calls=float(np.mean([s["side_calls"] for s in st])),
                            expansions=float(np.mean([s["expansions"] for s in st])), moves=len(st))
            r = row[who]
            print(f"  {who:9s} {r['sec_per_move']:.3f} s/move | per move: {r['base_calls']:.1f} base-policy calls, "
                  f"{r['side_calls']:.1f} sidekick calls, {r['expansions']:.0f} expansions ({r['moves']} moves)")
        summary[name] = dict(side_sims=side_sims, prod_sims=prod_sims, games=len(games), wins=w, draws=d,
                             losses=l, score=float(sc.mean()), ci95=[float(lo), float(hi)], engines=row)
    with open(os.path.join(OUT_DIR, "h2h_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved h2h_summary.json, h2h_<matchup>.json and h2h_<matchup>.pgn to {OUT_DIR}")


def main():
    global MAX_PLIES
    ap = argparse.ArgumentParser()
    ap.add_argument("--openings", type=int, default=len(BOOK), help="book lines to use (each is played with both colours)")
    ap.add_argument("--start-only", action="store_true", help="only the initial position, both colours (2 games per matchup)")
    ap.add_argument("--matchups", choices=["equal_time", "equal_sims", "both"], default="both")
    ap.add_argument("--prod-sims", type=int, default=PROD_SIMS, help="production simulations (default 400)")
    ap.add_argument("--max-plies", type=int, default=MAX_PLIES)
    args = ap.parse_args()

    MAX_PLIES = args.max_plies
    openings = [""] if args.start_only else BOOK[:args.openings]
    for line in openings:
        opening_board(line)    # fails loudly if a line is illegal
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}; {len(openings)} opening(s) x 2 colours = {2 * len(openings)} games per matchup")

    base_model, base_stoi = load_policy(BASE_CKPT, BASE_VOCAB, BASE_CFG)
    side_model, side_stoi = load_policy(SIDEKICK_CKPT, SIDEKICK_VOCAB, SIDEKICK_CFG)
    base_policy = PolicyFn(base_model, base_stoi, "base")
    side_policy = PolicyFn(side_model, side_stoi, "sidekick")
    value_fn = ValueFn(VALUE_CKPT)

    prod = Engine("prod", args.prod_sims, base_policy, base_policy, value_fn)
    sidekick = Engine("sidekick", args.prod_sims, base_policy, side_policy, value_fn)

    print("\nSelf-test")
    selftest(prod, sidekick, base_policy, side_policy, value_fn)

    matchups = {"equal_time": "auto", "equal_sims": args.prod_sims}
    if args.matchups != "both":
        matchups = {args.matchups: matchups[args.matchups]}

    equal_time_sims = None
    if "auto" in matchups.values():
        print("\nCalibration (equal time per move)")
        equal_time_sims, t_prod = calibrate(prod, sidekick, args.prod_sims)
        est_h = 2 * len(openings) * 80 * t_prod / 3600
        print(f"  rough estimate per matchup: {est_h:.1f} h (80-ply games, prod's {t_prod:.2f} s/move for both sides)")

    results = {}
    for name, s in matchups.items():
        side_sims = equal_time_sims if s == "auto" else int(s)
        print(f"\n=== {name}: sidekick @ {side_sims} sims vs prod @ {args.prod_sims} sims ===")
        side = Engine("sidekick", side_sims, base_policy, side_policy, value_fn)
        results[name] = (side_sims, run_matchup(name, prod, side, openings, args.prod_sims))

    report(results, args.prod_sims)


if __name__ == "__main__":
    main()
