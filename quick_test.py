# quick_test.py
from engine import find_best_move

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

move, score, pv, nodes, depth = find_best_move(
    START_FEN, max_depth=4, time_budget_ms=1000, node_limit=200_000, skill_level=6
)
print("best:", move, "score:", score, "pv:", pv, "nodes:", nodes, "depth:", depth)