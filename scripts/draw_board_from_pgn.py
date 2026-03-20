"""
Draw an SVG of a board state from a PGN file.
Picks a random game and a random position (or after N moves), renders board only — no legend, no highlights.
Requires: chess
"""
import argparse
import chess
import chess.pgn
import os
import random


def games_from_pgn(path: str):
    """Yield parsed games from a PGN file."""
    with open(path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game


def pick_board_from_pgn(pgn_path: str, game_index: int = None, ply: int = None):
    """
    Return (board, description) from pgn_path.
    If game_index is None, pick a random game. If ply is None, pick a random ply in that game.
    """
    games = list(games_from_pgn(pgn_path))
    if not games:
        raise ValueError(f"No games found in {pgn_path}")
    if game_index is None:
        game_index = random.randint(0, len(games) - 1)
    game = games[game_index]
    moves = list(game.mainline_moves())
    if ply is None:
        ply = random.randint(0, len(moves)) if moves else 0
    else:
        ply = max(0, min(ply, len(moves)))
    board = game.board()
    for m in moves[:ply]:
        board.push(m)
    desc = f"Game {game_index + 1}, after {ply} plies"
    return board, desc


def draw_board_from_pgn(
    pgn_path: str,
    output_path: str,
    size: int = 400,
    game_index: int = None,
    ply: int = None,
):
    """Render board state from PGN to SVG (board only)."""
    board, desc = pick_board_from_pgn(pgn_path, game_index=game_index, ply=ply)
    svg_str = chess.svg.board(board=board, size=size)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(svg_str)
    print(f"Saved: {output_path} ({desc})")
    return output_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(script_dir)
    default_pgn = os.path.join(root, "data", "elitegames", "Bobby Fischer-black.pgn")
    default_out = os.path.join(root, "assets", "board_from_pgn.svg")

    p = argparse.ArgumentParser(description="Draw SVG board from a random position in a PGN file.")
    p.add_argument("pgn", nargs="?", default=default_pgn, help=f"PGN file path (default: {default_pgn})")
    p.add_argument("-o", "--output", default=default_out, help=f"Output SVG path (default: {default_out})")
    p.add_argument("-s", "--size", type=int, default=400, help="Board size in pixels")
    p.add_argument("-g", "--game", type=int, default=None, help="Game index (0-based); default random")
    p.add_argument("-p", "--ply", type=int, default=None, help="Ply count (position after N half-moves); default random")
    args = p.parse_args()

    if not os.path.isfile(args.pgn):
        print(f"Error: PGN file not found: {args.pgn}")
        return 1
    draw_board_from_pgn(args.pgn, args.output, size=args.size, game_index=args.game, ply=args.ply)
    return 0


if __name__ == "__main__":
    exit(main())
