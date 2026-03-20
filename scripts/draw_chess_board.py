"""
Draw a default chess board with pieces and highlight specific moves.
Uses the same rendering as transformer-dev.ipynb (chess.svg.board).
Outputs PNG with a legend (and SVG). Requires: chess; for PNG also cairosvg (+ cairo).
"""
import chess
import chess.svg
import os
import re

# Red, blue, green — same solid color on board and in legend
HIGHLIGHTS = {
    "e2e4": ("#2E7D32", "#2E7D32"),   # green
    "g1f3": ("#1565C0", "#1565C0"),   # blue
    "b2b3": ("#C62828", "#C62828"),   # red
}


def uci_to_squares(uci: str):
    """Return (from_square, to_square) as chess.Square, or (None, None)."""
    if len(uci) < 4:
        return None, None
    try:
        return chess.parse_square(uci[:2]), chess.parse_square(uci[2:4])
    except (ValueError, IndexError):
        return None, None


def _svg_add_legend(svg_str: str, highlights: dict) -> str:
    """Widen SVG and add a legend on the right. highlights: UCI -> (solid_hex, fill_hex)."""
    match = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg_str)
    if not match:
        return svg_str
    w, h = int(match.group(1)), int(match.group(2))
    legend_width = 100
    new_w = w + legend_width
    svg_str = svg_str.replace(f'viewBox="0 0 {w} {h}"', f'viewBox="0 0 {new_w} {h}"', 1)
    # Widen display so legend is visible (keep height unchanged)
    new_display_w = int(400 * new_w / w) if w else 500
    svg_str = re.sub(r'width="\d+"', f'width="{new_display_w}"', svg_str, count=1)

    legend_x = w + 12
    legend = ['<g id="legend" font-family="sans-serif">']
    # Full-height white panel on the right
    legend.append(f'<rect x="{w}" y="0" width="{legend_width}" height="{h}" fill="#ffffff" stroke="#ccc" stroke-width="1"/>')
    legend.append(f'<text x="{legend_x}" y="28" font-size="14" font-weight="bold" fill="#1a1a1a">Moves</text>')
    for i, (uci, (solid_hex, _)) in enumerate(highlights.items()):
        y = 52 + i * 28
        legend.append(f'<rect x="{legend_x}" y="{y - 12}" width="20" height="20" fill="{solid_hex}" stroke="#333" stroke-width="1"/>')
        legend.append(f'<text x="{legend_x + 26}" y="{y + 2}" font-size="13" fill="#1a1a1a" font-family="monospace">{uci}</text>')
    legend.append("</g>")
    insert = "\n" + "\n".join(legend) + "\n"
    if not svg_str.strip().endswith("</svg>"):
        return svg_str
    return svg_str[:-6] + insert + "</svg>"


def draw_board(highlights: dict, output_path: str, size: int = 400):
    """
    highlights: dict mapping UCI -> (solid_hex_for_legend, fill_hex_for_board).
    Outputs PNG with board + legend. Falls back to SVG only if PNG conversion fails.
    """
    board = chess.Board()
    fill = {}
    for uci, (_solid, fill_hex) in highlights.items():
        from_sq, to_sq = uci_to_squares(uci)
        if from_sq is not None:
            fill[from_sq] = fill_hex
            fill[to_sq] = fill_hex

    svg_str = chess.svg.board(
        board=board,
        size=size,
        fill=fill,
    )
    # Add thin, even black border (inset 0.5 so 1px stroke is inside on all edges)
    for _uci, (solid_hex, fill_hex) in highlights.items():
        def repl(m, color=fill_hex):
            x, y = m.group(1), m.group(2)
            xi, yi = float(x) + 0.5, float(y) + 0.5
            return (
                f'<rect x="{x}" y="{y}" width="45" height="45" fill="{color}" stroke="none" />'
                f'<rect x="{xi}" y="{yi}" width="44" height="44" fill="none" stroke="#000" stroke-width="1" />'
            )
        # Library outputs: <rect x="..." y="..." width="45" height="45" stroke="none" fill="#HEX" />
        pattern = rf'<rect x="(\d+)" y="(\d+)" width="45" height="45" stroke="none" fill="{fill_hex}" \/>'
        svg_str = re.sub(pattern, repl, svg_str)
    svg_str = _svg_add_legend(svg_str, highlights)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    base, ext = os.path.splitext(output_path)
    svg_path = base + ".svg"
    png_path = output_path if ext.lower() == ".png" else base + ".png"

    with open(svg_path, "w") as f:
        f.write(svg_str)
    print(f"Saved: {svg_path}")

    # Convert to PNG (board + legend are already in the SVG)
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), write_to=png_path)
        print(f"Saved: {png_path}")
    except Exception:
        print("PNG skip (install cairosvg + cairo, e.g. brew install cairo): use the .svg file.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(script_dir)
    output_path = os.path.join(root, "assets", "chess_board_highlighted_moves.png")
    draw_board(HIGHLIGHTS, output_path, size=400)
