r"""Generate the README wordmark: block letters as hollow cells, in an SVG.

The wordmark is a pixel grid: each lit cell is drawn as a hollow rounded square,
which is what gives the outlined, diagrammatic look. Everything is derived from the
bitmaps below, so changing a letter is a four-character edit rather than a redraw.

Two colours are declared in the file and switched with ``prefers-color-scheme``, so
the mark reads correctly on GitHub's light and dark themes from a single asset.

Usage::

    python scripts/make_wordmark.py            # writes docs/images/lens-wordmark.svg
    python scripts/make_wordmark.py --stdout   # print the SVG instead

No dependencies: the geometry is plain arithmetic and the output is plain text.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

#: 7 rows by 5 columns per letter. ``#`` is a lit cell, ``.`` is empty.
BITMAPS: dict[str, tuple[str, ...]] = {
    "L": (
        "#....",
        "#....",
        "#....",
        "#....",
        "#....",
        "#....",
        "#####",
    ),
    "E": (
        "#####",
        "#....",
        "#....",
        "####.",
        "#....",
        "#....",
        "#####",
    ),
    "N": (
        "#...#",
        "##..#",
        "#.#.#",
        "#.#.#",
        "#..##",
        "#...#",
        "#...#",
    ),
    "S": (
        ".####",
        "#....",
        "#....",
        ".###.",
        "....#",
        "#...#",
        ".####",
    ),
}

CELL = 26  # cell pitch, in user units
GAP = 3  # gap between neighbouring cells
RADIUS = 3  # corner radius of a cell
PAD = 18  # padding inside the frame
LETTER_GAP = CELL  # space between two letters
FRAME_INSET = 4
FRAME_RADIUS = 14

#: Emit the SVG at twice its geometry size: GitHub scales a wide image down to the
#: content width, and the extra resolution keeps the cells crisp at that size.
SCALE = 2

INK_LIGHT = "#1f2328"
INK_DARK = "#e6edf3"
FRAME_LIGHT = "#d0d7de"
FRAME_DARK = "#30363d"

WORD = "LENS"


def build_svg(word: str = WORD) -> str:
    """Return the SVG document for ``word``."""
    rows = max(len(BITMAPS[letter]) for letter in word)
    width = 0

    for index, letter in enumerate(word):
        width += len(BITMAPS[letter][0]) * CELL

        if index:
            width += LETTER_GAP

    body_width = width + 2 * PAD
    body_height = rows * CELL + 2 * PAD

    cells: list[str] = []
    x_offset = PAD

    for letter in word:
        bitmap = BITMAPS[letter]

        for row, line in enumerate(bitmap):
            for column, mark in enumerate(line):
                if mark != "#":
                    continue

                x = x_offset + column * CELL + GAP / 2
                y = PAD + row * CELL + GAP / 2
                size = CELL - GAP
                cells.append(
                    f'    <rect x="{x:g}" y="{y:g}" width="{size:g}" height="{size:g}" '
                    f'rx="{RADIUS}" class="cell"/>'
                )

        x_offset += len(bitmap[0]) * CELL + LETTER_GAP

    frame_x = FRAME_INSET
    frame_y = FRAME_INSET
    frame_width = body_width - 2 * FRAME_INSET
    frame_height = body_height - 2 * FRAME_INSET

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{body_width * SCALE:g}" '
        f'height="{body_height * SCALE:g}" viewBox="0 0 {body_width} {body_height}" '
        f'role="img" aria-label="{word}">\n'
        f"  <title>{word}</title>\n"
        f"  <style>\n"
        f"    .cell {{ fill: none; stroke: {INK_LIGHT}; stroke-width: 2.8; }}\n"
        f"    .frame {{ fill: none; stroke: {FRAME_LIGHT}; stroke-width: 1.5; }}\n"
        f"    @media (prefers-color-scheme: dark) {{\n"
        f"      .cell {{ stroke: {INK_DARK}; }}\n"
        f"      .frame {{ stroke: {FRAME_DARK}; }}\n"
        f"    }}\n"
        f"  </style>\n"
        f'  <rect x="{frame_x}" y="{frame_y}" width="{frame_width}" '
        f'height="{frame_height}" rx="{FRAME_RADIUS}" class="frame"/>\n'
        f"{chr(10).join(cells)}\n"
        f"</svg>\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default=str(pathlib.Path(__file__).resolve().parents[1] / "docs/images/lens-wordmark.svg"),
        help="where to write the SVG",
    )
    parser.add_argument("--stdout", action="store_true", help="print instead of writing")
    parser.add_argument("--word", default=WORD, help="word to draw (letters: L E N S)")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    svg = build_svg(args.word.upper())

    if args.stdout or args.word.upper() != WORD:
        print(svg, end="")
        return 0

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8", newline="\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
