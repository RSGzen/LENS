r"""Print gold-triple rows as one-line search/verification prompts.

Each row is printed as::

    [7] Is it true that QASPER (Dataset) uses GPT-4 (Method) at arxiv paper 2401.12345
        https://arxiv.org/abs/2401.12345
        context: ... we evaluate on QASPER, HotpotQA and MuSiQue ...

Copy the line into Google, or open the arXiv URL, then fill the CSV by hand.

Recommended (fast) usage — source context + read each paper once:
  cd D:\lens
  .\.venv\Scripts\python.exe scripts\print_gold_queries.py --csv D:\lens-data\logs\gold_triples_03.csv --start 11 --url --with-context --group-by-paper

Resume only unfilled rows:
  .\.venv\Scripts\python.exe scripts\print_gold_queries.py --csv D:\lens-data\logs\gold_triples_03.csv --skip-filled --with-context --group-by-paper
"""

from __future__ import annotations

import argparse
import sys

from lens.kg_gold import (
    clean_surface,
    format_triple_query,
    read_triples_worksheet,
    snippet,
)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Print gold-triple prompts")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--start", type=int, default=1, help="first row (1-based)")
    parser.add_argument("--end", type=int, default=0, help="last row (0 = to the end)")
    parser.add_argument("--skip-filled", action="store_true",
                        help="skip rows whose gold_relation_ok is already filled")
    parser.add_argument("--url", action="store_true", help="also print the arXiv URL")
    parser.add_argument("--with-context", action="store_true",
                        help="fetch and print a source-chunk snippet (DB required)")
    parser.add_argument("--group-by-paper", action="store_true",
                        help="print rows sorted by arxiv_id (read each paper once)")
    parser.add_argument("--context-width", type=int, default=400)
    parser.add_argument("--clean", action="store_true",
                        help="apply punctuation cleanup before printing")
    args = parser.parse_args()

    rows = read_triples_worksheet(args.csv)
    selected = [
        (index, row)
        for index, row in enumerate(rows, 1)
        if index >= args.start
        and (not args.end or index <= args.end)
        and not (args.skip_filled and str(row.get("gold_relation_ok", "")).strip())
    ]

    if args.group_by_paper:
        selected.sort(key=lambda item: (item[1].get("arxiv_id", ""), item[0]))

    contexts: dict[str, str] = {}
    if args.with_context:
        from lens.db.connect import connect
        from lens.tools.retrieval import fetch_chunks

        chunk_ids = list({row.get("chunk_uuid") for _, row in selected if row.get("chunk_uuid")})
        with connect(readonly=True) as conn:
            contexts = {
                chunk["chunk_uuid"]: chunk["chunk_text"]
                for chunk in fetch_chunks(conn, chunk_ids)
            }

    for index, row in selected:
        if args.clean:
            row["head"] = clean_surface(row["head"])
            row["tail"] = clean_surface(row["tail"])
        print(f"[{index}] {format_triple_query(row)}")
        if args.url:
            print(f"    https://arxiv.org/abs/{row['arxiv_id']}")
        if args.with_context:
            text = contexts.get(row.get("chunk_uuid", ""), "")
            print(
                "    context: "
                + snippet(text, [row["head"], row["tail"]], args.context_width)
            )
        print()


if __name__ == "__main__":
    main()
