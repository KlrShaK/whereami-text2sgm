#!/usr/bin/env python3
"""
Sort an eval log table (pipe-separated) by a numeric column (default: 'Err (m)').

Usage:
  python sort_eval_table.py eval_loc_summary_mk3.log > sorted.log
  python sort_eval_table.py eval_loc_summary_mk3.log --sort "Err (m)" --desc --top 50
  python sort_eval_table.py eval_loc_summary_mk3.log --sort "R90%" --asc
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def _to_float(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def find_first_table(lines: List[str]) -> Tuple[int, List[str]]:
    """
    Returns (header_idx, columns) for the first table that looks like:
      Scene | Frame | ... | Err (m) | ...
    """
    for i, line in enumerate(lines):
        if "|" in line and "Scene" in line and "Frame" in line and "Err (m)" in line:
            cols = [c.strip() for c in line.split("|")]
            return i, cols
    raise ValueError("No table header found (expected a line containing Scene | Frame | ... | Err (m)).")


def parse_table(lines: List[str], header_idx: int, cols: List[str]) -> List[Dict[str, str]]:
    """
    Parses rows after the header. Assumes the next line is a separator of ---+---+---.
    Stops when it leaves the table (blank line / non-pipe line / 'Aggregate' line).
    """
    rows: List[Dict[str, str]] = []

    for line in lines[header_idx + 2 :]:
        if not line.strip():
            break
        if line.lstrip().startswith("Aggregate"):
            break
        if "|" not in line:
            break
        # skip separator-like lines
        stripped = line.strip()
        if stripped and all(ch in "-+" for ch in stripped):
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) != len(cols):
            # skip malformed lines
            continue
        rows.append(dict(zip(cols, parts)))
    return rows


def format_table(cols: List[str], rows: List[Dict[str, str]]) -> str:
    """
    Prints an aligned ascii table similar to the input:
      header with ' | '
      separator with '-' and '+'
      rows with ' | '
    """
    text_cols = {"Scene", "Frame"}  # left-align these, right-align the rest

    # compute widths from header + all cell strings
    widths = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))

    header = " | ".join(f"{c:<{widths[c]}}" for c in cols)
    sep = "+".join("-" * widths[c] for c in cols)

    out_lines = [header, sep]
    for r in rows:
        cells = []
        for c in cols:
            v = str(r.get(c, ""))
            if c in text_cols:
                cells.append(f"{v:<{widths[c]}}")
            else:
                cells.append(f"{v:>{widths[c]}}")
        out_lines.append(" | ".join(cells))

    return "\n".join(out_lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", type=Path, help="Path to the .log file containing the table")
    ap.add_argument("--sort", default="Err (m)", help='Column to sort by (default: "Err (m)")')
    ap.add_argument("--desc", action="store_true", help="Sort descending (default)")
    ap.add_argument("--asc", action="store_true", help="Sort ascending")
    ap.add_argument("--top", type=int, default=None, help="Print only top N rows after sorting")
    args = ap.parse_args()

    text = args.logfile.read_text(errors="replace")
    lines = text.splitlines()

    header_idx, cols = find_first_table(lines)
    rows = parse_table(lines, header_idx, cols)

    if args.sort not in cols:
        raise SystemExit(f'Column "{args.sort}" not found. Available columns:\n  ' + "\n  ".join(cols))

    # Decide sort order
    if args.asc and args.desc:
        raise SystemExit("Choose only one: --asc or --desc")
    descending = True
    if args.asc:
        descending = False
    if args.desc:
        descending = True

    # Sort by numeric value in args.sort (NaNs go last)
    def sort_key(r: Dict[str, str]) -> Tuple[int, float]:
        v = _to_float(r.get(args.sort, ""))
        isnan = 1 if math.isnan(v) else 0
        return (isnan, v)

    rows_sorted = sorted(rows, key=sort_key, reverse=descending)

    if args.top is not None:
        rows_sorted = rows_sorted[: args.top]

    print(format_table(cols, rows_sorted))


if __name__ == "__main__":
    main()
