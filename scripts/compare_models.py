#!/usr/bin/env python3
"""Compare all 6 blog-post recommender architectures on the same frozen eval split.

Usage:
    python scripts/compare_models.py --events data/processed/events.parquet

Output: table of R-precision, NDCG@20, Recall@500 for each model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from playlist_continue.data.lfm2b import derive_sessions
from playlist_continue.eval.comparison import run_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all recommender architectures.")
    parser.add_argument("--events", required=True, help="Path to events parquet file")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    print("Loading events...")
    events = pq.read_table(args.events)
    sessioned = derive_sessions(events)

    print(f"Running comparison on {len(sessioned):,} events...")
    results = run_comparison(sessioned, embed_dim=args.embed_dim, epochs=args.epochs)

    header = f"{'Model':<22}  {'R-prec':>8}  {'NDCG@20':>8}  {'Rec@500':>8}"
    print("\n" + header)
    print("-" * len(header))
    for model, metrics in results.items():
        print(
            f"{model:<22}  {metrics['r_precision']:>8.4f}  "
            f"{metrics['ndcg_at_20']:>8.4f}  {metrics['recall_at_500']:>8.4f}"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
