#!/usr/bin/env python3
"""End-to-end pipeline: subsample -> sessions -> splits -> train -> eval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from playlist_continue.data.lfm2b import derive_sessions
from playlist_continue.eval.comparison import run_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, help="Path to events parquet file")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()

    print("Loading events...")
    events = pq.read_table(args.events)

    print("Deriving sessions...")
    sessioned = derive_sessions(events)

    print("Running all model comparisons...")
    results = run_comparison(sessioned, embed_dim=args.embed_dim, epochs=args.epochs)

    for model, metrics in results.items():
        print(f"  {model}: {metrics}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
