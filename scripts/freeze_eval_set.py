#!/usr/bin/env python3
"""Freeze the eval split from a processed events parquet file.

Run ONCE when first setting up the project with real LFM-2b data.
The output file data/eval/eval_set.parquet is never regenerated.

Usage:
    python scripts/freeze_eval_set.py --events data/processed/events.parquet
"""

from __future__ import annotations

import argparse

import pyarrow.parquet as pq

from playlist_continue.data.freeze import freeze_eval_set
from playlist_continue.data.lfm2b import derive_sessions


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze eval split to parquet.")
    parser.add_argument("--events", required=True, help="Path to processed events parquet")
    parser.add_argument("--output", default="data/eval/eval_set.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading events...")
    events = pq.read_table(args.events)

    print("Deriving sessions...")
    sessioned = derive_sessions(events)

    print(f"Freezing eval set -> {args.output}")
    eval_set = freeze_eval_set(sessioned, seed=args.seed, output_path=args.output)
    n_users = len(set(eval_set.column("user_id").to_pylist()))
    print(f"Wrote {len(eval_set):,} eval events ({n_users:,} users)")


if __name__ == "__main__":
    main()
