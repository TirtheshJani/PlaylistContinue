#!/usr/bin/env python3
"""End-to-end pipeline: subsample -> sessions -> splits -> train -> eval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from playlist_continue.data.lfm2b import derive_sessions
from playlist_continue.data.splits import make_splits
from playlist_continue.eval.harness import evaluate
from playlist_continue.models.popularity import PopularityRecommender
from playlist_continue.models.two_tower import TwoTowerModel


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

    print("Splitting...")
    train, val, eval_set = make_splits(sessioned, val_fraction=0.1, seed=42)

    results: dict[str, dict[str, float]] = {}

    print("Evaluating popularity baseline...")
    pop = PopularityRecommender()
    pop.fit(train)
    results["popularity"] = evaluate(pop, eval_set)
    print(f"  popularity: {results['popularity']}")

    n_items = int(events.column("track_id").max().as_py()) + 1
    print(f"Training two-tower (n_items={n_items}, embed_dim={args.embed_dim})...")
    tt = TwoTowerModel(n_items=n_items, embed_dim=args.embed_dim)
    tt.fit(train, epochs=args.epochs)
    results["two_tower"] = evaluate(tt, eval_set)
    print(f"  two_tower:  {results['two_tower']}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
