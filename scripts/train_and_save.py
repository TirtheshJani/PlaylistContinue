#!/usr/bin/env python3
"""Train the full cascade and save it for serving.

Usage:
    python scripts/train_and_save.py \
        --events data/processed/events.parquet \
        --output models/cascade.pkl

The saved model can be loaded by the serve container via:
    MODEL_PATH=models/cascade.pkl uvicorn playlist_continue.serve.main:app
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq

from playlist_continue.data.lfm2b import derive_sessions
from playlist_continue.data.splits import make_splits
from playlist_continue.models.cascade import CascadeRecommender


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cascade and save for serving.")
    parser.add_argument("--events", required=True, help="Path to processed events parquet")
    parser.add_argument("--output", default="models/cascade.pkl", help="Output model path")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--retrieval-k", type=int, default=500)
    parser.add_argument("--rerank-k", type=int, default=100)
    args = parser.parse_args()

    print("Loading events...")
    events = pq.read_table(args.events)
    n_items = int(events.column("track_id").max().as_py()) + 1

    print("Deriving sessions...")
    sessioned = derive_sessions(events)

    print("Making splits...")
    train, _val, _eval = make_splits(sessioned, val_fraction=0.1, seed=42)

    print(f"Training cascade (n_items={n_items:,}, embed_dim={args.embed_dim})...")
    t0 = time.time()
    model = CascadeRecommender(
        n_items=n_items,
        retrieval_k=args.retrieval_k,
        rerank_k=args.rerank_k,
        embed_dim=args.embed_dim,
    )
    model.fit(train, two_tower_epochs=args.epochs, sasrec_epochs=args.epochs)
    print(f"Training done in {time.time() - t0:.1f}s")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
