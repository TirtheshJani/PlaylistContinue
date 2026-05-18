#!/usr/bin/env python3
"""End-to-end pipeline run on a synthetic LFM-2b-shaped corpus.

Stages: generate -> ingest -> derive sessions -> freeze eval -> compare 6 models.
Writes the metric table to artifacts/comparison_results.json so the blog-post
comparison is reproducible from the recorded seed + hyperparams.

Usage:
    python scripts/full_run.py
    python scripts/full_run.py --epochs 5 --embed-dim 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from playlist_continue.data.freeze import freeze_eval_set
from playlist_continue.data.ingest import write_subsampled_parquet
from playlist_continue.data.lfm2b import derive_sessions
from playlist_continue.data.synthetic import generate_synthetic_lfm2b
from playlist_continue.eval.comparison import run_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end synthetic run.")
    parser.add_argument("--raw-dir", default="data/raw_synthetic")
    parser.add_argument("--events-out", default="data/processed/events.parquet")
    parser.add_argument("--eval-out", default="data/eval/eval_set.parquet")
    parser.add_argument("--results-out", default="artifacts/comparison_results.json")
    parser.add_argument("--n-users", type=int, default=2_000)
    parser.add_argument("--n-tracks", type=int, default=5_000)
    parser.add_argument("--n-artists", type=int, default=500)
    parser.add_argument("--n-events", type=int, default=200_000)
    parser.add_argument("--n-chunks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    print(f"[1/5] Generating synthetic corpus -> {args.raw_dir}")
    t0 = time.time()
    chunks = generate_synthetic_lfm2b(
        output_dir=args.raw_dir,
        n_users=args.n_users,
        n_tracks=args.n_tracks,
        n_artists=args.n_artists,
        n_events=args.n_events,
        n_chunks=args.n_chunks,
        seed=args.seed,
    )
    print(f"  {len(chunks)} chunks in {time.time() - t0:.1f}s")

    print(f"[2/5] Ingesting -> {args.events_out}")
    t0 = time.time()
    Path(args.events_out).parent.mkdir(parents=True, exist_ok=True)
    write_subsampled_parquet(
        chunk_paths=[str(p) for p in chunks],
        output_path=args.events_out,
        n_top_tracks=args.n_tracks,
        n_top_users=args.n_users,
    )
    events = pq.read_table(args.events_out)  # type: ignore[no-untyped-call]
    print(f"  {len(events):,} events in {time.time() - t0:.1f}s")

    print("[3/5] Deriving sessions")
    t0 = time.time()
    sessioned = derive_sessions(events)
    n_sessions = len(set(sessioned.column("session_id").to_pylist()))
    print(f"  {n_sessions:,} sessions in {time.time() - t0:.1f}s")

    print(f"[4/5] Freezing eval split -> {args.eval_out}")
    t0 = time.time()
    Path(args.eval_out).parent.mkdir(parents=True, exist_ok=True)
    eval_set = freeze_eval_set(sessioned, seed=args.seed, output_path=args.eval_out)
    print(f"  {len(eval_set):,} eval events in {time.time() - t0:.1f}s")

    print(f"[5/5] Comparing 6 models (embed_dim={args.embed_dim}, epochs={args.epochs})")
    t0 = time.time()
    metrics = run_comparison(
        sessioned,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    comparison_seconds = time.time() - t0

    header = f"{'Model':<22}  {'R-prec':>8}  {'NDCG@20':>8}  {'Rec@500':>8}"
    print("\n" + header)
    print("-" * len(header))
    for model, m in metrics.items():
        print(
            f"{model:<22}  {m['r_precision']:>8.4f}  "
            f"{m['ndcg_at_20']:>8.4f}  {m['recall_at_500']:>8.4f}"
        )

    payload = {
        "dataset": "synthetic_lfm2b",
        "n_users": args.n_users,
        "n_tracks": args.n_tracks,
        "n_artists": args.n_artists,
        "n_events": int(len(events)),
        "n_sessions": n_sessions,
        "n_eval_events": int(len(eval_set)),
        "seed": args.seed,
        "embed_dim": args.embed_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "comparison_seconds": round(comparison_seconds, 2),
        "metrics": metrics,
    }
    Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_out).write_text(json.dumps(payload, indent=2))
    print(f"\nResults -> {args.results_out}")


if __name__ == "__main__":
    main()
