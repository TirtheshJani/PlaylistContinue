#!/usr/bin/env python3
"""Generate a deterministic synthetic LFM-2b-shaped TSV corpus for local end-to-end runs.

Real LFM-2b is ~100 GB and not redistributed inline. This produces a small corpus that
matches the raw LFM-2b schema so the rest of the pipeline (ingest, freeze, compare)
runs unchanged.

Usage:
    python scripts/generate_synthetic_lfm2b.py --output-dir data/raw_synthetic/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playlist_continue.data.synthetic import generate_synthetic_lfm2b


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic LFM-2b TSV chunks.")
    parser.add_argument("--output-dir", default="data/raw_synthetic", help="Where to write chunks")
    parser.add_argument("--n-users", type=int, default=2_000)
    parser.add_argument("--n-tracks", type=int, default=5_000)
    parser.add_argument("--n-artists", type=int, default=500)
    parser.add_argument("--n-events", type=int, default=200_000)
    parser.add_argument("--n-chunks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    paths = generate_synthetic_lfm2b(
        output_dir=args.output_dir,
        n_users=args.n_users,
        n_tracks=args.n_tracks,
        n_artists=args.n_artists,
        n_events=args.n_events,
        n_chunks=args.n_chunks,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    total_bytes = sum(Path(p).stat().st_size for p in paths)
    print(f"Wrote {len(paths)} chunk(s), {total_bytes / 1e6:.1f} MB in {elapsed:.1f}s")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
