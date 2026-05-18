#!/usr/bin/env python3
"""Ingest raw LFM-2b TSV chunks into a single subsampled parquet file.

Usage:
    python scripts/ingest_lfm2b.py --input-dir /data/lfm2b/raw/ \
        --output data/processed/events.parquet

The raw LFM-2b files are tab-separated with columns:
    user_id  artist_id  artist_name  track_id  track_name  timestamp

Download LFM-2b from: http://www.cp.jku.at/datasets/LFM-2b/
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

from playlist_continue.data.ingest import write_subsampled_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest LFM-2b raw TSV files to parquet.")
    parser.add_argument("--input-dir", required=True, help="Directory containing LFM-2b TSV chunks")
    parser.add_argument(
        "--output", default="data/processed/events.parquet", help="Output parquet path"
    )
    parser.add_argument(
        "--n-top-tracks", type=int, default=1_000_000, help="Keep top-N tracks by play count"
    )
    parser.add_argument(
        "--n-top-users", type=int, default=50_000, help="Keep top-K users by activity"
    )
    parser.add_argument("--glob", default="*.tsv", help="Glob pattern for TSV files")
    args = parser.parse_args()

    chunk_paths = sorted(glob.glob(str(Path(args.input_dir) / args.glob)))
    if not chunk_paths:
        raise SystemExit(f"No files matched {Path(args.input_dir) / args.glob}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(chunk_paths)} chunk files.")
    print(f"Subsampling: top {args.n_top_tracks:,} tracks, top {args.n_top_users:,} users.")
    t0 = time.time()
    write_subsampled_parquet(
        chunk_paths=chunk_paths,
        output_path=args.output,
        n_top_tracks=args.n_top_tracks,
        n_top_users=args.n_top_users,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
