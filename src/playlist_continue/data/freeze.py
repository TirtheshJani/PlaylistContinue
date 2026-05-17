"""Freeze the eval split to a canonical parquet file."""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from playlist_continue.data.splits import make_splits


def freeze_eval_set(
    sessioned: pa.Table,
    seed: int = 42,
    val_fraction: float = 0.1,
    output_path: str | None = None,
) -> pa.Table:
    """Extract the eval split (last session per user) and optionally write to parquet.

    Returns the eval split table. If output_path is given, also writes it to disk.
    This function is deterministic: same input + seed -> same output.
    """
    _train, _val, eval_set = make_splits(sessioned, val_fraction=val_fraction, seed=seed)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(eval_set, output_path)
    return eval_set
