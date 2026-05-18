"""Subsample LFM-2b to top-N tracks and top-K users by play count."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc


def compute_top_ids(table: pa.Table, column: str, n: int) -> set[int]:
    """Return the n ids with the highest row count in `column`."""
    value_counts = pc.value_counts(table.column(column))
    values = value_counts.field("values")
    counts = value_counts.field("counts")
    order = pc.sort_indices(counts, sort_keys=[("x", "descending")])
    top = pc.take(values, order[:n])
    return set(top.to_pylist())


def filter_events(
    table: pa.Table,
    top_tracks: set[int],
    top_users: set[int],
) -> pa.Table:
    """Keep only rows where track_id in top_tracks AND user_id in top_users."""
    track_mask = pc.is_in(
        table.column("track_id"),
        value_set=pa.array(list(top_tracks), type=pa.int32()),
    )
    user_mask = pc.is_in(
        table.column("user_id"),
        value_set=pa.array(list(top_users), type=pa.int32()),
    )
    return table.filter(pc.and_(track_mask, user_mask))
