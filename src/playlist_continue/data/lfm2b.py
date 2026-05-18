"""LFM-2b ingestion and session derivation."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

_GAP_SECONDS = 1800


def derive_sessions(
    events: pa.Table,
    gap_seconds: int = _GAP_SECONDS,
) -> pa.Table:
    """Add a `session_id` column by splitting each user's stream on inactivity gaps.

    Expects columns: user_id (int32), track_id (int32), timestamp (int64).
    Session ids are globally unique monotonic integers.
    """
    idx = pc.sort_indices(events, sort_keys=[("user_id", "ascending"), ("timestamp", "ascending")])
    events = pc.take(events, idx)

    users = events.column("user_id").to_pylist()
    timestamps = events.column("timestamp").to_pylist()

    session_ids: list[int] = []
    current_session = 0
    prev_user: int | None = None
    prev_ts: int = 0

    for user, ts in zip(users, timestamps, strict=False):
        if user != prev_user:
            current_session += 1
            prev_user = user
            prev_ts = ts
        elif ts - prev_ts > gap_seconds:
            current_session += 1
        session_ids.append(current_session)
        prev_ts = ts

    return events.append_column("session_id", pa.array(session_ids, type=pa.int64()))
