"""Train / val / eval split creation for the LFM-2b corpus."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def make_splits(
    sessioned: pa.Table,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[pa.Table, pa.Table, pa.Table]:
    """Split sessioned events into train, val, eval.

    Eval: last session per user (by session_id max).
    Train/val: remaining sessions split 90/10 by session.
    """
    user_ids = sessioned.column("user_id").to_pylist()
    session_ids = sessioned.column("session_id").to_pylist()

    last_session: dict[int, int] = {}
    for uid, sid in zip(user_ids, session_ids):
        if uid not in last_session or sid > last_session[uid]:
            last_session[uid] = sid

    eval_session_ids = set(last_session.values())
    remaining = sorted({sid for sid in set(session_ids) if sid not in eval_session_ids})

    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)
    n_val = max(1, int(len(remaining) * val_fraction))
    val_sids = set(remaining[:n_val])
    train_sids = set(remaining[n_val:])

    def _filter(sids: set[int]) -> pa.Table:
        mask = pc.is_in(
            sessioned.column("session_id"),
            value_set=pa.array(list(sids), type=pa.int64()),
        )
        return sessioned.filter(mask)

    return _filter(train_sids), _filter(val_sids), _filter(eval_session_ids)
