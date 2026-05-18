import pyarrow as pa

from playlist_continue.data.lfm2b import derive_sessions
from playlist_continue.data.splits import make_splits


def test_make_splits_returns_three_tables(tiny_events_table):
    sessioned = derive_sessions(tiny_events_table)
    train, val, eval_set = make_splits(sessioned, val_fraction=0.1, seed=42)
    assert isinstance(train, pa.Table)
    assert isinstance(val, pa.Table)
    assert isinstance(eval_set, pa.Table)


def test_eval_set_contains_one_session_per_user(tiny_events_table):
    sessioned = derive_sessions(tiny_events_table)
    train, val, eval_set = make_splits(sessioned, val_fraction=0.1, seed=42)
    eval_users = set(eval_set.column("user_id").to_pylist())
    user_session_pairs = set(
        zip(
            eval_set.column("user_id").to_pylist(),
            eval_set.column("session_id").to_pylist(),
            strict=False,
        )
    )
    assert len(user_session_pairs) == len(eval_users)


def test_eval_sessions_not_in_train(tiny_events_table):
    sessioned = derive_sessions(tiny_events_table)
    train, val, eval_set = make_splits(sessioned, val_fraction=0.1, seed=42)
    eval_sessions = set(eval_set.column("session_id").to_pylist())
    train_sessions = set(train.column("session_id").to_pylist())
    assert eval_sessions.isdisjoint(train_sessions)


def test_eval_sessions_not_in_val(tiny_events_table):
    sessioned = derive_sessions(tiny_events_table)
    train, val, eval_set = make_splits(sessioned, val_fraction=0.1, seed=42)
    eval_sessions = set(eval_set.column("session_id").to_pylist())
    val_sessions = set(val.column("session_id").to_pylist())
    assert eval_sessions.isdisjoint(val_sessions)
