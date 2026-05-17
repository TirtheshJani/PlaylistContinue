import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from playlist_continue.data.freeze import freeze_eval_set
from playlist_continue.data.lfm2b import derive_sessions


def test_freeze_eval_set_returns_table(tiny_events_table):
    sessioned = derive_sessions(tiny_events_table)
    result = freeze_eval_set(sessioned, seed=42)
    assert isinstance(result, pa.Table)


def test_freeze_eval_set_has_required_columns(tiny_events_table):
    sessioned = derive_sessions(tiny_events_table)
    result = freeze_eval_set(sessioned, seed=42)
    for col in ("user_id", "track_id", "timestamp", "session_id"):
        assert col in result.schema.names, f"missing column: {col}"


def test_freeze_eval_set_is_deterministic(tiny_events_table):
    sessioned = derive_sessions(tiny_events_table)
    r1 = freeze_eval_set(sessioned, seed=42)
    r2 = freeze_eval_set(sessioned, seed=42)
    assert r1.equals(r2), "freeze_eval_set must be deterministic"


def test_freeze_eval_set_writes_and_reads_parquet(tiny_events_table, tmp_path):
    sessioned = derive_sessions(tiny_events_table)
    out = tmp_path / "eval_set.parquet"
    eval_set = freeze_eval_set(sessioned, seed=42, output_path=str(out))
    assert out.exists()
    loaded = pq.read_table(str(out))
    assert loaded.equals(eval_set)


def test_freeze_eval_set_no_duplicate_sessions_per_user(tiny_events_table):
    sessioned = derive_sessions(tiny_events_table)
    result = freeze_eval_set(sessioned, seed=42)
    user_session_pairs = set(
        zip(result.column("user_id").to_pylist(), result.column("session_id").to_pylist())
    )
    users = set(result.column("user_id").to_pylist())
    assert len(user_session_pairs) == len(users), "each user must have exactly one eval session"
