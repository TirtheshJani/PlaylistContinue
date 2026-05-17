import pyarrow as pa

from playlist_continue.data.lfm2b import derive_sessions


def test_derive_sessions_splits_on_30min_gap():
    events = pa.table(
        {
            "user_id": pa.array([1, 1, 1], type=pa.int32()),
            "track_id": pa.array([10, 20, 30], type=pa.int32()),
            "timestamp": pa.array(
                [1_000_000, 1_000_000 + 1000, 1_000_000 + 1000 + 1801],
                type=pa.int64(),
            ),
        }
    )
    result = derive_sessions(events, gap_seconds=1800)
    sids = result.column("session_id").to_pylist()
    assert sids[0] == sids[1], "first two events same session"
    assert sids[1] != sids[2], "third event is a new session"


def test_derive_sessions_different_users_always_different_sessions():
    events = pa.table(
        {
            "user_id": pa.array([1, 2], type=pa.int32()),
            "track_id": pa.array([10, 10], type=pa.int32()),
            "timestamp": pa.array([1_000_000, 1_000_000], type=pa.int64()),
        }
    )
    result = derive_sessions(events, gap_seconds=1800)
    sids = result.column("session_id").to_pylist()
    assert sids[0] != sids[1]


def test_derive_sessions_output_has_session_id_column():
    events = pa.table(
        {
            "user_id": pa.array([1], type=pa.int32()),
            "track_id": pa.array([10], type=pa.int32()),
            "timestamp": pa.array([1_000_000], type=pa.int64()),
        }
    )
    result = derive_sessions(events)
    assert "session_id" in result.schema.names
