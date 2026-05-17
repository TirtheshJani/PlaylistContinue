import pyarrow as pa

from playlist_continue.data.subsample import compute_top_ids, filter_events


def test_compute_top_ids_returns_top_n_by_count():
    events = pa.table(
        {
            "track_id": pa.array([1, 1, 1, 2, 2, 3], type=pa.int32()),
            "user_id": pa.array([1, 2, 3, 1, 2, 1], type=pa.int32()),
        }
    )
    top_tracks = compute_top_ids(events, column="track_id", n=2)
    assert set(top_tracks) == {1, 2}


def test_compute_top_ids_respects_n():
    events = pa.table(
        {
            "track_id": pa.array([1, 1, 2, 2, 3], type=pa.int32()),
            "user_id": pa.array([1, 2, 1, 2, 1], type=pa.int32()),
        }
    )
    top_tracks = compute_top_ids(events, column="track_id", n=1)
    assert len(top_tracks) == 1


def test_filter_events_removes_outside_top_ids(tiny_events_table):
    top_tracks = compute_top_ids(tiny_events_table, column="track_id", n=3)
    top_users = compute_top_ids(tiny_events_table, column="user_id", n=2)
    filtered = filter_events(tiny_events_table, top_tracks=top_tracks, top_users=top_users)
    assert set(filtered.column("track_id").to_pylist()).issubset(top_tracks)
    assert set(filtered.column("user_id").to_pylist()).issubset(top_users)


def test_filter_events_retains_all_columns(tiny_events_table):
    top_tracks = compute_top_ids(tiny_events_table, column="track_id", n=5)
    top_users = compute_top_ids(tiny_events_table, column="user_id", n=3)
    filtered = filter_events(tiny_events_table, top_tracks=top_tracks, top_users=top_users)
    assert set(filtered.schema.names) == set(tiny_events_table.schema.names)
