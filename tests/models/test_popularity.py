import pyarrow as pa

from playlist_continue.models.popularity import PopularityRecommender


def _make_events(track_ids: list[int]) -> pa.Table:
    n = len(track_ids)
    return pa.table(
        {
            "track_id": pa.array(track_ids, type=pa.int32()),
            "user_id": pa.array(list(range(n)), type=pa.int32()),
            "timestamp": pa.array(list(range(n)), type=pa.int64()),
            "session_id": pa.array(list(range(n)), type=pa.int64()),
        }
    )


def test_popularity_recommends_most_played_first():
    events = _make_events([1, 1, 1, 2, 2, 3])
    model = PopularityRecommender()
    model.fit(events)
    recs = model.recommend(seed_tracks=[], n=3)
    assert recs[0] == 1
    assert recs[1] == 2
    assert recs[2] == 3


def test_popularity_excludes_seed_tracks():
    events = _make_events([1, 1, 2, 3])
    model = PopularityRecommender()
    model.fit(events)
    recs = model.recommend(seed_tracks=[1], n=10)
    assert 1 not in recs


def test_popularity_respects_n():
    events = _make_events([1, 2, 3, 4, 5])
    model = PopularityRecommender()
    model.fit(events)
    recs = model.recommend(seed_tracks=[], n=3)
    assert len(recs) == 3
