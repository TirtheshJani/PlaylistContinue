import pyarrow as pa

from playlist_continue.models.artist_cooccurrence import ArtistCooccurrenceRecommender


def _events() -> pa.Table:
    # 3 sessions, 2 artists, 4 tracks
    # session 1: tracks 0,1 (artist 0), track 2 (artist 1)
    # session 2: tracks 0 (artist 0), track 3 (artist 1)
    # session 3: tracks 2 (artist 1), track 3 (artist 1)
    return pa.table(
        {
            "user_id": pa.array([0, 0, 0, 1, 1, 2, 2], type=pa.int32()),
            "artist_id": pa.array([0, 0, 1, 0, 1, 1, 1], type=pa.int32()),
            "track_id": pa.array([0, 1, 2, 0, 3, 2, 3], type=pa.int32()),
            "timestamp": pa.array([1, 2, 3, 4, 5, 6, 7], type=pa.int64()),
            "session_id": pa.array([1, 1, 1, 2, 2, 3, 3], type=pa.int64()),
        }
    )


def test_fit_and_recommend_returns_list():
    model = ArtistCooccurrenceRecommender()
    model.fit(_events())
    recs = model.recommend(seed_tracks=[0], n=10)
    assert isinstance(recs, list)


def test_recommend_excludes_seed_tracks():
    model = ArtistCooccurrenceRecommender()
    model.fit(_events())
    recs = model.recommend(seed_tracks=[0, 1], n=10)
    assert 0 not in recs
    assert 1 not in recs


def test_recommend_respects_n():
    model = ArtistCooccurrenceRecommender()
    model.fit(_events())
    recs = model.recommend(seed_tracks=[], n=2)
    assert len(recs) <= 2


def test_recommend_prefers_cooccurring_artists():
    # seed track 0 is artist 0; artist 1 co-occurs in sessions 1 and 2
    # tracks 2, 3 are from artist 1 and should appear in recs
    model = ArtistCooccurrenceRecommender()
    model.fit(_events())
    recs = model.recommend(seed_tracks=[0], n=10)
    # at least one of artist 1's tracks should appear
    assert any(t in recs for t in [2, 3])


def test_recommend_returns_ints():
    model = ArtistCooccurrenceRecommender()
    model.fit(_events())
    recs = model.recommend(seed_tracks=[0], n=5)
    assert all(isinstance(r, int) for r in recs)
