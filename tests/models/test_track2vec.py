import pyarrow as pa
import pytest

from playlist_continue.models.track2vec import Track2VecRecommender


@pytest.fixture
def events_with_sessions() -> pa.Table:
    # 4 sessions, 6 tracks - enough for Word2Vec to learn something
    return pa.table(
        {
            "user_id": pa.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], type=pa.int32()),
            "artist_id": pa.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0], type=pa.int32()),
            "track_id": pa.array([0, 1, 2, 0, 2, 3, 1, 2, 4, 3, 4, 5], type=pa.int32()),
            "timestamp": pa.array(list(range(12)), type=pa.int64()),
            "session_id": pa.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], type=pa.int64()),
        }
    )


def test_track2vec_fit_runs(events_with_sessions):
    model = Track2VecRecommender(vector_size=8, min_count=1)
    model.fit(events_with_sessions)


def test_track2vec_recommend_returns_list(events_with_sessions):
    model = Track2VecRecommender(vector_size=8, min_count=1)
    model.fit(events_with_sessions)
    recs = model.recommend(seed_tracks=[0, 1], n=3)
    assert isinstance(recs, list)


def test_track2vec_recommend_excludes_seed(events_with_sessions):
    model = Track2VecRecommender(vector_size=8, min_count=1)
    model.fit(events_with_sessions)
    seed = [0, 1]
    recs = model.recommend(seed_tracks=seed, n=10)
    assert 0 not in recs
    assert 1 not in recs


def test_track2vec_recommend_respects_n(events_with_sessions):
    model = Track2VecRecommender(vector_size=8, min_count=1)
    model.fit(events_with_sessions)
    recs = model.recommend(seed_tracks=[0], n=2)
    assert len(recs) <= 2


def test_track2vec_recommend_returns_ints(events_with_sessions):
    model = Track2VecRecommender(vector_size=8, min_count=1)
    model.fit(events_with_sessions)
    recs = model.recommend(seed_tracks=[0], n=5)
    assert all(isinstance(r, int) for r in recs)
