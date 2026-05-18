import pyarrow as pa
import pytest

from playlist_continue.models.two_tower import TwoTowerModel


@pytest.fixture
def tiny_events():
    return pa.table(
        {
            "user_id": pa.array([0, 0, 1, 1, 2, 2, 3, 3], type=pa.int32()),
            "track_id": pa.array([0, 1, 1, 2, 2, 3, 3, 4], type=pa.int32()),
            "timestamp": pa.array([1, 2, 3, 4, 5, 6, 7, 8], type=pa.int64()),
            "session_id": pa.array([1, 1, 2, 2, 3, 3, 4, 4], type=pa.int64()),
        }
    )


def test_two_tower_fit_runs(tiny_events):
    model = TwoTowerModel(n_items=5, embed_dim=8)
    model.fit(tiny_events, epochs=2, batch_size=4)


def test_two_tower_recommend_returns_correct_count(tiny_events):
    model = TwoTowerModel(n_items=5, embed_dim=8)
    model.fit(tiny_events, epochs=1, batch_size=4)
    recs = model.recommend(seed_tracks=[0, 1], n=3)
    assert len(recs) == 3


def test_two_tower_recommend_excludes_seed(tiny_events):
    model = TwoTowerModel(n_items=5, embed_dim=8)
    model.fit(tiny_events, epochs=1, batch_size=4)
    recs = model.recommend(seed_tracks=[0, 1], n=10)
    assert 0 not in recs
    assert 1 not in recs


def test_two_tower_recommend_returns_ints(tiny_events):
    model = TwoTowerModel(n_items=5, embed_dim=8)
    model.fit(tiny_events, epochs=1, batch_size=4)
    recs = model.recommend(seed_tracks=[0], n=3)
    assert all(isinstance(r, int) for r in recs)


def test_two_tower_builds_faiss_index_after_fit(tiny_events):
    from playlist_continue.index.base import BaseIndex

    model = TwoTowerModel(n_items=5, embed_dim=8)
    model.fit(tiny_events, epochs=1, batch_size=4)
    assert hasattr(model, "_index")
    assert isinstance(model._index, BaseIndex)
