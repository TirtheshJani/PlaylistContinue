import pyarrow as pa
import pytest

from playlist_continue.models.two_tower_sasrec import TwoTowerSASRecRecommender


@pytest.fixture
def events() -> pa.Table:
    tracks = [0, 1, 2, 3, 0, 1, 4, 5, 2, 3, 6, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 4, 2, 6]
    users = [u for u in range(6) for _ in range(4)]
    sids = [s for s in range(1, 7) for _ in range(4)]
    return pa.table(
        {
            "user_id": pa.array(users, type=pa.int32()),
            "track_id": pa.array(tracks, type=pa.int32()),
            "timestamp": pa.array(list(range(24)), type=pa.int64()),
            "session_id": pa.array(sids, type=pa.int64()),
        }
    )


def test_fit_runs(events):
    model = TwoTowerSASRecRecommender(n_items=8, embed_dim=8, max_seq_len=4, n_heads=2, n_layers=1)
    model.fit(events, epochs=1, batch_size=4)


def test_recommend_returns_list(events):
    model = TwoTowerSASRecRecommender(n_items=8, embed_dim=8, max_seq_len=4, n_heads=2, n_layers=1)
    model.fit(events, epochs=1, batch_size=4)
    recs = model.recommend(seed_tracks=[0, 1], n=4)
    assert isinstance(recs, list)


def test_recommend_excludes_seed(events):
    model = TwoTowerSASRecRecommender(n_items=8, embed_dim=8, max_seq_len=4, n_heads=2, n_layers=1)
    model.fit(events, epochs=1, batch_size=4)
    recs = model.recommend(seed_tracks=[0, 1], n=8)
    assert 0 not in recs
    assert 1 not in recs


def test_recommend_respects_n(events):
    model = TwoTowerSASRecRecommender(n_items=8, embed_dim=8, max_seq_len=4, n_heads=2, n_layers=1)
    model.fit(events, epochs=1, batch_size=4)
    recs = model.recommend(seed_tracks=[0], n=3)
    assert len(recs) <= 3


def test_recommend_returns_ints(events):
    model = TwoTowerSASRecRecommender(n_items=8, embed_dim=8, max_seq_len=4, n_heads=2, n_layers=1)
    model.fit(events, epochs=1, batch_size=4)
    recs = model.recommend(seed_tracks=[0], n=5)
    assert all(isinstance(r, int) for r in recs)
