import pyarrow as pa
import pytest

from playlist_continue.models.cascade import CascadeRecommender


@pytest.fixture
def cascade_events() -> pa.Table:
    # 6 sessions x 4 tracks each = 24 events across 6 users, 8 tracks
    # Enough for two-tower in-batch softmax (batch_size=4, drop_last=True needs >=4 rows)
    tracks = [0, 1, 2, 3, 0, 1, 4, 5, 2, 3, 6, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 4, 2, 6]
    users = [u for u in range(6) for _ in range(4)]
    sids = [s for s in range(1, 7) for _ in range(4)]
    ts = list(range(24))
    return pa.table(
        {
            "user_id": pa.array(users, type=pa.int32()),
            "track_id": pa.array(tracks, type=pa.int32()),
            "timestamp": pa.array(ts, type=pa.int64()),
            "session_id": pa.array(sids, type=pa.int64()),
        }
    )


def test_cascade_fit_runs(cascade_events):
    model = CascadeRecommender(n_items=8, retrieval_k=6, rerank_k=4, embed_dim=8, max_seq_len=4, n_heads=2, n_layers=1)
    model.fit(cascade_events, two_tower_epochs=1, sasrec_epochs=1, two_tower_batch=4)


def test_cascade_recommend_returns_list(cascade_events):
    model = CascadeRecommender(n_items=8, retrieval_k=6, rerank_k=4, embed_dim=8, max_seq_len=4, n_heads=2, n_layers=1)
    model.fit(cascade_events, two_tower_epochs=1, sasrec_epochs=1, two_tower_batch=4)
    recs = model.recommend(seed_tracks=[0, 1], n=4)
    assert isinstance(recs, list)


def test_cascade_recommend_excludes_seed(cascade_events):
    model = CascadeRecommender(n_items=8, retrieval_k=6, rerank_k=4, embed_dim=8, max_seq_len=4, n_heads=2, n_layers=1)
    model.fit(cascade_events, two_tower_epochs=1, sasrec_epochs=1, two_tower_batch=4)
    recs = model.recommend(seed_tracks=[0, 1], n=8)
    assert 0 not in recs
    assert 1 not in recs


def test_cascade_recommend_respects_n(cascade_events):
    model = CascadeRecommender(n_items=8, retrieval_k=6, rerank_k=4, embed_dim=8, max_seq_len=4, n_heads=2, n_layers=1)
    model.fit(cascade_events, two_tower_epochs=1, sasrec_epochs=1, two_tower_batch=4)
    recs = model.recommend(seed_tracks=[0], n=3)
    assert len(recs) <= 3


def test_cascade_returns_ints(cascade_events):
    model = CascadeRecommender(n_items=8, retrieval_k=6, rerank_k=4, embed_dim=8, max_seq_len=4, n_heads=2, n_layers=1)
    model.fit(cascade_events, two_tower_epochs=1, sasrec_epochs=1, two_tower_batch=4)
    recs = model.recommend(seed_tracks=[0], n=5)
    assert all(isinstance(r, int) for r in recs)
