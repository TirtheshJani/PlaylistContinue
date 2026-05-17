"""Tests for model save/load persistence."""
import pyarrow as pa
import pytest
import numpy as np


def _tiny_events() -> pa.Table:
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


def test_two_tower_save_load_same_recommendations(tmp_path):
    from playlist_continue.models.two_tower import TwoTowerModel
    events = _tiny_events()
    model = TwoTowerModel(n_items=8, embed_dim=8)
    model.fit(events, epochs=1, batch_size=4)
    path = str(tmp_path / "tt.pkl")
    model.save(path)
    loaded = TwoTowerModel.load(path)
    assert model.recommend([0, 1], n=3) == loaded.recommend([0, 1], n=3)


def test_sasrec_save_load_same_reranking(tmp_path):
    from playlist_continue.models.sasrec import SASRecReranker
    events = _tiny_events()
    model = SASRecReranker(n_items=8, max_seq_len=4, n_heads=2, n_layers=1, embed_dim=8)
    model.fit(events, epochs=1, batch_size=4)
    path = str(tmp_path / "sas.pkl")
    model.save(path)
    loaded = SASRecReranker.load(path)
    cands = [2, 3, 4, 5]
    assert model.rerank([0, 1], cands) == loaded.rerank([0, 1], cands)


def test_lgbm_save_load_same_scores(tmp_path):
    from playlist_continue.models.lgbm_head import LGBMScoringHead
    rng = np.random.default_rng(0)
    X = rng.random((30, 3)).astype(np.float32)
    y = rng.integers(0, 2, 30).astype(np.float32)
    model = LGBMScoringHead(n_estimators=10)
    model.fit(X, y)
    path = str(tmp_path / "lgbm.pkl")
    model.save(path)
    loaded = LGBMScoringHead.load(path)
    np.testing.assert_allclose(model.score(X), loaded.score(X), rtol=1e-5)


def test_cascade_save_load_returns_list(tmp_path):
    from playlist_continue.models.cascade import CascadeRecommender
    events = _tiny_events()
    model = CascadeRecommender(
        n_items=8, retrieval_k=6, rerank_k=4, embed_dim=8, max_seq_len=4, n_heads=2, n_layers=1
    )
    model.fit(events, two_tower_epochs=1, sasrec_epochs=1, two_tower_batch=4)
    path = str(tmp_path / "cascade.pkl")
    model.save(path)
    loaded = CascadeRecommender.load(path)
    recs = loaded.recommend([0, 1], n=3)
    assert isinstance(recs, list)
    assert all(isinstance(r, int) for r in recs)
