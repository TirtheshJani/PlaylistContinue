"""End-to-end integration smoke test: synthetic data through full pipeline."""

import pyarrow as pa


def _make_events(n_users: int = 10, n_tracks: int = 12, sessions_per_user: int = 3) -> pa.Table:
    """Generate synthetic events large enough to train all 6 models."""
    import numpy as np

    rng = np.random.default_rng(0)
    user_ids, track_ids, timestamps, session_ids = [], [], [], []
    sid = 1
    t = 1_600_000_000
    for u in range(n_users):
        for _ in range(sessions_per_user):
            n_events = rng.integers(4, 8)
            tracks = rng.integers(0, n_tracks, n_events).tolist()
            for track in tracks:
                user_ids.append(u)
                track_ids.append(int(track))
                timestamps.append(t)
                session_ids.append(sid)
                t += rng.integers(60, 300)
            t += 2000  # gap > 30 min between sessions
            sid += 1

    return pa.table(
        {
            "user_id": pa.array(user_ids, type=pa.int32()),
            "artist_id": pa.array([tid % 4 for tid in track_ids], type=pa.int32()),
            "track_id": pa.array(track_ids, type=pa.int32()),
            "timestamp": pa.array(timestamps, type=pa.int64()),
            "session_id": pa.array(session_ids, type=pa.int64()),
        }
    )


EXPECTED_MODELS = {
    "popularity",
    "artist_cooccurrence",
    "track2vec",
    "two_tower",
    "two_tower_sasrec",
    "cascade",
}


def test_run_comparison_all_models_present():
    """All 6 models must appear in comparison results."""
    from playlist_continue.eval.comparison import run_comparison

    events = _make_events()
    results = run_comparison(
        events,
        embed_dim=8,
        epochs=1,
        batch_size=8,
        max_seq_len=6,
        n_heads=2,
        n_layers=1,
        w2v_min_count=1,
        retrieval_k=10,
        rerank_k=6,
    )
    assert set(results.keys()) == EXPECTED_MODELS


def test_run_comparison_all_scores_valid():
    """All metric scores must be floats in [0.0, 1.0]."""
    from playlist_continue.eval.comparison import run_comparison

    events = _make_events()
    results = run_comparison(
        events,
        embed_dim=8,
        epochs=1,
        batch_size=8,
        max_seq_len=6,
        n_heads=2,
        n_layers=1,
        w2v_min_count=1,
        retrieval_k=10,
        rerank_k=6,
    )
    for model, metrics in results.items():
        for metric, score in metrics.items():
            assert isinstance(score, float), f"{model}/{metric} is not float"
            assert 0.0 <= score <= 1.0, f"{model}/{metric}={score} out of [0,1]"


def test_api_recommend_with_popularity_model():
    """FastAPI /recommend endpoint returns valid response."""
    from fastapi.testclient import TestClient

    from playlist_continue.models.popularity import PopularityRecommender
    from playlist_continue.serve.api import create_app

    events = _make_events()
    model = PopularityRecommender()
    model.fit(events)
    client = TestClient(create_app(model))

    resp = client.post("/recommend", json={"seed_tracks": [0, 1, 2], "n": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert "recommendations" in data
    recs = data["recommendations"]
    assert isinstance(recs, list)
    assert len(recs) <= 5
    assert 0 not in recs  # seed tracks excluded
    assert 1 not in recs
    assert 2 not in recs


def test_session_derivation_produces_sessions():
    """Session derivation assigns unique session ids across users."""
    from playlist_continue.data.lfm2b import derive_sessions

    raw = _make_events(n_users=3, n_tracks=8, sessions_per_user=2)
    # derive_sessions expects no pre-existing session_id column
    events = raw.select(["user_id", "artist_id", "track_id", "timestamp"])
    sessioned = derive_sessions(events)
    assert "session_id" in sessioned.schema.names
    unique_sessions = set(sessioned.column("session_id").to_pylist())
    # 3 users x 2 sessions each = at least 6 sessions
    assert len(unique_sessions) >= 6


def test_splits_eval_sessions_disjoint_from_train():
    """Eval sessions must never appear in training data."""
    from playlist_continue.data.lfm2b import derive_sessions
    from playlist_continue.data.splits import make_splits

    raw = _make_events()
    # derive_sessions expects no pre-existing session_id column
    events = raw.select(["user_id", "artist_id", "track_id", "timestamp"])
    sessioned = derive_sessions(events)
    train, val, eval_set = make_splits(sessioned, val_fraction=0.1, seed=42)

    eval_sids = set(eval_set.column("session_id").to_pylist())
    train_sids = set(train.column("session_id").to_pylist())
    assert eval_sids.isdisjoint(train_sids)
