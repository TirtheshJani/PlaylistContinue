import pyarrow as pa
import pytest

from playlist_continue.models.sasrec import SASRecReranker


@pytest.fixture
def tiny_events():
    return pa.table(
        {
            "user_id": pa.array([0, 0, 0, 1, 1, 1], type=pa.int32()),
            "track_id": pa.array([0, 1, 2, 2, 3, 4], type=pa.int32()),
            "timestamp": pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
            "session_id": pa.array([1, 1, 1, 2, 2, 2], type=pa.int64()),
        }
    )


def test_rerank_returns_same_candidates_as_set(tiny_events):
    model = SASRecReranker(n_items=5, max_seq_len=4, n_heads=2, n_layers=1, embed_dim=8)
    model.fit(tiny_events, epochs=1, batch_size=2)
    candidates = [2, 3, 4]
    reranked = model.rerank(seed_tracks=[0, 1], candidates=candidates)
    assert set(reranked) == set(candidates)


def test_rerank_length_matches_candidates(tiny_events):
    model = SASRecReranker(n_items=5, max_seq_len=4, n_heads=2, n_layers=1, embed_dim=8)
    model.fit(tiny_events, epochs=1, batch_size=2)
    candidates = [2, 3, 4]
    reranked = model.rerank(seed_tracks=[0, 1], candidates=candidates)
    assert len(reranked) == len(candidates)


def test_rerank_returns_ints(tiny_events):
    model = SASRecReranker(n_items=5, max_seq_len=4, n_heads=2, n_layers=1, embed_dim=8)
    model.fit(tiny_events, epochs=1, batch_size=2)
    reranked = model.rerank(seed_tracks=[0], candidates=[1, 2, 3])
    assert all(isinstance(r, int) for r in reranked)
