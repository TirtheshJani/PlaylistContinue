import pyarrow as pa

from playlist_continue.data.lfm2b import derive_sessions
from playlist_continue.data.splits import make_splits
from playlist_continue.eval.harness import evaluate
from playlist_continue.models.base import Recommender


class ConstantRecommender(Recommender):
    """Always returns tracks 0..n-1 regardless of seed."""

    def fit(self, events: pa.Table) -> None:
        pass

    def recommend(self, seed_tracks: list[int], n: int = 500) -> list[int]:
        return list(range(n))


def test_evaluate_returns_all_metric_keys(tiny_events_table):
    sessioned = derive_sessions(tiny_events_table)
    train, val, eval_set = make_splits(sessioned, val_fraction=0.1, seed=42)
    model = ConstantRecommender()
    model.fit(train)
    results = evaluate(model, eval_set)
    assert "r_precision" in results
    assert "ndcg_at_20" in results
    assert "recall_at_500" in results


def test_evaluate_scores_in_valid_range(tiny_events_table):
    sessioned = derive_sessions(tiny_events_table)
    train, val, eval_set = make_splits(sessioned, val_fraction=0.1, seed=42)
    model = ConstantRecommender()
    model.fit(train)
    results = evaluate(model, eval_set)
    for metric, score in results.items():
        assert 0.0 <= score <= 1.0, f"{metric}={score} out of [0,1]"
