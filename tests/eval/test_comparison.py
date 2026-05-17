import pyarrow as pa

from playlist_continue.data.lfm2b import derive_sessions
from playlist_continue.eval.comparison import run_comparison

EXPECTED_MODELS = {
    "popularity",
    "artist_cooccurrence",
    "track2vec",
    "two_tower",
    "two_tower_sasrec",
    "cascade",
}


def test_run_comparison_returns_all_model_keys(tiny_events_table):
    sessioned = derive_sessions(tiny_events_table)
    results = run_comparison(
        sessioned,
        embed_dim=8,
        epochs=1,
        batch_size=4,
        max_seq_len=4,
        n_heads=2,
        n_layers=1,
        w2v_min_count=1,
    )
    assert set(results.keys()) == EXPECTED_MODELS


def test_run_comparison_metric_keys(tiny_events_table):
    sessioned = derive_sessions(tiny_events_table)
    results = run_comparison(
        sessioned,
        embed_dim=8,
        epochs=1,
        batch_size=4,
        max_seq_len=4,
        n_heads=2,
        n_layers=1,
        w2v_min_count=1,
    )
    for model_name, metrics in results.items():
        assert "r_precision" in metrics, f"{model_name} missing r_precision"
        assert "ndcg_at_20" in metrics, f"{model_name} missing ndcg_at_20"
        assert "recall_at_500" in metrics, f"{model_name} missing recall_at_500"


def test_run_comparison_scores_in_range(tiny_events_table):
    sessioned = derive_sessions(tiny_events_table)
    results = run_comparison(
        sessioned,
        embed_dim=8,
        epochs=1,
        batch_size=4,
        max_seq_len=4,
        n_heads=2,
        n_layers=1,
        w2v_min_count=1,
    )
    for model_name, metrics in results.items():
        for metric, score in metrics.items():
            assert 0.0 <= score <= 1.0, f"{model_name}/{metric}={score} out of range"
