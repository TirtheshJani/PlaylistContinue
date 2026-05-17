"""Eval harness: run a Recommender against the frozen eval split."""
from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from playlist_continue.eval.metrics import ndcg_at_k, r_precision, recall_at_k
from playlist_continue.models.base import Recommender


def evaluate(
    model: Recommender,
    eval_set: pa.Table,
    seed_fraction: float = 0.5,
    n_recs: int = 500,
) -> dict[str, float]:
    """Compute mean R-precision, NDCG@20, Recall@500 over eval sessions.

    For each session: first `seed_fraction` tracks are the seed;
    the remainder are ground truth.
    """
    session_ids = pc.unique(eval_set.column("session_id")).to_pylist()
    rp_scores: list[float] = []
    ndcg_scores: list[float] = []
    recall_scores: list[float] = []

    for sid in session_ids:
        mask = pc.equal(eval_set.column("session_id"), sid)
        session = eval_set.filter(mask)
        order = pc.sort_indices(session, sort_keys=[("timestamp", "ascending")])
        tracks = pc.take(session.column("track_id"), order).to_pylist()

        n_seed = max(1, int(len(tracks) * seed_fraction))
        seed = tracks[:n_seed]
        ground_truth = set(tracks[n_seed:])

        if not ground_truth:
            continue

        recs = model.recommend(seed, n=n_recs)
        rp_scores.append(r_precision(recs, ground_truth))
        ndcg_scores.append(ndcg_at_k(recs, ground_truth, k=20))
        recall_scores.append(recall_at_k(recs, ground_truth, k=500))

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "r_precision": _mean(rp_scores),
        "ndcg_at_20": _mean(ndcg_scores),
        "recall_at_500": _mean(recall_scores),
    }
