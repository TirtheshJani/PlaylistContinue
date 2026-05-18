"""MPD challenge evaluation metrics: R-precision, NDCG@k, Recall@k."""

from __future__ import annotations

import math


def r_precision(recommendations: list[int], ground_truth: set[int]) -> float:
    """Fraction of relevant items in the top-|ground_truth| recommendations."""
    r = len(ground_truth)
    if r == 0:
        return 0.0
    hits = sum(1 for item in recommendations[:r] if item in ground_truth)
    return hits / r


def ndcg_at_k(recommendations: list[int], ground_truth: set[int], k: int = 20) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    if not ground_truth:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2) for i, item in enumerate(recommendations[:k]) if item in ground_truth
    )
    n_ideal = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    return dcg / idcg if idcg > 0.0 else 0.0


def recall_at_k(recommendations: list[int], ground_truth: set[int], k: int = 500) -> float:
    """Fraction of relevant items found in the top-k recommendations."""
    if not ground_truth:
        return 0.0
    hits = sum(1 for item in recommendations[:k] if item in ground_truth)
    return hits / len(ground_truth)
