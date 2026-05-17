import pytest

from playlist_continue.eval.metrics import ndcg_at_k, r_precision, recall_at_k


def test_r_precision_perfect():
    assert r_precision([1, 2, 3, 4, 5], {1, 2, 3}) == pytest.approx(1.0)


def test_r_precision_zero():
    assert r_precision([4, 5, 6], {1, 2, 3}) == pytest.approx(0.0)


def test_r_precision_partial():
    assert r_precision([1, 5, 6, 2], {1, 2, 3, 4}) == pytest.approx(0.5)


def test_r_precision_empty_gt():
    assert r_precision([1, 2, 3], set()) == pytest.approx(0.0)


def test_ndcg_at_k_perfect():
    assert ndcg_at_k([1, 2, 3, 4, 5], {1, 2, 3}, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_zero():
    assert ndcg_at_k([4, 5, 6], {1, 2, 3}, k=3) == pytest.approx(0.0)


def test_ndcg_at_k_partial():
    score = ndcg_at_k([3, 1, 2], {1, 2}, k=3)
    assert 0 < score < 1.0


def test_recall_at_k_perfect():
    gt = {1, 2, 3}
    recs = list(range(1, 501))
    assert recall_at_k(recs, gt, k=500) == pytest.approx(1.0)


def test_recall_at_k_zero():
    assert recall_at_k([1, 2, 3], {10, 20, 30}, k=500) == pytest.approx(0.0)


def test_recall_at_k_partial():
    assert recall_at_k([1, 2, 5, 6], {1, 2, 3, 4}, k=4) == pytest.approx(0.5)


def test_recall_at_k_empty_gt():
    assert recall_at_k([1, 2, 3], set(), k=500) == pytest.approx(0.0)
