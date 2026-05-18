import numpy as np

from playlist_continue.models.lgbm_head import LGBMScoringHead


def test_fit_and_score_shape():
    rng = np.random.default_rng(42)
    X = rng.random((50, 4)).astype(np.float32)
    y = rng.integers(0, 2, 50).astype(np.float32)
    model = LGBMScoringHead(n_estimators=10)
    model.fit(X, y)
    scores = model.score(X)
    assert scores.shape == (50,)


def test_score_in_probability_range():
    rng = np.random.default_rng(0)
    X = rng.random((30, 4)).astype(np.float32)
    y = rng.integers(0, 2, 30).astype(np.float32)
    model = LGBMScoringHead(n_estimators=10)
    model.fit(X, y)
    scores = model.score(X)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_rank_returns_descending_order():
    rng = np.random.default_rng(1)
    X = rng.random((10, 4)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.float32)
    model = LGBMScoringHead(n_estimators=10)
    model.fit(X, y)
    ranked = model.rank(X)
    scores = model.score(X)
    assert scores[ranked[0]] >= scores[ranked[-1]]
