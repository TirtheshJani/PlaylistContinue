"""LightGBM final-scoring head for the three-stage cascade."""
from __future__ import annotations

import numpy as np
import lightgbm as lgb


class LGBMScoringHead:
    """Pointwise relevance scorer using LightGBM."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.05) -> None:
        self._model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=31,
            verbose=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """X: (N, F) feature matrix, y: (N,) binary relevance labels."""
        self._model.fit(X, y)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return (N,) relevance probabilities."""
        return self._model.predict_proba(X)[:, 1]  # type: ignore[return-value]

    def rank(self, X: np.ndarray) -> list[int]:
        """Return row indices sorted by descending relevance score."""
        return list(np.argsort(-self.score(X)))

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "LGBMScoringHead":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
