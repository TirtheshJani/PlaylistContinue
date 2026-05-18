"""Abstract base class for ANN index implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseIndex(ABC):
    """Common interface for FAISS and hnswlib index implementations."""

    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        """Build index from (N, dim) float32 unit-norm vectors."""

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        """Return (n_queries, k) array of item indices."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist index to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk."""
