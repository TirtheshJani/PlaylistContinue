"""HnswlibIndex: hnswlib-based ANN index, ARM-compatible alternative to FAISS."""
from __future__ import annotations

import numpy as np
import hnswlib

from playlist_continue.index.base import BaseIndex


class HnswlibIndex(BaseIndex):
    """Inner-product HNSW index using hnswlib (drop-in swap for FaissIndex)."""

    def __init__(self, dim: int, max_elements: int = 2_000_000, ef: int = 50, M: int = 16) -> None:
        self.dim = dim
        self.max_elements = max_elements
        self.ef = ef
        self.M = M
        self._index: hnswlib.Index | None = None
        self._n_items: int = 0

    def build(self, vectors: np.ndarray) -> None:
        """Build from (N, dim) float32 unit-norm vectors."""
        n = vectors.shape[0]
        self._index = hnswlib.Index(space="ip", dim=self.dim)
        self._index.init_index(max_elements=max(n, self.max_elements), ef_construction=200, M=self.M)
        self._index.add_items(vectors.astype(np.float32), list(range(n)))
        self._index.set_ef(self.ef)
        self._n_items = n

    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        """Return (n_queries, k) array of item indices."""
        assert self._index is not None, "call build() or load() first"
        k_clamped = min(k, self._n_items)
        labels, _ = self._index.knn_query(query.astype(np.float32), k=k_clamped)
        # Pad with -1 if k > n_items. Cast to int64 to allow sentinel -1.
        if k_clamped < k:
            labels = labels.astype(np.int64)
            pad = np.full((labels.shape[0], k - k_clamped), -1, dtype=np.int64)
            labels = np.concatenate([labels, pad], axis=1)
        return labels

    def save(self, path: str) -> None:
        assert self._index is not None
        self._index.save_index(path)

    def load(self, path: str) -> None:
        self._index = hnswlib.Index(space="ip", dim=self.dim)
        self._index.load_index(path, max_elements=self.max_elements)
        self._index.set_ef(self.ef)
        self._n_items = self._index.get_current_count()
