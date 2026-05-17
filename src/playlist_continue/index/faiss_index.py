"""FAISS IndexFlatIP wrapper for item retrieval."""
from __future__ import annotations

import numpy as np
import faiss

from playlist_continue.index.base import BaseIndex


class FaissIndex(BaseIndex):
    """Inner-product flat index (cosine after L2-normalizing vectors)."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._index: faiss.IndexFlatIP | None = None

    def build(self, vectors: np.ndarray) -> None:
        """Build from (N, dim) float32 unit-norm vectors."""
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(vectors.astype(np.float32))

    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        """Return (n_queries, k) array of item indices."""
        assert self._index is not None, "call build() or load() first"
        _, indices = self._index.search(query.astype(np.float32), k)
        return indices

    def save(self, path: str) -> None:
        assert self._index is not None
        faiss.write_index(self._index, path)

    def load(self, path: str) -> None:
        self._index = faiss.read_index(path)
