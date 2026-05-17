"""Abstract recommender interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pyarrow as pa


class Recommender(ABC):
    @abstractmethod
    def fit(self, events: pa.Table) -> None:
        """Train on the provided event table."""

    @abstractmethod
    def recommend(self, seed_tracks: list[int], n: int = 500) -> list[int]:
        """Return up to n track_ids ranked by relevance."""
