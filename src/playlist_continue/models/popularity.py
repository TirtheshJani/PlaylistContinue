"""Global popularity baseline recommender."""
from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from playlist_continue.models.base import Recommender


class PopularityRecommender(Recommender):
    """Recommends tracks ranked by global play count, excluding seed tracks."""

    def __init__(self) -> None:
        self._ranked: list[int] = []

    def fit(self, events: pa.Table) -> None:
        value_counts = pc.value_counts(events.column("track_id"))
        values = value_counts.field("values")
        counts = value_counts.field("counts")
        order = pc.sort_indices(counts, sort_keys=[("x", "descending")])
        self._ranked = pc.take(values, order).to_pylist()

    def recommend(self, seed_tracks: list[int], n: int = 500) -> list[int]:
        seed_set = set(seed_tracks)
        return [t for t in self._ranked if t not in seed_set][:n]
