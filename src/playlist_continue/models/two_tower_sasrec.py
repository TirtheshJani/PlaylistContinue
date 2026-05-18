"""Two-tower retrieval + SASRec reranker (no LightGBM head)."""

from __future__ import annotations

import pyarrow as pa

from playlist_continue.models.base import Recommender
from playlist_continue.models.sasrec import SASRecReranker
from playlist_continue.models.two_tower import TwoTowerModel


class TwoTowerSASRecRecommender(Recommender):
    """Stage 1 (two-tower) + Stage 2 (SASRec) without the LightGBM scoring head."""

    def __init__(
        self,
        n_items: int,
        embed_dim: int = 128,
        max_seq_len: int = 50,
        n_heads: int = 4,
        n_layers: int = 2,
        retrieval_k: int = 500,
    ) -> None:
        self.two_tower = TwoTowerModel(n_items=n_items, embed_dim=embed_dim)
        self.sasrec = SASRecReranker(
            n_items=n_items,
            max_seq_len=max_seq_len,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )
        self.retrieval_k = retrieval_k

    def fit(
        self,
        events: pa.Table,
        epochs: int = 5,
        batch_size: int = 1024,
    ) -> None:
        self.two_tower.fit(events, epochs=epochs, batch_size=batch_size)
        self.sasrec.fit(events, epochs=epochs)

    def recommend(self, seed_tracks: list[int], n: int = 500) -> list[int]:
        candidates = self.two_tower.recommend(seed_tracks, n=self.retrieval_k)
        if not candidates:
            return []
        return self.sasrec.rerank(seed_tracks, candidates)[:n]
