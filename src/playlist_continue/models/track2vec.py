"""Track2Vec: Word2Vec trained on listening session sequences."""

from __future__ import annotations

import logging

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from gensim.models import Word2Vec

from playlist_continue.models.base import Recommender

logging.getLogger("gensim").setLevel(logging.WARNING)


class Track2VecRecommender(Recommender):
    """Recommends tracks using Word2Vec embeddings learned from session sequences."""

    def __init__(
        self,
        vector_size: int = 128,
        window: int = 5,
        min_count: int = 5,
        epochs: int = 10,
        seed: int = 42,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed
        self._model: Word2Vec | None = None
        self._known_tracks: set[int] = set()

    def fit(self, events: pa.Table) -> None:
        session_ids = pc.unique(events.column("session_id")).to_pylist()  # type: ignore[attr-defined]
        sequences: list[list[str]] = []
        for sid in session_ids:
            mask = pc.equal(events.column("session_id"), sid)  # type: ignore[attr-defined]
            session = events.filter(mask)
            order = pc.sort_indices(session, sort_keys=[("timestamp", "ascending")])  # type: ignore[attr-defined]
            tracks = pc.take(session.column("track_id"), order).to_pylist()  # type: ignore[no-untyped-call]
            # Word2Vec expects string tokens.
            sequences.append([str(t) for t in tracks])

        self._model = Word2Vec(
            sentences=sequences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=1,
            seed=self.seed,
            epochs=self.epochs,
        )
        self._known_tracks = {int(k) for k in self._model.wv.key_to_index}

    def recommend(self, seed_tracks: list[int], n: int = 500) -> list[int]:
        assert self._model is not None, "call fit() first"
        seed_set = set(seed_tracks)

        known_seeds = [t for t in seed_tracks if t in self._known_tracks]
        if not known_seeds:
            # Fall back to arbitrary ordering of known tracks.
            return [t for t in self._known_tracks if t not in seed_set][:n]

        # Mean-pool seed vectors as query.
        seed_vecs = np.array([self._model.wv[str(t)] for t in known_seeds])
        query = seed_vecs.mean(axis=0)

        # Find most similar tracks excluding seeds.
        exclude = {str(t) for t in seed_tracks}
        similar = self._model.wv.similar_by_vector(query, topn=n + len(seed_tracks))
        result = [int(word) for word, _ in similar if word not in exclude]
        return result[:n]
