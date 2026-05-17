"""Three-stage cascade: two-tower retrieval -> SASRec reranker -> LightGBM scorer."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
import torch.nn.functional as F

from playlist_continue.models.base import Recommender
from playlist_continue.models.lgbm_head import LGBMScoringHead
from playlist_continue.models.sasrec import SASRecReranker
from playlist_continue.models.two_tower import TwoTowerModel


class CascadeRecommender(Recommender):
    """Retrieve (two-tower) -> Rerank (SASRec) -> Score (LightGBM)."""

    def __init__(
        self,
        n_items: int,
        retrieval_k: int = 500,
        rerank_k: int = 100,
        embed_dim: int = 128,
        max_seq_len: int = 50,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        self.n_items = n_items
        self.retrieval_k = retrieval_k
        self.rerank_k = rerank_k
        self.two_tower = TwoTowerModel(n_items=n_items, embed_dim=embed_dim)
        self.sasrec = SASRecReranker(
            n_items=n_items,
            max_seq_len=max_seq_len,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )
        self.lgbm = LGBMScoringHead()
        self._popularity = np.zeros(n_items, dtype=np.float32)
        self._lgbm_trained = False

    def fit(
        self,
        events: pa.Table,
        two_tower_epochs: int = 5,
        sasrec_epochs: int = 5,
        two_tower_batch: int = 1024,
        sasrec_batch: int = 256,
    ) -> None:
        self.two_tower.fit(events, epochs=two_tower_epochs, batch_size=two_tower_batch)
        self.sasrec.fit(events, epochs=sasrec_epochs, batch_size=sasrec_batch)
        self._compute_popularity(events)
        X, y = self._generate_lgbm_data(events)
        # Only train LGBM if we have both positive and negative examples.
        if len(y) > 0 and 0 < y.sum() < len(y):
            self.lgbm.fit(X, y)
            self._lgbm_trained = True

    def _compute_popularity(self, events: pa.Table) -> None:
        total = len(events)
        if total == 0:
            return
        value_counts = pc.value_counts(events.column("track_id"))
        for struct in value_counts:
            tid = struct["values"].as_py()
            cnt = struct["counts"].as_py()
            if 0 <= tid < self.n_items:
                self._popularity[tid] = cnt / total

    def _retrieval_scores(self, seed_tracks: list[int], candidates: list[int]) -> np.ndarray:
        assert self.two_tower._item_vecs is not None
        if seed_tracks:
            seed_t = torch.tensor(seed_tracks, dtype=torch.long, device=self.two_tower.device)
            with torch.no_grad():
                user_vec = self.two_tower._net(seed_t.unsqueeze(0)).squeeze(0).cpu().numpy()
        else:
            user_vec = self.two_tower._item_vecs.mean(axis=0)
        return self.two_tower._item_vecs[candidates] @ user_vec

    def _sasrec_scores(self, seed_tracks: list[int], candidates: list[int]) -> np.ndarray:
        seq = self.sasrec._pad_sequence(seed_tracks).unsqueeze(0).to(self.sasrec.device)
        cand_t = torch.tensor(
            [c + 1 for c in candidates], dtype=torch.long, device=self.sasrec.device
        )
        with torch.no_grad():
            user_vec = self.sasrec._net(seq).squeeze(0)
            cand_vecs = F.normalize(self.sasrec._net.item_embed(cand_t), dim=-1)
            scores = (cand_vecs @ user_vec).cpu().numpy()
        return scores

    def _build_features(self, seed_tracks: list[int], candidates: list[int]) -> np.ndarray:
        ret_scores = self._retrieval_scores(seed_tracks, candidates)
        sas_scores = self._sasrec_scores(seed_tracks, candidates)
        pop_scores = self._popularity[candidates]
        return np.column_stack([ret_scores, sas_scores, pop_scores]).astype(np.float32)

    def _generate_lgbm_data(self, events: pa.Table) -> tuple[np.ndarray, np.ndarray]:
        session_ids = pc.unique(events.column("session_id")).to_pylist()
        X_rows: list[np.ndarray] = []
        y_rows: list[int] = []
        for sid in session_ids:
            mask = pc.equal(events.column("session_id"), sid)
            session = events.filter(mask)
            order = pc.sort_indices(session, sort_keys=[("timestamp", "ascending")])
            tracks = pc.take(session.column("track_id"), order).to_pylist()
            if len(tracks) < 2:
                continue
            n_seed = max(1, len(tracks) // 2)
            seed = tracks[:n_seed]
            ground_truth = set(tracks[n_seed:])
            candidates = self.two_tower.recommend(seed, n=self.retrieval_k)
            if not candidates:
                continue
            candidates = self.sasrec.rerank(seed, candidates)[: self.rerank_k]
            features = self._build_features(seed, candidates)
            for i, c in enumerate(candidates):
                X_rows.append(features[i])
                y_rows.append(1 if c in ground_truth else 0)
        if not X_rows:
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
        return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)

    def recommend(self, seed_tracks: list[int], n: int = 500) -> list[int]:
        # Stage 1: two-tower retrieval.
        candidates = self.two_tower.recommend(seed_tracks, n=self.retrieval_k)
        if not candidates:
            return []
        # Stage 2: SASRec rerank.
        candidates = self.sasrec.rerank(seed_tracks, candidates)[: self.rerank_k]
        if not candidates:
            return []
        # Stage 3: LightGBM scoring (if trained, else return SASRec order).
        if self._lgbm_trained:
            features = self._build_features(seed_tracks, candidates)
            ranked = self.lgbm.rank(features)
            candidates = [candidates[i] for i in ranked]
        return candidates[:n]

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "CascadeRecommender":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
