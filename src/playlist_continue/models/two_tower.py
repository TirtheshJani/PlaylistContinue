"""Two-tower retrieval model with in-batch sampled-softmax."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from playlist_continue.index.base import BaseIndex
from playlist_continue.index.faiss_index import FaissIndex
from playlist_continue.models.base import Recommender


class _TwoTowerNet(nn.Module):
    def __init__(self, n_items: int, embed_dim: int) -> None:
        super().__init__()
        self.item_embed = nn.Embedding(n_items, embed_dim)

    def item_vectors(self) -> torch.Tensor:
        return F.normalize(self.item_embed.weight, dim=-1)

    def forward(self, seed_item_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pool seed embeddings into a user representation. (B, S) -> (B, D)"""
        embeds = self.item_embed(seed_item_ids)
        return F.normalize(embeds.mean(dim=1), dim=-1)


class TwoTowerModel(Recommender):
    """Item-to-item two-tower using seed track embeddings as user proxy."""

    def __init__(
        self,
        n_items: int,
        embed_dim: int = 128,
        device: str | None = None,
    ) -> None:
        self.n_items = n_items
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._net = _TwoTowerNet(n_items, embed_dim).to(self.device)
        self._item_vecs: np.ndarray | None = None
        self._index: BaseIndex | None = None

    def fit(
        self,
        events: pa.Table,
        epochs: int = 5,
        batch_size: int = 1024,
        lr: float = 1e-3,
    ) -> None:
        track_ids = torch.tensor(events.column("track_id").to_pylist(), dtype=torch.long)
        loader = DataLoader(
            TensorDataset(track_ids), batch_size=batch_size, shuffle=True, drop_last=True
        )
        opt = torch.optim.Adam(self._net.parameters(), lr=lr)

        self._net.train()
        for _ in range(epochs):
            for (batch,) in loader:
                batch = batch.to(self.device)
                vecs = F.normalize(self._net.item_embed(batch), dim=-1)
                logits = vecs @ vecs.T
                targets = torch.arange(batch.size(0), device=self.device)
                loss = F.cross_entropy(logits, targets)
                opt.zero_grad()
                loss.backward()
                opt.step()

        self._net.eval()
        with torch.no_grad():
            self._item_vecs = self._net.item_vectors().cpu().numpy()
        self._index = FaissIndex(dim=self._item_vecs.shape[1])
        self._index.build(self._item_vecs)

    def recommend(self, seed_tracks: list[int], n: int = 500) -> list[int]:
        assert self._index is not None, "call fit() first"
        seed_set = set(seed_tracks)
        if seed_tracks:
            seed_tensor = torch.tensor(seed_tracks, dtype=torch.long, device=self.device)
            with torch.no_grad():
                user_vec = self._net(seed_tensor.unsqueeze(0)).squeeze(0).cpu().numpy()
        else:
            assert self._item_vecs is not None
            user_vec = self._item_vecs.mean(axis=0)
        k = min(n + len(seed_set) + 1, self.n_items)
        indices = self._index.search(user_vec.reshape(1, -1), k=k)[0]
        return [int(i) for i in indices if i not in seed_set and i >= 0][:n]

    def save(self, path: str) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> TwoTowerModel:
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
