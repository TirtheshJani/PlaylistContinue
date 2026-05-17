"""SASRec: causal self-attention reranker over listening sequences."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class _SASRecNet(nn.Module):
    def __init__(
        self,
        n_items: int,
        max_seq_len: int,
        embed_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # item 0 is the padding token
        self.item_embed = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.max_seq_len = max_seq_len

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, S) int ids, 0=pad. Returns (B, D) sequence representation."""
        B, S = seq.shape
        pos = torch.arange(S, device=seq.device).unsqueeze(0).expand(B, S)
        x = self.item_embed(seq) + self.pos_embed(pos)
        mask = torch.triu(torch.ones(S, S, device=seq.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        lengths = (seq != 0).sum(dim=1).clamp(min=1) - 1
        return F.normalize(x[torch.arange(B, device=seq.device), lengths], dim=-1)


class SASRecReranker:
    """Reranks a candidate list using a SASRec sequence model."""

    def __init__(
        self,
        n_items: int,
        max_seq_len: int = 50,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        device: str | None = None,
    ) -> None:
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._net = _SASRecNet(
            n_items=n_items,
            max_seq_len=max_seq_len,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        ).to(self.device)

    def _pad_sequence(self, track_ids: list[int]) -> torch.Tensor:
        """Left-pad to max_seq_len with 0; track ids shifted +1 (0 is pad)."""
        seq = track_ids[-self.max_seq_len :]
        padded = [0] * (self.max_seq_len - len(seq)) + [t + 1 for t in seq]
        return torch.tensor(padded, dtype=torch.long)

    def fit(
        self,
        events: pa.Table,
        epochs: int = 5,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> None:
        session_ids = pc.unique(events.column("session_id")).to_pylist()
        inputs, targets = [], []
        for sid in session_ids:
            mask = pc.equal(events.column("session_id"), sid)
            session = events.filter(mask)
            order = pc.sort_indices(session, sort_keys=[("timestamp", "ascending")])
            tracks = pc.take(session.column("track_id"), order).to_pylist()
            for i in range(1, len(tracks)):
                inputs.append(self._pad_sequence(tracks[:i]))
                targets.append(tracks[i])

        if not inputs:
            return

        inp_t = torch.stack(inputs)
        tgt_t = torch.tensor(targets, dtype=torch.long)
        loader = DataLoader(TensorDataset(inp_t, tgt_t), batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self._net.parameters(), lr=lr)

        self._net.train()
        for _ in range(epochs):
            for inp_batch, tgt_batch in loader:
                inp_batch = inp_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)
                user_vec = self._net(inp_batch)
                all_items = F.normalize(self._net.item_embed.weight[1:], dim=-1)
                logits = user_vec @ all_items.T
                loss = F.cross_entropy(logits, tgt_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._net.eval()

    def rerank(self, seed_tracks: list[int], candidates: list[int]) -> list[int]:
        seq = self._pad_sequence(seed_tracks).unsqueeze(0).to(self.device)
        with torch.no_grad():
            user_vec = self._net(seq).squeeze(0)
            cand_tensor = torch.tensor(
                [c + 1 for c in candidates], dtype=torch.long, device=self.device
            )
            cand_vecs = F.normalize(self._net.item_embed(cand_tensor), dim=-1)
            scores = (cand_vecs @ user_vec).cpu().numpy()
        return [candidates[i] for i in np.argsort(-scores)]
