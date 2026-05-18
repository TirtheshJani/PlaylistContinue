"""Run all blog-post models on the same frozen eval split and return results."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from playlist_continue.data.splits import make_splits
from playlist_continue.eval.harness import evaluate
from playlist_continue.models.artist_cooccurrence import ArtistCooccurrenceRecommender
from playlist_continue.models.cascade import CascadeRecommender
from playlist_continue.models.popularity import PopularityRecommender
from playlist_continue.models.track2vec import Track2VecRecommender
from playlist_continue.models.two_tower import TwoTowerModel
from playlist_continue.models.two_tower_sasrec import TwoTowerSASRecRecommender


def run_comparison(
    sessioned: pa.Table,
    embed_dim: int = 128,
    epochs: int = 5,
    batch_size: int = 1024,
    max_seq_len: int = 50,
    n_heads: int = 4,
    n_layers: int = 2,
    w2v_min_count: int = 5,
    retrieval_k: int = 500,
    rerank_k: int = 100,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Train and evaluate all 6 blog-post models on the same frozen eval split.

    Returns: model_name -> {r_precision, ndcg_at_20, recall_at_500}
    """
    train, _val, eval_set = make_splits(sessioned, val_fraction=val_fraction, seed=seed)
    n_items = int(pc.max(sessioned.column("track_id")).as_py()) + 1  # type: ignore[attr-defined]

    results: dict[str, dict[str, float]] = {}

    pop = PopularityRecommender()
    pop.fit(train)
    results["popularity"] = evaluate(pop, eval_set)

    ac = ArtistCooccurrenceRecommender()
    ac.fit(train)
    results["artist_cooccurrence"] = evaluate(ac, eval_set)

    t2v = Track2VecRecommender(vector_size=embed_dim, min_count=w2v_min_count, epochs=epochs)
    t2v.fit(train)
    results["track2vec"] = evaluate(t2v, eval_set)

    tt = TwoTowerModel(n_items=n_items, embed_dim=embed_dim)
    tt.fit(train, epochs=epochs, batch_size=batch_size)
    results["two_tower"] = evaluate(tt, eval_set)

    tt_sas = TwoTowerSASRecRecommender(
        n_items=n_items,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        n_heads=n_heads,
        n_layers=n_layers,
        retrieval_k=retrieval_k,
    )
    tt_sas.fit(train, epochs=epochs, batch_size=batch_size)
    results["two_tower_sasrec"] = evaluate(tt_sas, eval_set)

    cascade = CascadeRecommender(
        n_items=n_items,
        retrieval_k=retrieval_k,
        rerank_k=rerank_k,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    cascade.fit(train, two_tower_epochs=epochs, sasrec_epochs=epochs, two_tower_batch=batch_size)
    results["cascade"] = evaluate(cascade, eval_set)

    return results
