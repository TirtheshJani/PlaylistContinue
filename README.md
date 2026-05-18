# PlaylistContinue

Playlist-continuation recommender on LFM-2b, evaluated with the original Spotify MPD
challenge metrics (R-precision, NDCG@20, Recall@500).

## Architecture: three-stage cascade

1. **Two-tower retrieval** (PyTorch, in-batch sampled-softmax) indexed in FAISS.
2. **SASRec reranker** rescoring the top-N candidates with causal self-attention.
3. **LightGBM scoring head** combining retrieval/rerank scores with content features.

Five non-cascade baselines (popularity, artist co-occurrence, Track2Vec,
two-tower alone, two-tower + SASRec) run on the same frozen eval split for the
blog-post comparison.

## End-to-end run on synthetic data

The real LFM-2b export (~100 GB, 2.4 B events) is not redistributed inline.
For a reproducible smoke run, generate a deterministic synthetic corpus that
matches the LFM-2b raw schema and exercise the full pipeline:

```bash
python -m pip install -e ".[dev]"
python scripts/full_run.py
```

Default settings: 2 000 users, 5 000 tracks, 500 artists, 200 000 events,
embed_dim=32, 2 epochs. Runs in ~5 to 10 minutes on a laptop CPU. Output lands in
`artifacts/comparison_results.json`.

Sample numbers from a recent run (seed=42, 200 K synthetic events):

| Model               | R-precision | NDCG@20 | Recall@500 |
|---------------------|------------:|--------:|-----------:|
| popularity          |      0.0478 |  0.0958 |     0.5163 |
| artist_cooccurrence |      0.0375 |  0.0581 |     0.2654 |
| track2vec           |      0.0075 |  0.0157 |     0.3533 |
| two_tower           |      0.0001 |  0.0011 |     0.0778 |
| two_tower_sasrec    |      0.0093 |  0.0193 |     0.0764 |
| cascade             |      0.0121 |  0.0226 |     0.0573 |

Numbers reflect a small synthetic corpus with limited training budget; ordering will
shift on the real subsampled LFM-2b corpus (1 M tracks / 50 K users / ~200 M events)
where the neural models converge.

## End-to-end run on real LFM-2b

Once real TSV chunks are on disk:

```bash
python scripts/ingest_lfm2b.py --input-dir /data/lfm2b/raw/ \
    --output data/processed/events.parquet
python scripts/freeze_eval_set.py --events data/processed/events.parquet
python scripts/compare_models.py --events data/processed/events.parquet \
    --out artifacts/comparison_results.json
```

## Serving

```bash
uvicorn playlist_continue.serve.main:app --host 0.0.0.0 --port 8000
cd frontend && npm run dev
```

Docker (multi-arch, ARM-compatible for Oracle Cloud Always Free):

```bash
docker build -t playlist-continue-serve .
```

## Development

```bash
ruff check . && ruff format --check .
python -m pytest tests/
```

CI runs lint, tests, and frontend type-check on every push.

## License

MIT, see `LICENSE`.
