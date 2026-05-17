# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

The repository is a greenfield project. Only `README.md` and `LICENSE` exist on `main`. All architectural details below describe the *intended* system, not code that has been written yet. When adding the first modules, prefer extending this file with concrete commands as they materialize rather than inventing structure ahead of need.

## What this project is

A playlist-continuation recommender framed against the original Spotify Million Playlist Dataset (MPD) challenge spec, but built on **LFM-2b** (2.4B Last.fm listening events: user, artist, track, timestamp) since Spotify withdrew MPD from public distribution in 2024. The system is positioned as "what Spotify Horizon actually does, on the closest open dataset."

Evaluation uses the original MPD challenge metrics: **Recall@K, NDCG@K, R-precision**. A seed track list must produce a continuation.

## Architecture: three-stage cascade

The recommender is a retrieve → rerank → score cascade. Each stage exists because the previous one cannot do its job at the next stage's quality/latency point — don't collapse stages without a reason.

1. **Two-tower retrieval (candidate generation).** User-tower and item-tower trained with **in-batch negatives** in PyTorch. Item embeddings are indexed in **FAISS** (or Annoy) for ANN lookup at serve time. This is the only stage that sees the full catalog.
   - Embedding tables for LFM-2b are large enough that they must be **sharded across CPU memory**; the GPU (target hardware: RTX 4080) only holds the active batch's slices. Expect ~4–6 hours per epoch.
2. **Sequential reranker.** **SASRec** or **BERT4Rec** over the user's recent listening sequence, rescoring the top-N candidates from stage 1. Fits trivially on the 4080 once the two-tower is done.
3. **LightGBM final-scoring head.** Takes the two-tower retrieval score, the sequential reranker score, and **content features** (artist, album, popularity, etc.) and produces the final ranking.

The cascade ordering matters: a sequence model cannot score the whole catalog, and LightGBM cannot generate candidates — keep stage responsibilities separate.

## Compute & data constraints (these shape design choices)

- **GPU**: laptop RTX 4080, supplemented by **Kaggle's 30 GPU-hours/week**. Total compute budget assumes the laptop is busy; long training jobs go to Kaggle.
- **CPU RAM is the bottleneck for embedding tables**, not GPU VRAM. Any change to vocabulary size, embedding dim, or sharding scheme should be evaluated against host-memory footprint first.
- LFM-2b is openly redistributable; keep ingestion code aware of its schema (user, artist, track, timestamp) and don't assume MPD-style playlist grouping — sessions/sequences must be derived from timestamps.

## Serving

- **Backend**: FastAPI on **Oracle Cloud Always Free** (4 OCPU ARM Ampere, 24 GB RAM). FAISS retrieval runs in-process. ARM compatibility matters when picking native deps — verify `faiss`, `lightgbm`, and any torch inference path build/run on aarch64 before adding them.
- **Frontend**: Vercel free tier. Accepts a seed track list, hits the FastAPI service, renders continuations.
- The two-tower training pipeline and the serving stack are different deployment targets; keep the inference path (item index + reranker + LGBM head) independently packageable from training code.

## Deliverables that constrain choices

- Public GitHub repo, live demo (Oracle + Vercel), technical blog post comparing the three architectures (two-tower vs. SASRec vs. BERT4Rec head-to-head), optional Kaggle notebook.
- The blog-post angle means model comparisons need to be runnable side-by-side and metrics reported on the **same eval split** with the MPD-style metrics named above. Build the eval harness early so each new model is measured the same way.

## Branch policy

Active development branch for this work is `claude/music-recommendation-system-a1qbZ`. Do not push to `main` without explicit instruction.
