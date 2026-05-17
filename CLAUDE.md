# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

The repository is a greenfield project. Only `README.md`, `LICENSE`, and `.claude/skills/` exist on the active branch. All architectural details below describe the *intended* system, not code that has been written yet. When adding the first modules, prefer extending this file with concrete commands as they materialize rather than inventing structure ahead of need.

The active implementation plan lives at `/root/.claude/plans/the-spotify-million-playlist-vectorized-brook.md`.

## Skills available in this repo (read these before doing the matching kind of work)

Located in `.claude/skills/`:

- **writing-plans**: use before touching code, when given a spec or multi-step task. Plans must be bite-sized, name files to touch, follow DRY / YAGNI / TDD, and assume the executor knows nothing about this codebase.
- **executing-plans**: use when you have a written plan to execute. Load it, review critically, execute, report.
- **dispatching-parallel-agents**: use when 2+ tasks are independent (different test files, different subsystems). One agent per problem domain, concurrent.
- **test-driven-development**: use for any feature or bugfix. Write the test first, watch it fail, write minimal code to pass. The TDD scope here is the **data pipeline and eval harness**, not training loops.
- **karpathy-guidelines**: behavioral checks against common LLM coding mistakes. Think before coding, surface assumptions, make surgical changes, define verifiable success criteria.

When you start a task that matches one of these, announce it (e.g. "I'm using the writing-plans skill...") and follow the SKILL.md instructions.

## What this project is

A playlist-continuation recommender framed against the original Spotify Million Playlist Dataset (MPD) challenge spec, but built on **LFM-2b** (2.4 B Last.fm listening events: user, artist, track, timestamp) since Spotify withdrew MPD from public distribution in 2024. Positioned as "what Spotify Horizon actually does, on the closest open dataset."

Evaluation uses the original MPD challenge metrics: **R-precision, NDCG@20, Recall@500**. A seed track list must produce a continuation.

## Architecture: three-stage cascade

The recommender is a retrieve, rerank, score cascade. Each stage exists because the previous one cannot do its job at the next stage's quality and latency point. Do not collapse stages without a reason.

1. **Two-tower retrieval (candidate generation).** User-tower and item-tower trained with **in-batch sampled-softmax** in PyTorch. Item embeddings indexed in **FAISS** (`IndexFlatIP`, with `IndexIVFPQ` as a stretch optimization) for ANN lookup at serve time. The only stage that sees the full catalog.
2. **Sequential reranker (SASRec).** Causal self-attention over the user's recent listening sequence, rescoring the top-N candidates from stage 1. BERT4Rec is a stretch comparison for the blog post; it does not gate the live demo.
3. **LightGBM final-scoring head.** Takes the two-tower retrieval score, the SASRec score, and content features (artist overlap, popularity, recency, etc.) and produces the final ranking.

The cascade ordering matters: a sequence model cannot score the whole catalog, and LightGBM cannot generate candidates. Keep stage responsibilities separate.

## Compute and data constraints (these shape design choices)

- **GPU**: laptop RTX 4080, 16 GB VRAM. Supplemented by Kaggle 30 GPU-hours per week.
- **System RAM**: 16 GB. This is the binding constraint for data prep; stream parquet via `pyarrow.dataset` row groups and never pandas-load the full subsampled corpus.
- **Data subsample is locked in**: top 1 M tracks by play count, top 50 K users by activity, around 200 M events. At this size the item embedding table (1 M x 128 fp32 = 512 MB) fits on GPU; CPU sharding is not needed.
- LFM-2b is openly redistributable; ingestion code is aware of its schema (user, artist, track, timestamp) and does not assume MPD-style playlist grouping. Sessions are derived from timestamps with a 30-minute inactivity gap.

## Tooling

- Python 3.11+, **uv** for env management, **pyproject.toml** (not `requirements.txt`), src/ layout.
- **ruff + mypy** for lint and types, **pytest** for tests, **pre-commit** hooks.
- CI on GitHub Actions **ubuntu-latest only** (public repo, unlimited minutes).
- Docker multi-arch via `buildx --platform linux/arm64` for the serve image.
- No em dashes in any output, code comment, commit message, PR body, or doc.

## Serving

- **Backend**: FastAPI on Oracle Cloud Always Free (4 OCPU ARM Ampere, 24 GB RAM). FAISS retrieval runs in-process. ARM compatibility matters when picking native deps; verify `faiss-cpu`, `lightgbm`, and any torch inference path build and run on aarch64 before adding them. Fallback: `hnswlib`, kept as a one-file swap behind the abstracted index API.
- **Frontend**: Vite + React + TypeScript on Vercel free tier. Accepts a seed track list, hits the FastAPI service, renders continuations.
- The training pipeline and the serving stack are different deployment targets; keep the inference path (item index + reranker + LGBM head) independently packageable from training code.

## Deliverables that constrain choices

- Public GitHub repo, live demo (Oracle + Vercel), technical blog post comparing the architectures (popularity, artist co-occurrence, Word2Vec, two-tower, two-tower+SASRec, full cascade), optional Kaggle notebook.
- The blog-post angle means model comparisons must be runnable side-by-side and reported on the **same frozen eval split** (`data/eval/eval_set.parquet`) with the MPD-style metrics. Build the eval harness in Phase 1 so each new model is measured the same way.

## Branch policy

Active development branch is `claude/music-recommendation-system-a1qbZ`. Do not push to `main` without explicit instruction.
