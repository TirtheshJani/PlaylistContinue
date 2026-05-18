"""Deterministic synthetic LFM-2b-shaped TSV generator for end-to-end testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def generate_synthetic_lfm2b(
    output_dir: str | Path,
    n_users: int = 2_000,
    n_tracks: int = 5_000,
    n_artists: int = 500,
    n_events: int = 200_000,
    n_chunks: int = 4,
    seed: int = 42,
    start_timestamp: int = 1_600_000_000,
    session_gap_seconds: int = 1800,
    avg_session_len: int = 8,
    user_artist_concentration: int = 10,
    track_popularity_alpha: float = 1.1,
) -> list[Path]:
    """Generate deterministic LFM-2b-shaped TSV chunks. Returns chunk paths."""
    if n_users < 1 or n_tracks < 1 or n_artists < 1 or n_events < 1 or n_chunks < 1:
        raise ValueError("counts must be positive")
    if n_artists > n_tracks:
        raise ValueError("n_artists must be <= n_tracks")
    if user_artist_concentration > n_artists:
        raise ValueError("user_artist_concentration must be <= n_artists")
    if avg_session_len < 1:
        raise ValueError("avg_session_len must be >= 1")
    if track_popularity_alpha <= 0.0:
        raise ValueError("track_popularity_alpha must be positive")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # WHY: each track maps to one artist by integer division so artists own a contiguous range.
    tracks_per_artist = max(1, n_tracks // n_artists)
    track_to_artist = np.minimum(
        np.arange(n_tracks, dtype=np.int64) // tracks_per_artist,
        n_artists - 1,
    ).astype(np.int32)

    # WHY: Zipfian weights over a random track permutation give heavy-tailed popularity without
    # rng.zipf's unbounded support.
    track_perm = rng.permutation(n_tracks)
    ranks = np.arange(1, n_tracks + 1, dtype=np.float64)
    zipf_weights = 1.0 / np.power(ranks, track_popularity_alpha)
    zipf_weights = zipf_weights / zipf_weights.sum()
    global_track_probs = np.empty(n_tracks, dtype=np.float64)
    global_track_probs[track_perm] = zipf_weights

    # WHY: each user has a small favorite-artist set; 70% of plays draw from their tracks so
    # artist-co-occurrence models can learn real signal.
    user_favorite_artists: list[np.ndarray] = [
        rng.choice(n_artists, size=user_artist_concentration, replace=False) for _ in range(n_users)
    ]
    artist_to_tracks: list[np.ndarray] = [
        np.where(track_to_artist == a)[0] for a in range(n_artists)
    ]

    # WHY: Zipfian over users so some users dominate the corpus; every user gets at least 2 plays.
    user_ranks = np.arange(1, n_users + 1, dtype=np.float64)
    user_weights = 1.0 / np.power(user_ranks, 1.3)
    user_weights = user_weights / user_weights.sum()
    user_perm = rng.permutation(n_users)
    user_probs = np.empty(n_users, dtype=np.float64)
    user_probs[user_perm] = user_weights

    min_per_user = 2
    base_events = min_per_user * n_users
    if n_events < base_events:
        raise ValueError(f"n_events must be >= {base_events} (min {min_per_user} per user)")
    extra = n_events - base_events
    extra_alloc = rng.multinomial(extra, user_probs) if extra > 0 else np.zeros(n_users, dtype=int)
    events_per_user = (np.full(n_users, min_per_user, dtype=np.int64) + extra_alloc).astype(int)

    all_user_ids: list[int] = []
    all_track_ids: list[int] = []
    all_artist_ids: list[int] = []
    all_timestamps: list[int] = []

    for user_id in range(n_users):
        n_user_events = int(events_per_user[user_id])
        favorites = user_favorite_artists[user_id]
        favorite_tracks = np.concatenate([artist_to_tracks[a] for a in favorites])
        fav_probs_raw = global_track_probs[favorite_tracks]
        fav_probs = fav_probs_raw / fav_probs_raw.sum()

        from_fav_mask = rng.random(n_user_events) < 0.7
        n_fav = int(from_fav_mask.sum())
        n_global = n_user_events - n_fav

        fav_picks = rng.choice(favorite_tracks, size=n_fav, p=fav_probs, replace=True)
        global_picks = rng.choice(n_tracks, size=n_global, p=global_track_probs, replace=True)
        user_tracks = np.empty(n_user_events, dtype=np.int64)
        user_tracks[from_fav_mask] = fav_picks
        user_tracks[~from_fav_mask] = global_picks

        # WHY: build timestamps by walking sessions of Poisson-mean avg_session_len, with small
        # in-session steps and large between-session gaps so derive_sessions splits cleanly.
        user_offset = int(rng.integers(0, 30 * 86400))
        current_ts = start_timestamp + user_offset
        timestamps = np.empty(n_user_events, dtype=np.int64)
        idx = 0
        while idx < n_user_events:
            session_len = max(1, int(rng.poisson(avg_session_len)))
            session_len = min(session_len, n_user_events - idx)
            for k in range(session_len):
                if k > 0:
                    current_ts += int(rng.integers(30, 301))
                timestamps[idx] = current_ts
                idx += 1
            if idx < n_user_events:
                current_ts += session_gap_seconds + int(rng.integers(1, 86_401))

        all_user_ids.extend([user_id] * n_user_events)
        all_track_ids.extend(user_tracks.tolist())
        all_artist_ids.extend(track_to_artist[user_tracks].tolist())
        all_timestamps.extend(timestamps.tolist())

    total = len(all_user_ids)
    user_arr = np.asarray(all_user_ids, dtype=np.int64)
    track_arr = np.asarray(all_track_ids, dtype=np.int64)
    artist_arr = np.asarray(all_artist_ids, dtype=np.int64)
    ts_arr = np.asarray(all_timestamps, dtype=np.int64)

    perm = rng.permutation(total)
    user_arr = user_arr[perm]
    track_arr = track_arr[perm]
    artist_arr = artist_arr[perm]
    ts_arr = ts_arr[perm]

    chunk_paths: list[Path] = []
    boundaries = np.linspace(0, total, n_chunks + 1, dtype=np.int64)
    for c in range(n_chunks):
        lo = int(boundaries[c])
        hi = int(boundaries[c + 1])
        chunk_path = out / f"chunk_{c + 1:04d}.tsv"
        with open(chunk_path, "w", encoding="utf-8", newline="\n") as f:
            for i in range(lo, hi):
                u = int(user_arr[i])
                a = int(artist_arr[i])
                t = int(track_arr[i])
                ts = int(ts_arr[i])
                f.write(f"{u}\t{a}\tartist_{a}\t{t}\ttrack_{t}\t{ts}\n")
        chunk_paths.append(chunk_path)

    return chunk_paths
