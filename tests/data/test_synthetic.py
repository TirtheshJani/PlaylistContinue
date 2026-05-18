from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from playlist_continue.data.ingest import write_subsampled_parquet
from playlist_continue.data.lfm2b import derive_sessions
from playlist_continue.data.synthetic import generate_synthetic_lfm2b


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def test_determinism(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    paths_a = generate_synthetic_lfm2b(
        dir_a,
        n_users=80,
        n_tracks=200,
        n_artists=20,
        n_events=1_500,
        n_chunks=3,
        seed=123,
    )
    paths_b = generate_synthetic_lfm2b(
        dir_b,
        n_users=80,
        n_tracks=200,
        n_artists=20,
        n_events=1_500,
        n_chunks=3,
        seed=123,
    )
    assert [p.name for p in paths_a] == [p.name for p in paths_b]
    for pa_path, pb_path in zip(paths_a, paths_b, strict=True):
        assert _sha256(pa_path) == _sha256(pb_path)


def test_schema_round_trips_through_ingest(tmp_path: Path) -> None:
    chunks = generate_synthetic_lfm2b(
        tmp_path / "raw",
        n_users=100,
        n_tracks=200,
        n_artists=20,
        n_events=2_000,
        n_chunks=2,
        seed=7,
    )
    out = tmp_path / "events.parquet"
    write_subsampled_parquet(
        chunk_paths=[str(p) for p in chunks],
        output_path=str(out),
        n_top_tracks=200,
        n_top_users=100,
    )
    assert out.exists()
    table = pq.read_table(str(out))
    assert table.schema.names == ["user_id", "artist_id", "track_id", "timestamp"]
    assert table.schema.field("user_id").type == pa.int32()
    assert table.schema.field("artist_id").type == pa.int32()
    assert table.schema.field("track_id").type == pa.int32()
    assert table.schema.field("timestamp").type == pa.int64()
    assert len(table) > 0


def test_popularity_skew(tmp_path: Path) -> None:
    chunks = generate_synthetic_lfm2b(
        tmp_path,
        n_users=200,
        n_tracks=400,
        n_artists=40,
        n_events=10_000,
        n_chunks=2,
        seed=11,
    )
    counter: Counter[int] = Counter()
    for chunk in chunks:
        for line in chunk.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            counter[int(parts[3])] += 1
    total = sum(counter.values())
    top_n = max(1, int(round(0.05 * len(counter))))
    top_count = sum(c for _, c in counter.most_common(top_n))
    assert top_count / total >= 0.35


def test_session_gaps_split(tmp_path: Path) -> None:
    chunks = generate_synthetic_lfm2b(
        tmp_path / "raw",
        n_users=100,
        n_tracks=200,
        n_artists=20,
        n_events=2_000,
        n_chunks=2,
        seed=3,
    )
    out = tmp_path / "events.parquet"
    write_subsampled_parquet(
        chunk_paths=[str(p) for p in chunks],
        output_path=str(out),
        n_top_tracks=200,
        n_top_users=100,
    )
    events = pq.read_table(str(out))
    with_sessions = derive_sessions(events)
    session_ids = with_sessions.column("session_id").to_pylist()
    session_counts = Counter(session_ids)
    lengths = sorted(session_counts.values())
    median_len = lengths[len(lengths) // 2]
    n_users = len(set(with_sessions.column("user_id").to_pylist()))
    n_sessions = len(session_counts)
    assert median_len > 1
    assert n_sessions > n_users


def test_chunk_count(tmp_path: Path) -> None:
    chunks = generate_synthetic_lfm2b(
        tmp_path,
        n_users=50,
        n_tracks=100,
        n_artists=10,
        n_events=600,
        n_chunks=5,
        seed=1,
    )
    assert len(chunks) == 5
    for i, p in enumerate(chunks, start=1):
        assert p.name == f"chunk_{i:04d}.tsv"
        assert p.exists()
