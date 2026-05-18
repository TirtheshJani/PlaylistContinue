from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from playlist_continue.data.ingest import parse_lfm2b_chunk, write_subsampled_parquet


def _write_tsv(path: Path, rows: list[tuple]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")


def test_parse_lfm2b_chunk_returns_table(tmp_path):
    tsv = tmp_path / "chunk.tsv"
    _write_tsv(
        tsv,
        [
            (1, 10, "ArtistA", 100, "TrackA", 1609459200),
            (2, 20, "ArtistB", 200, "TrackB", 1609459260),
        ],
    )
    table = parse_lfm2b_chunk(str(tsv))
    assert isinstance(table, pa.Table)


def test_parse_lfm2b_chunk_has_correct_columns(tmp_path):
    tsv = tmp_path / "chunk.tsv"
    _write_tsv(tsv, [(1, 10, "ArtistA", 100, "TrackA", 1609459200)])
    table = parse_lfm2b_chunk(str(tsv))
    assert set(table.schema.names) == {"user_id", "artist_id", "track_id", "timestamp"}


def test_parse_lfm2b_chunk_correct_types(tmp_path):
    tsv = tmp_path / "chunk.tsv"
    _write_tsv(tsv, [(1, 10, "ArtistA", 100, "TrackA", 1609459200)])
    table = parse_lfm2b_chunk(str(tsv))
    assert table.schema.field("user_id").type == pa.int32()
    assert table.schema.field("artist_id").type == pa.int32()
    assert table.schema.field("track_id").type == pa.int32()
    assert table.schema.field("timestamp").type == pa.int64()


def test_parse_lfm2b_chunk_correct_values(tmp_path):
    tsv = tmp_path / "chunk.tsv"
    _write_tsv(tsv, [(42, 99, "Art", 777, "Track", 1700000000)])
    table = parse_lfm2b_chunk(str(tsv))
    assert table.column("user_id").to_pylist() == [42]
    assert table.column("track_id").to_pylist() == [777]
    assert table.column("timestamp").to_pylist() == [1700000000]


def test_write_subsampled_parquet(tmp_path):
    # Two chunk files
    tsv1 = tmp_path / "chunk1.tsv"
    tsv2 = tmp_path / "chunk2.tsv"
    rows1 = [(u, u % 3, "A", u * 10, "T", 1000000 + u) for u in range(6)]
    rows2 = [(u, u % 3, "A", u * 10, "T", 1000000 + u) for u in range(6, 10)]
    _write_tsv(tsv1, rows1)
    _write_tsv(tsv2, rows2)

    out = tmp_path / "events.parquet"
    write_subsampled_parquet(
        chunk_paths=[str(tsv1), str(tsv2)],
        output_path=str(out),
        n_top_tracks=5,
        n_top_users=4,
    )
    assert out.exists()
    result = pq.read_table(str(out))
    assert "user_id" in result.schema.names
    assert "track_id" in result.schema.names
    assert len(result) > 0
