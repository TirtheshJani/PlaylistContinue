"""LFM-2b raw TSV ingestion: parse chunks, subsample, write parquet."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

from playlist_continue.data.subsample import compute_top_ids, filter_events

_COLUMN_NAMES = ["user_id", "artist_id", "artist_name", "track_id", "track_name", "timestamp"]
_KEEP_COLUMNS = ["user_id", "artist_id", "track_id", "timestamp"]
_CAST_TYPES: dict[str, pa.DataType] = {
    "user_id": pa.int32(),
    "artist_id": pa.int32(),
    "track_id": pa.int32(),
    "timestamp": pa.int64(),
}


def parse_lfm2b_chunk(path: str) -> pa.Table:
    """Read one LFM-2b TSV chunk and return a typed table with 4 columns."""
    read_opts = pa_csv.ReadOptions(
        column_names=_COLUMN_NAMES,
        autogenerate_column_names=False,
    )
    parse_opts = pa_csv.ParseOptions(delimiter="\t")
    convert_opts = pa_csv.ConvertOptions(
        include_columns=_KEEP_COLUMNS,
        column_types={col: _CAST_TYPES[col] for col in _KEEP_COLUMNS},
    )
    return pa_csv.read_csv(
        path,
        read_options=read_opts,
        parse_options=parse_opts,
        convert_options=convert_opts,
    )


def write_subsampled_parquet(
    chunk_paths: list[str],
    output_path: str,
    n_top_tracks: int = 1_000_000,
    n_top_users: int = 50_000,
) -> None:
    """Stream all chunks, determine top tracks/users, filter, write parquet.

    Two-pass approach to stay within 16 GB RAM:
    Pass 1: accumulate value counts (cheap - just two int32 columns).
    Pass 2: filter and write row groups.
    """
    # Pass 1: count tracks and users across all chunks.
    all_chunks: list[pa.Table] = []
    for path in chunk_paths:
        chunk = parse_lfm2b_chunk(path)
        all_chunks.append(chunk.select(["user_id", "track_id"]))

    counts_table = pa.concat_tables(all_chunks)
    top_tracks = compute_top_ids(counts_table, column="track_id", n=n_top_tracks)
    top_users = compute_top_ids(counts_table, column="user_id", n=n_top_users)
    del counts_table, all_chunks

    # Pass 2: filter each chunk and write to parquet.
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    try:
        for path in chunk_paths:
            chunk = parse_lfm2b_chunk(path)
            filtered = filter_events(chunk, top_tracks=top_tracks, top_users=top_users)
            if len(filtered) == 0:
                continue
            if writer is None:
                schema = filtered.schema
                writer = pq.ParquetWriter(output_path, schema=schema)
            writer.write_table(filtered)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        # No data passed the filter - write empty parquet.
        empty = pa.table({col: pa.array([], type=_CAST_TYPES[col]) for col in _KEEP_COLUMNS})
        pq.write_table(empty, output_path)
