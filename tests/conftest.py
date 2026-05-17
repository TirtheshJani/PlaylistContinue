import numpy as np
import pyarrow as pa
import pytest


@pytest.fixture
def tiny_events_table() -> pa.Table:
    """50 events across 5 users and 10 tracks for fast unit tests."""
    rng = np.random.default_rng(42)
    n = 50
    return pa.table(
        {
            "user_id": pa.array(rng.integers(0, 5, n).tolist(), type=pa.int32()),
            "artist_id": pa.array(rng.integers(0, 8, n).tolist(), type=pa.int32()),
            "track_id": pa.array(rng.integers(0, 10, n).tolist(), type=pa.int32()),
            "timestamp": pa.array(
                sorted(rng.integers(1_600_000_000, 1_700_000_000, n).tolist()),
                type=pa.int64(),
            ),
        }
    )
