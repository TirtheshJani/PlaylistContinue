import pyarrow as pa
import pytest
from fastapi.testclient import TestClient

from playlist_continue.models.popularity import PopularityRecommender
from playlist_continue.serve.api import create_app


def _make_events() -> pa.Table:
    return pa.table(
        {
            "track_id": pa.array([1, 1, 2, 3, 4, 5], type=pa.int32()),
            "user_id": pa.array([1, 2, 1, 1, 2, 1], type=pa.int32()),
            "timestamp": pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
            "session_id": pa.array([1, 2, 1, 1, 2, 1], type=pa.int64()),
        }
    )


@pytest.fixture
def client() -> TestClient:
    model = PopularityRecommender()
    model.fit(_make_events())
    return TestClient(create_app(model))


def test_recommend_returns_200(client: TestClient) -> None:
    resp = client.post("/recommend", json={"seed_tracks": [1, 2], "n": 5})
    assert resp.status_code == 200


def test_recommend_response_has_recommendations_key(client: TestClient) -> None:
    resp = client.post("/recommend", json={"seed_tracks": [1], "n": 5})
    assert "recommendations" in resp.json()


def test_recommend_respects_n(client: TestClient) -> None:
    resp = client.post("/recommend", json={"seed_tracks": [1], "n": 3})
    assert len(resp.json()["recommendations"]) <= 3


def test_health_returns_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
