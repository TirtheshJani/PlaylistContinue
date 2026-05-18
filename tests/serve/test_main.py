import importlib


def test_app_is_fastapi_instance():
    """main.py must export a FastAPI app object."""
    import fastapi

    # Reimport to pick up env changes.
    import playlist_continue.serve.main as main_mod

    importlib.reload(main_mod)
    assert isinstance(main_mod.app, fastapi.FastAPI)


def test_health_endpoint_returns_ok():
    """The app loaded by main.py must respond to GET /health."""
    from fastapi.testclient import TestClient

    import playlist_continue.serve.main as main_mod

    importlib.reload(main_mod)
    client = TestClient(main_mod.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_recommend_endpoint_exists():
    """POST /recommend must be present regardless of model."""
    from fastapi.testclient import TestClient

    import playlist_continue.serve.main as main_mod

    importlib.reload(main_mod)
    client = TestClient(main_mod.app)
    resp = client.post("/recommend", json={"seed_tracks": [], "n": 5})
    assert resp.status_code == 200
