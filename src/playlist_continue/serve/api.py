"""FastAPI serving endpoint for playlist continuation."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from playlist_continue.models.base import Recommender


class RecommendRequest(BaseModel):
    seed_tracks: list[int]
    n: int = 500


class RecommendResponse(BaseModel):
    recommendations: list[int]


def create_app(model: Recommender) -> FastAPI:
    app = FastAPI(title="PlaylistContinue")

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest) -> RecommendResponse:
        return RecommendResponse(recommendations=model.recommend(req.seed_tracks, n=req.n))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
