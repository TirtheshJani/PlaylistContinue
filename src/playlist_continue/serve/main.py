"""Uvicorn entrypoint. Loads a pickled model from MODEL_PATH or falls back to popularity stub."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

from playlist_continue.models.popularity import PopularityRecommender
from playlist_continue.serve.api import create_app

_model_path = os.environ.get("MODEL_PATH", "")
if _model_path and Path(_model_path).exists():
    with open(_model_path, "rb") as _f:
        _model = pickle.load(_f)
else:
    _model = PopularityRecommender()

app = create_app(_model)
