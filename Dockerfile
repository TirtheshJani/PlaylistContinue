# Build stage
FROM python:3.11-slim AS build

WORKDIR /app

# Install uv for fast dependency resolution.
RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY src/ ./src/

# Install only production deps (no dev extras).
RUN uv pip install --system --no-cache-dir -e .

# -----------------------------------------------------------
# Runtime stage - same base, copy installed packages
# -----------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY src/ ./src/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "playlist_continue.serve.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
