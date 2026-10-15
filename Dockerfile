FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FTS_DEVICE=cpu \
    FTS_MLFLOW_TRACKING_URI=sqlite:///mlflow.db

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts

RUN pip install --upgrade pip && pip install -e ".[dev]"

# Checkpoint is mounted at runtime (see docker-compose). Train first:
#   python -m fts_forecaster.train --smoke
# then `docker compose up --build api` with FTS_API_KEY set.

EXPOSE 8000
CMD ["uvicorn", "fts_forecaster.api:app", "--host", "0.0.0.0", "--port", "8000"]
