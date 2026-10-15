#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec uvicorn fts_forecaster.api:app --host "${FTS_HOST:-0.0.0.0}" --port "${FTS_PORT:-8000}"
