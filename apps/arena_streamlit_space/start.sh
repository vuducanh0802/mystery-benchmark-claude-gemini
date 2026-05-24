#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${ARENA_ROOT:-/data/arena/results}"

uvicorn api_app:app \
  --host "${ARENA_API_HOST:-127.0.0.1}" \
  --port "${ARENA_API_PORT:-8000}" &
api_pid=$!

streamlit run app.py \
  --server.address "${STREAMLIT_SERVER_ADDRESS:-127.0.0.1}" \
  --server.port "${STREAMLIT_SERVER_PORT:-8501}" &
streamlit_pid=$!

nginx -g "daemon off;" &
nginx_pid=$!

trap 'kill "$api_pid" "$streamlit_pid" "$nginx_pid" 2>/dev/null || true' INT TERM EXIT

wait -n "$api_pid" "$streamlit_pid" "$nginx_pid"
