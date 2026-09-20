#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Create the client environment first: python3 -m venv .venv && .venv/bin/pip install -r requirements-experiment.txt" >&2
  exit 2
fi
BENCHMARK_DIR="${BENCHMARK_DIR:-$ROOT/data/benchmark_v1}"
if [[ "${PREPARE_BENCHMARK:-1}" == "1" ]]; then
  "$PYTHON_BIN" scripts/prepare_paired_benchmark.py --output-dir "$BENCHMARK_DIR"
fi
read -r -a models <<< "${MODELS:-gpt-oss kimi}"
read -r -a policies <<< "${POLICIES:-vanilla guarded}"
read -r -a levels <<< "${LEVELS:-TRIVIAL EASY MEDIUM HARD EXPERT}"
cmd=("$PYTHON_BIN" scripts/run_gpt_oss_kimi_baselines.py
  --benchmark-dir "$BENCHMARK_DIR"
  --output-dir "${OUTPUT_DIR:-$ROOT/results/gpt_oss_kimi_vanilla_guarded}"
  --models "${models[@]}" --policies "${policies[@]}" --levels "${levels[@]}"
  --gpt-oss-workers "${GPT_OSS_WORKERS:-1}" --kimi-workers "${KIMI_WORKERS:-1}"
  --max-tokens "${MAX_TOKENS:-16384}" --history-window "${HISTORY_WINDOW:-10}"
  --timeout-seconds "${TIMEOUT_SECONDS:-300}" --max-retries "${MAX_RETRIES:-2}"
  --retry-rounds "${RETRY_ROUNDS:-0}")
[[ -z "${PER_LEVEL:-}" ]] || cmd+=(--per-level "$PER_LEVEL")
[[ "${VALIDATE_ONLY:-0}" != "1" ]] || cmd+=(--validate-only)
[[ "${PREFLIGHT_ONLY:-0}" != "1" ]] || cmd+=(--preflight-only)
[[ "${REPORT_ONLY:-0}" != "1" ]] || cmd+=(--report-only)
exec "${cmd[@]}" "$@"
