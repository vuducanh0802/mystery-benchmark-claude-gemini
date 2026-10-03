#!/usr/bin/env bash
set -euo pipefail
ABLATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ABLATION_ROOT"
ENV_FILE="${ENV_FILE:-$ABLATION_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
PYTHON_BIN="${PYTHON_BIN:-$ABLATION_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Create .venv and install requirements-experiment.txt first, or set PYTHON_BIN." >&2
  exit 2
fi
BENCHMARK_DIR="${BENCHMARK_DIR:-$ABLATION_ROOT/data/benchmark_v1}"
if [[ "${PREPARE_BENCHMARK:-1}" == "1" ]]; then
  "$PYTHON_BIN" scripts/prepare_paired_benchmark.py --output-dir "$BENCHMARK_DIR"
fi
read -r -a variants <<< "${VARIANTS:-full_minus_accuse delay_control prompt_ledger}"
read -r -a levels <<< "${LEVELS:-TRIVIAL EASY MEDIUM HARD EXPERT}"
cmd=("$PYTHON_BIN" scripts/run_gpt_oss_ablations.py
  --benchmark-dir "$BENCHMARK_DIR"
  --output-dir "${OUTPUT_DIR:-$ABLATION_ROOT/results/gpt_oss_guarded_ablations}"
  --experiment-id "${EXPERIMENT_ID:-gpt_oss_guarded_ablations_v1}"
  --variants "${variants[@]}" --levels "${levels[@]}"
  --per-level "${PER_LEVEL:-200}" --gpt-oss-workers "${GPT_OSS_WORKERS:-1}"
  --max-tokens "${MAX_TOKENS:-8192}" --history-window "${HISTORY_WINDOW:-10}"
  --max-accuse-blocks "${MAX_ACCUSE_BLOCKS:-2}"
  --timeout-seconds "${TIMEOUT_SECONDS:-180}" --max-retries "${MAX_RETRIES:-2}"
  --retry-backoff-seconds "${RETRY_BACKOFF_SECONDS:-1}"
  --retry-rounds "${RETRY_ROUNDS:-2}")
[[ -z "${REASONING_EFFORT:-}" ]] || cmd+=(--reasoning-effort "$REASONING_EFFORT")
[[ -z "${TEMPERATURE:-}" ]] || cmd+=(--temperature "$TEMPERATURE")
[[ "${VALIDATE_ONLY:-0}" != "1" ]] || cmd+=(--validate-only)
[[ "${PREFLIGHT_ONLY:-0}" != "1" ]] || cmd+=(--preflight-only)
[[ "${REPORT_ONLY:-0}" != "1" ]] || cmd+=(--report-only)
exec "${cmd[@]}" "$@"
