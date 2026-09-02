#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${ENV_FILE:-$ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if command -v uv >/dev/null 2>&1; then
  python_cmd=(uv run python)
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  if ! "$PYTHON_BIN" -c 'import anthropic, openai, structlog' >/dev/null 2>&1; then
    echo "uv is unavailable and $PYTHON_BIN lacks project dependencies." >&2
    echo "Install with uv sync, or set PYTHON_BIN to the prepared environment." >&2
    exit 2
  fi
  python_cmd=("$PYTHON_BIN")
fi
MODELS="${MODELS:-claude gemini}"
if [[ " $MODELS " == *" claude "* && -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ANTHROPIC_API_KEY is missing. Paste it into $ENV_FILE." >&2
  exit 2
fi
if [[ " $MODELS " == *" gemini "* && -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "GEMINI_API_KEY is missing. Paste it into $ENV_FILE." >&2
  exit 2
fi
if [[ " $MODELS " == *" gpt4o "* && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is missing. Paste it into $ENV_FILE." >&2
  exit 2
fi

BENCHMARK_DIR="${BENCHMARK_DIR:-$ROOT/data/benchmark_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/claude_gemini_vanilla_guarded}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-sonnet-4-6}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.6-flash}"
GPT4O_MODEL="${GPT4O_MODEL:-gpt-4o}"
CLAUDE_WORKERS="${CLAUDE_WORKERS:-4}"
GEMINI_WORKERS="${GEMINI_WORKERS:-8}"
GPT4O_WORKERS="${GPT4O_WORKERS:-8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
HISTORY_WINDOW="${HISTORY_WINDOW:-10}"
RETRY_ROUNDS="${RETRY_ROUNDS:-1}"
LEVELS="${LEVELS:-TRIVIAL EASY MEDIUM HARD EXPERT}"
POLICIES="${POLICIES:-vanilla guarded}"

if [[ ! -s "$BENCHMARK_DIR/manifest.json" ]]; then
  if [[ "${GENERATE_BENCHMARK:-1}" != "1" ]]; then
    echo "Missing $BENCHMARK_DIR/manifest.json" >&2
    exit 2
  fi
  echo "Benchmark manifest is absent; generating 200 deterministic cases per level."
  "${python_cmd[@]}" scripts/generate_benchmark.py \
    --levels TRIVIAL EASY MEDIUM HARD EXPERT \
    --instances-per-level 200 \
    --seed 42 \
    --output-dir "$BENCHMARK_DIR"
fi

read -r -a level_args <<< "$LEVELS"
read -r -a model_args <<< "$MODELS"
read -r -a policy_args <<< "$POLICIES"

cmd=(
  "${python_cmd[@]}" scripts/run_claude_gemini_baselines.py
  --benchmark-dir "$BENCHMARK_DIR"
  --output-dir "$OUTPUT_DIR"
  --claude-model "$CLAUDE_MODEL"
  --gemini-model "$GEMINI_MODEL"
  --gpt4o-model "$GPT4O_MODEL"
  --claude-workers "$CLAUDE_WORKERS"
  --gemini-workers "$GEMINI_WORKERS"
  --gpt4o-workers "$GPT4O_WORKERS"
  --max-tokens "$MAX_TOKENS"
  --history-window "$HISTORY_WINDOW"
  --retry-rounds "$RETRY_ROUNDS"
  --models "${model_args[@]}"
  --policies "${policy_args[@]}"
  --levels "${level_args[@]}"
)

if [[ -n "${PER_LEVEL:-}" ]]; then
  cmd+=(--per-level "$PER_LEVEL")
fi
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  cmd+=(--validate-only)
fi

echo "Launching paired API-model baselines. API key values will not be printed."
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
