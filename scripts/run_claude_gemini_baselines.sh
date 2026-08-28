#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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
  echo "ANTHROPIC_API_KEY is unset in this shell." >&2
  exit 2
fi
if [[ " $MODELS " == *" gemini "* && -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "GEMINI_API_KEY or GOOGLE_API_KEY is unset in this shell." >&2
  exit 2
fi

BENCHMARK_DIR="${BENCHMARK_DIR:-$ROOT/data/benchmark_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/claude_gemini_vanilla_guarded}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-sonnet-4-6}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.6-flash}"
CLAUDE_WORKERS="${CLAUDE_WORKERS:-4}"
GEMINI_WORKERS="${GEMINI_WORKERS:-8}"
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
  --claude-workers "$CLAUDE_WORKERS"
  --gemini-workers "$GEMINI_WORKERS"
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

echo "Launching Claude/Gemini paired baselines. API key values will not be printed."
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
