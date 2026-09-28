#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODELS=gpt4o
export POLICIES="${POLICIES:-vanilla guarded}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/gpt4o_vanilla_guarded_200x5}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-gpt4o_vanilla_guarded_v1}"

exec bash "$ROOT/scripts/run_claude_gemini_baselines.sh"
