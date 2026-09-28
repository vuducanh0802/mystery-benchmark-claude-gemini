# GPT-4o Vanilla vs Guarded Experiment

This branch runs GPT-4o as Detective under both policies on the same five-level benchmark manifest. It uses the OpenAI API directly and requires `OPENAI_API_KEY`.

## Fresh run

```bash
git clone --branch experiments/gpt4o-vanilla-guarded --single-branch \
  https://github.com/vuducanh0802/mystery-benchmark-claude-gemini.git
cd mystery-benchmark-claude-gemini
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
```

Set `OPENAI_API_KEY` in your shell, or put it in a local `.env` file:

```bash
cp .env.example .env
nano .env
```

Check the 2,000-job matrix without calling the API, then launch:

```bash
VALIDATE_ONLY=1 bash scripts/run_gpt4o_baselines.sh
bash scripts/run_gpt4o_baselines.sh
```

For a paired 20-case-per-level pilot that remains resumable into the full run, set `TARGET_PER_LEVEL=20` (not `PER_LEVEL=20`):

```bash
TARGET_PER_LEVEL=20 bash scripts/run_gpt4o_baselines.sh
```

The launcher generates 200 cases per level if the manifest is missing. Re-running the same command skips complete, API-backed trajectories and retries incomplete episodes. Results are in `results/gpt4o_vanilla_guarded_200x5/summary.csv`, `validation.json`, and `trajectories/`.

## Resume the September run on this server

The earlier run has 15 valid trajectories in the Claude/Gemini worktree. To reuse those exact files and its manifest, run from this branch with the original paths and experiment ID:

```bash
BENCHMARK_DIR=/home/vda/mystery-benchmark-claude-gemini/data/benchmark_v1 \
OUTPUT_DIR=/home/vda/mystery-benchmark-claude-gemini/results/gpt4o_vanilla_guarded_200x5 \
EXPERIMENT_ID=claude_gemini_vanilla_guarded_v1 \
TARGET_PER_LEVEL=20 \
bash scripts/run_gpt4o_baselines.sh
```

An invalid model action is recorded as an episode error and retried; it no longer blocks the entire provider queue. API errors remain errors, with no heuristic fallback. Generated data, results, `.env`, and logs are excluded from git.

Check completed, API-backed cases by policy and level while it runs:

```bash
python scripts/track_gpt4o_progress.py /home/vda/mystery-benchmark-claude-gemini/results/gpt4o_vanilla_guarded_200x5
```
