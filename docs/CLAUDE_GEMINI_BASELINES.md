# Claude/Gemini Vanilla vs Guarded Baselines

This experiment runs four paired cells on the same serialized benchmark cases:

| Model | Policy |
|---|---|
| Claude | Vanilla |
| Claude | Guarded |
| Gemini | Vanilla |
| Gemini | Guarded |

The culprit remains passive and interviews use the shared deterministic fallback.
The only treatment difference within each model is the detective policy.

## Setup

```bash
uv sync
cp .env.example .env
```

Paste the two keys into `.env`:

```dotenv
ANTHROPIC_API_KEY=paste_claude_key_here
GEMINI_API_KEY=paste_gemini_key_here
```

The launcher loads `.env` automatically. `GOOGLE_API_KEY` is also accepted as
the Gemini variable. Keys are never written to trajectories, run configuration,
logs, or the repository.

Validate the manifest, matrix, and presence of both keys without making API calls:

```bash
VALIDATE_ONLY=1 bash scripts/run_claude_gemini_baselines.sh
```

Launch the complete experiment:

```bash
bash scripts/run_claude_gemini_baselines.sh
```

For a long run in tmux:

```bash
tmux new-session -d -s mystery-api \
  'cd /path/to/mystery-benchmark && bash scripts/run_claude_gemini_baselines.sh 2>&1 | tee results/claude_gemini_run.log'
tmux attach -t mystery-api
```

Defaults are pinned to `claude-sonnet-4-6` and `gemini-3.6-flash`. Override
them explicitly when needed:

```bash
CLAUDE_MODEL=claude-sonnet-5 \
GEMINI_MODEL=gemini-3.7-flash \
bash scripts/run_claude_gemini_baselines.sh
```

Useful controls:

```bash
# Lower concurrency if the lab account has tight rate limits.
CLAUDE_WORKERS=2 GEMINI_WORKERS=4 bash scripts/run_claude_gemini_baselines.sh

# Run selected levels or a bounded paper pilot.
LEVELS="HARD EXPERT" PER_LEVEL=20 bash scripts/run_claude_gemini_baselines.sh

# Run one model or one policy without changing output identities.
MODELS="gemini" POLICIES="guarded" bash scripts/run_claude_gemini_baselines.sh
```

If `data/benchmark_v1/manifest.json` is absent, the shell entry point generates
200 deterministic instances per level using seed 42. Set
`GENERATE_BENCHMARK=0` to require an externally supplied canonical manifest.
For comparison with an existing experiment, point `BENCHMARK_DIR` at that exact
suite rather than regenerating it.

## Output And Resume

Outputs are written under:

```text
results/claude_gemini_vanilla_guarded/
  run_config.json
  validation.json
  summary.csv
  summary.json
  trajectories/{model_identity}/{policy}/{level}/*.jsonl
```

`results/` and API environment files are gitignored. Do not force-add them.

Resume with the same command. A trajectory is skipped only when it has the
expected model, provider, policy, source hash, successful footer, metrics, and
non-zero API token usage. Missing, partial, zero-token, and error trajectories
are rerun. Authentication or model-access errors stop new work for that provider
so a bad configuration does not fan out across the full matrix.

Guarded trajectories retain both `proposed_action` and the executed action plus
`guard_intervention`, allowing intervention-rate analysis without hidden-state
access. API/model failures are terminal episode errors and are excluded from
performance denominators; no heuristic action is silently substituted.
