# Claude/Gemini Vanilla vs Guarded Experiment

## Run From A Fresh Server

```bash
git clone --branch experiments/claude-gemini-vanilla-guarded --single-branch \
  https://github.com/vuducanh0802/mystery-benchmark-claude-gemini.git
cd mystery-benchmark-claude-gemini
```

Install dependencies:

```bash
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
```

Create the local key file:

```bash
cp .env.example .env
nano .env
```

Paste both keys after `=` and save:

```dotenv
ANTHROPIC_API_KEY=paste_claude_key_here
GEMINI_API_KEY=paste_gemini_key_here
```

Validate setup without making API calls:

```bash
VALIDATE_ONLY=1 bash scripts/run_claude_gemini_baselines.sh
```

Run the complete experiment:

```bash
bash scripts/run_claude_gemini_baselines.sh
```

The same command resumes an interrupted run. Results are written to:

```text
results/claude_gemini_vanilla_guarded/summary.csv
results/claude_gemini_vanilla_guarded/validation.json
results/claude_gemini_vanilla_guarded/trajectories/
```

Check completion and results:

```bash
cat results/claude_gemini_vanilla_guarded/validation.json
column -s, -t < results/claude_gemini_vanilla_guarded/summary.csv | less -S
```

`.env`, generated benchmark data, trajectories, and results are gitignored.
