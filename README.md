# MysteryArena — Procedural Murder-Mystery Benchmark

A fully-automated benchmark for evaluating LLM-based detective agents on procedurally generated murder-mystery scenarios. Agents must determine **who** committed the murder, **what** weapon was used, and **where** it occurred — under partial observability, within an action budget, against NPCs that may lie.

> **Branch:** `master` — **textual** observation modality. Agents see only natural-language descriptions of rooms, objects, and NPC dialogue.
>
> Other modalities live on dedicated branches:
> - `thong/graphics_2d` — adds 2D top-down rendering, a web client, and VLM agents (image + text observations).
> - `thong/graphics_3d` — adds a Godot 4 first-person 3D client, a WebSocket server, and VLM agents.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          MysteryArena                                │
│                                                                      │
│  ┌────────────────┐   ┌──────────────────────┐   ┌───────────────┐  │
│  │   benchmark/   │──▶│   mystery_world/     │──▶│ evaluation/   │  │
│  │  generate.py   │   │                      │   │  runner.py    │  │
│  │  verify.py     │   │  ┌────────────────┐  │   │  metrics.py   │  │
│  └────────────────┘   │  │   WorldState   │  │   │  trajectory  │  │
│          │            │  │  locations     │  │   └───────┬───────┘  │
│  ┌───────▼──────────┐ │  │  characters    │  │           │          │
│  │  ComplexityConfig│ │  │  objects       │  │   ┌───────▼───────┐  │
│  │  5 presets:      │ │  │  evidence      │  │   │   agents/     │  │
│  │  TRIVIAL → EXPERT│ │  └────────────────┘  │   │  LLMAgent     │  │
│  └──────────────────┘ │                      │   │  HeuristicAgt │  │
│                       │  ┌────────────────┐  │   │  SymbolicAgt  │  │
│                       │  │  Events Engine │  │   │  OracleAgent  │  │
│                       │  │  events.py     │  │   └───────────────┘  │
│                       │  │  narrator.py   │  │                      │
│                       │  │  npc_responder │  │                      │
│                       │  └────────────────┘  │                      │
│                       └──────────────────────┘                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Episode Loop

```
generate_mystery(seed, config) ──► WorldState
                          │
    ┌─────────────────────▼──────────────────────────────────────────┐
    │  for step in range(num_time_steps):                            │
    │    process_weather_change(state, rng)       ◄── events.py      │
    │    process_npc_movement(state, rng)         ◄── Option A or B  │
    │    process_culprit_tampering(state, rng)    ◄── hidden events  │
    │    process_evidence_decay(state, rng)                          │
    │                                                                │
    │    obs = narrator.render(state, events)     ◄── partial obs.   │
    │    action, kwargs = agent.decide_action(obs)                   │
    │    result = env.step(action, **kwargs)                         │
    │         ├── MOVE / EXAMINE_LOCATION / EXAMINE_OBJECT           │
    │         ├── TALK_TO ──► NPCResponder (LLM or fallback)         │
    │         ├── ACCUSE ──► score & end episode                     │
    │         └── WAIT / TAKE_OBJECT / CHECK_INVENTORY / ...         │
    └────────────────────────────────────────────────────────────────┘
                          │
                     EpisodeMetrics + JSONL trajectory
```

---

## Installation

```bash
git clone https://github.com/nguyentthong/mystery-benchmark.git
cd mystery-benchmark
uv sync
```

[uv](https://docs.astral.sh/uv/) is the recommended package manager. If you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### API Keys

Export whichever keys you need (only the providers you'll actually call):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # Claude detective + Claude NPCs
export OPENAI_API_KEY="sk-..."          # GPT detective + OpenAI-direct NPCs
export GOOGLE_API_KEY="..."             # Gemini detective
export OPENROUTER_API_KEY="sk-or-..."   # OpenRouter detective + OpenRouter NPCs
```

---

## Quick Start

### Play it yourself (human mode)

```bash
# Random MEDIUM case, NPCs powered by GPT-4o-mini
uv run scripts/play.py --npc-backend openai --npc-model gpt-4o-mini

# Pre-generated curated example
uv run scripts/play.py --example trivial_seed_0 --npc-backend openai

# Pick difficulty + seed
uv run scripts/play.py --level HARD --seed 42 --npc-backend openai

# Deterministic NPCs (no API calls)
uv run scripts/play.py --level EASY --seed 0 --npc-backend fallback
```

In-game commands: `look`, `go <room>`, `examine <object>`, `search`, `talk <name>`, `take <object>`, `inventory`, `map`, `suspects`, `accuse`, `wait`, `hint`, `help`, `quit`. The `hint` command runs the oracle on the current state and prints the recommended next action.

### Pre-generated examples

Twenty curated cases (4 per difficulty level) ship in `examples/`. Each records the ground-truth answer and the oracle's full action sequence — ideal for benchmarking and human comparison.

```bash
python scripts/list_examples.py                       # list all
python scripts/list_examples.py --level EASY          # filter by level
python scripts/list_examples.py --show easy_seed_0    # full details + oracle walkthrough
```

---

## Generate a benchmark suite

```bash
uv run scripts/generate_benchmark.py \
    --levels TRIVIAL EASY MEDIUM HARD EXPERT \
    --instances-per-level 20 \
    --seed 42 \
    --output-dir data/benchmark_v1
```

Layout:
```
data/benchmark_v1/
  level_1/  instance_*.json + solution_*.json
  level_2/  ...
  manifest.json          ← index of all instances
```

---

## Evaluation

There are two entry points. **`sweep_eval.py` is the recommended one** — it parallelises across seeds, resumes on crash, and writes a self-describing JSONL trajectory per episode.

### NPC backends

NPCs are stateful and lying-aware (the system prompt injects deception directives from the ground truth). You pick a backend per run:

| `--npc-provider` | Endpoint | Auth |
|-----|-----|-----|
| `fallback` | deterministic templates, no LLM | none |
| `openai` | `api.openai.com` | `OPENAI_API_KEY` |
| `openrouter` | `openrouter.ai` | `OPENROUTER_API_KEY` |
| `vllm` | self-hosted (`--npc-url ...`) | `EMPTY` |

The vLLM-specific `chat_template_kwargs={"enable_thinking": false}` is sent **only** to vLLM endpoints; OpenAI/OpenRouter would reject it.

### Concurrent sweep (recommended)

```bash
# Heuristic baseline — no API key needed
uv run scripts/sweep_eval.py \
    --agent heuristic \
    --levels TRIVIAL EASY MEDIUM HARD EXPERT --seeds 0-19 \
    --trajectory-dir results/trajectories \
    --workers 8

# Claude detective + OpenAI NPCs
uv run scripts/sweep_eval.py \
    --agent claude --model claude-sonnet-4-6 \
    --levels TRIVIAL EASY MEDIUM --seeds 0-19 \
    --npc-provider openai --npc-model gpt-4o-mini \
    --trajectory-dir results/trajectories --workers 4

# GPT-4o-mini detective + OpenAI NPCs
uv run scripts/sweep_eval.py \
    --agent chatgpt-mini \
    --levels TRIVIAL EASY --seeds 0-9 \
    --npc-provider openai --npc-model gpt-4o-mini \
    --trajectory-dir results/trajectories --workers 4

# Gemini detective + OpenRouter NPCs
uv run scripts/sweep_eval.py \
    --agent gemini --model gemini-2.0-flash \
    --levels TRIVIAL EASY MEDIUM --seeds 0-19 \
    --npc-provider openrouter --npc-model qwen/qwen-2.5-72b-instruct \
    --trajectory-dir results/trajectories --workers 4

# Self-hosted vLLM NPC server
uv run scripts/sweep_eval.py \
    --agent claude --model claude-sonnet-4-6 \
    --levels TRIVIAL EASY MEDIUM --seeds 0-19 \
    --npc-provider vllm --npc-url http://localhost:8200/v1 --npc-model Qwen/Qwen3.5-27B \
    --trajectory-dir results/trajectories --workers 4
```

Built-in agent slots (`scripts/sweep_eval.py:AGENT_CONFIGS`):

| Slot | Provider | Default model |
|---|---|---|
| `heuristic` | — | rule-based, no LLM |
| `oracle_min` | — | minimum-action oracle (calibration upper bound) |
| `oracle_max` | — | maximum-score oracle |
| `claude` | anthropic | `claude-sonnet-4-6` |
| `claude-opus` | anthropic | `claude-opus-4-7` |
| `chatgpt` | openai | `gpt-4o` |
| `chatgpt-mini` | openai | `gpt-4o-mini` |
| `gemini` | google | `gemini-2.0-flash` |
| `openrouter` | openrouter | `qwen/qwen3.5-27b` |

Output layout (one JSONL per episode):
```
results/trajectories/{agent}/{LEVEL}/seed_{n}.jsonl
```

Each file is `header → step₁ → step₂ → … → footer`. The footer carries `episode_summary`, `metrics` (the full `EpisodeMetrics` dict), `elapsed_seconds`, and any error string. Re-running the same command is a no-op for seeds whose JSONL already exists, so it's safe to resume after a crash. To re-run a failed seed, **delete its JSONL first** — the sweep checks for file existence, not success.

### Aggregate results into a table

```bash
uv run scripts/build_results_table.py \
    --trajectory-dir results/trajectories \
    --output results/
```

Produces `results/summary.csv` and `results/summary.md` with per-(agent, level) means: `n, solve_rate, composite, accusation, triangle, alibi, elimination, avg_actions, avg_tokens`.

For per-edge / per-seed breakdowns and plots, use:

```bash
uv run scripts/analyze_results.py --trajectory-dir results/trajectories
```

### Single-agent legacy runner

`scripts/run_evaluation.py` is the older single-process entry point. It writes per-episode JSON plus a `summary.json`. Prefer `sweep_eval.py` unless you specifically need its CLI flags.

```bash
uv run scripts/run_evaluation.py \
    --agent claude --model claude-sonnet-4-6 \
    --benchmark-dir data/benchmark_v1 \
    --output-dir results/claude_sonnet \
    --npc-url http://localhost:8200/v1 --npc-model Qwen/Qwen3.5-27B
```

### Replay & reproducibility

Every JSONL trajectory contains the world seed, NPC seed, and per-step events, so a run can be replayed bit-for-bit:

```bash
uv run scripts/replay.py results/trajectories/claude/MEDIUM/seed_3.jsonl
uv run scripts/verify_reproducibility.py results/trajectories/claude/MEDIUM/seed_3.jsonl
```

### Generalisation eval

`scripts/eval_generalisation.py` runs an agent on held-out seeds to measure novel-world generalisation (RQ3). See its `--help` for flags.

### Oracle calibration

The oracle reads ground truth and executes the cheapest legal proof — one clue per Locard triangle edge plus an alibi contradiction — through the normal game API. It cannot skip discovery.

```python
from agents.minimum_action_oracle_agent import OracleAgent
from mystery_world import COMPLEXITY_PRESETS, ComplexityLevel
from mystery_world.generator import generate_mystery
from mystery_world.narrator import render_initial_briefing
from mystery_world.world import MysteryEnvironment

config = COMPLEXITY_PRESETS[ComplexityLevel.MEDIUM]
state  = generate_mystery(seed=42, config=config)
env    = MysteryEnvironment(state)
result = OracleAgent().run(env, render_initial_briefing(env))
print(result["accusation_correct"], result["actions_taken"])
```

---

## Metrics

Every episode is scored on multiple dimensions; the headline number is **`composite_score`**.

### Per-episode

| Metric | Range | What it measures |
|---|---|---|
| `solved` | 0/1 | All three of (suspect, weapon, location) correct |
| `accusation_score` | 0–1 | Fraction of (suspect, weapon, location) correct |
| `triangle_score` | 0–3 | Sum of F1 over three Locard edges (SUSPECT_WEAPON, WEAPON_VICTIM, SUSPECT_ROOM) against the valid-evidence-ID set per edge |
| `alibi_score` | 0–1 | Cited culprit's alibi + valid contradiction |
| `elimination_score` | 0–1 | `max(0, (correct − 2·incorrect) / total_innocents)` from SUSPECT_ELSEWHERE evidence |
| `examine_efficiency` | 0–1 | `examine_hit / examine_total` — focused-investigation reward |
| `clue_efficiency` | 0–1 | discovered relevant evidence / total relevant |
| `final_belief_accuracy` | 0–1 | Probability mass on the true culprit at the final step |
| `action_efficiency` | 0–1 | `actions_used / budget` |
| `total_tokens` | — | LLM tokens consumed |

### Composite (headline)

```
base      = 0.35·accusation + 0.35·(triangle/3) + 0.15·alibi + 0.15·elimination
composite = base × (0.8 + 0.2·examine_efficiency)
```

### Aggregated (per agent × level)

`build_results_table.py` emits: `n, solve_rate, mean_composite, mean_accusation, mean_triangle, mean_alibi, mean_elimination, avg_actions, avg_tokens`. Standard error of `solve_rate` is included for confidence-interval reporting.

---

## ComplexityConfig Presets

| Level | Locations | Suspects | Budget | Freshness | Route Constraints |
|-------|-----------|----------|--------|-----------|-------------------|
| TRIVIAL | 3 | 2 | 40 | 3.0 | 0 |
| EASY | 4 | 3 | 50 | 2.5 | 0 |
| MEDIUM | 5 | 4 | 75 | 2.0 | 1 |
| HARD | 7 | 5 | 100 | 1.5 | 2 |
| EXPERT | 10 | 7 | 150 | 1.0 | 3 |

Full knob list: see `mystery_world/__init__.py:ComplexityConfig`. Notable axes include `evidence_decay_rate`, `culprit_tamper_prob`, `testimony_unreliability`, `evidence_ambiguity`, and `reactive_events` (the Option A vs B NPC movement toggle).

---

## NPC Lying System

Lying is deterministically controlled from ground-truth flags — the LLM has no autonomy over deception:

| Condition | Instruction injected into NPC system prompt |
|-----------|---------------------------------------------|
| `char.is_culprit == True` | Deny any involvement; claim your alibi |
| `alibi_corroboration_is_genuine == False` | Confirm the culprit was with you (false alibi) |
| `char.alibi_has_gap == True` | (no special instruction — honest but incomplete) |
| Otherwise | (no instruction — fully truthful NPC) |

The LLM generates fluent dialogue within these constraints. The agent must infer deception from logical inconsistencies; it cannot read these flags.

### NPC Movement (Option A vs B)

Controlled by `reactive_events` in `ComplexityConfig`:

- **Option A** (TRIVIAL/EASY/MEDIUM, `reactive_events=False`): each NPC independently relocates to a random adjacent room with probability `npc_move_prob` per step.
- **Option B** (HARD/EXPERT, `reactive_events=True`): NPCs follow believable routines (home → errand/social → home) with social visits to liked characters. The culprit follows the same visible routine and only deviates for probability-gated, agent-invisible tamper runs.

---

## Adding a New Agent

1. Create `agents/my_agent.py` subclassing `BaseAgent`:

```python
from agents.base_agent import BaseAgent
from mystery_world.world import AgentAction, MysteryEnvironment

class MyAgent(BaseAgent):
    def initialize(self, env: MysteryEnvironment, briefing: str) -> None: ...
    def decide_action(self, observation: str) -> tuple[AgentAction, dict]: ...
    def update_beliefs(self, observation: str) -> None: ...
```

2. Register the slot in `scripts/sweep_eval.py:AGENT_CONFIGS` and route construction in `_make_agent`.

3. Sweep:

```bash
uv run scripts/sweep_eval.py --agent my_agent \
    --levels TRIVIAL --seeds 0-9 \
    --trajectory-dir results/trajectories
```

---

## Module Reference

| Path | Purpose |
|------|---------|
| `mystery_world/world.py` | `WorldState`, `MysteryEnvironment` (action handlers, Locard scorer), `AgentAction` enum |
| `mystery_world/generator.py` | `generate_mystery(seed, config)` — procedural builder |
| `mystery_world/events.py` | Tick functions: weather, NPC movement (A/B), culprit tampering, evidence decay |
| `mystery_world/narrator.py` | Partial-observation natural-language rendering |
| `mystery_world/npc_responder.py` | OpenAI-compatible client; supports vLLM, OpenAI direct, OpenRouter |
| `benchmark/generate.py` | `generate_benchmark_suite`, `load_benchmark_suite` |
| `benchmark/verify.py` | Structural / Locard solvability checks |
| `agents/llm_agent.py` | `LLMAgent` — supports `anthropic`, `openai`, `google`, `openrouter` |
| `agents/heuristic_agent.py` | Rule-based baseline |
| `agents/symbolic_agent.py` | Logical-fact-base agent |
| `agents/minimum_action_oracle_agent.py` | Calibration upper bound (cheapest legal proof) |
| `agents/maximum_score_oracle_agent.py` | Score-maximising oracle (cites every valid edge ID) |
| `evaluation/runner.py` | `run_episode`, `run_benchmark` |
| `evaluation/metrics.py` | `EpisodeMetrics`, `compute_episode_metrics`, `aggregate_metrics` |
| `evaluation/trajectory.py` | `TrajectoryWriter` — JSONL header/step/footer schema |
| `scripts/sweep_eval.py` | Concurrent multi-seed driver with resume |
| `scripts/build_results_table.py` | Aggregate trajectories → CSV / Markdown table |
| `scripts/replay.py`, `scripts/verify_reproducibility.py` | Replay JSONL trajectories deterministically |

---

## Reproducibility

Every benchmark instance is fully determined by:
- A **world seed** (controls all procedural generation).
- A **`ComplexityConfig`** (specifies every difficulty knob).
- A separate **`--npc-seed`** passed to `NPCResponder` for NPC-LLM determinism.
- **Deterministic transition functions** (world state evolves identically given the same seed).
- **JSONL trajectory logs** (full audit trail; replay verified by `verify_reproducibility.py`).

---

## Research Questions

1. **RQ1** — To what extent can current LLM agents gather clues, maintain belief states, and perform sound abductive/deductive inference over long horizons in narrative environments?
2. **RQ2** — Does augmenting LLM agents with explicit symbolic state tracking improve solve rates and reasoning faithfulness vs pure prompting?
3. **RQ3** — How well do agents generalise to novel worlds with different rules, entity distributions, and solution structures?

---

## Citation

```bibtex
@inproceedings{mysteryarena2026,
  title     = {MysteryArena: A Procedural Benchmark for Evaluating
               LLM Agents on Abductive Reasoning in Dynamic Narrative Environments},
  author    = {...},
  booktitle = {...},
  year      = {2026}
}
```
