# MysteryArena — Procedural Murder-Mystery Benchmark

A fully-automated benchmark for evaluating LLM-based detective agents on procedurally generated murder-mystery scenarios. Agents must determine **who** committed the murder, **what** weapon was used, and **where** it occurred — under partial observability, within an action budget, against NPCs that may lie.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          MysteryArena                                │
│                                                                      │
│  ┌────────────────┐   ┌──────────────────────┐   ┌───────────────┐  │
│  │   benchmark/   │──▶│   mystery_world/     │──▶│ evaluation/   │  │
│  │  Generator     │   │                      │   │  runner.py    │  │
│  │  Serialiser    │   │  ┌────────────────┐  │   │  metrics.py   │  │
│  │  Validator     │   │  │   WorldState   │  │   └───────┬───────┘  │
│  └────────────────┘   │  │  locations     │  │           │          │
│          │            │  │  characters    │  │   ┌───────▼───────┐  │
│  ┌───────▼──────────┐ │  │  objects       │  │   │   agents/     │  │
│  │  ComplexityConfig│ │  │  evidence      │  │   │  LLMAgent     │  │
│  │  5 presets:      │ │  └────────────────┘  │   │  HeuristicAgt │  │
│  │  TRIVIAL → EXPERT│ │                      │   └───────────────┘  │
│  └──────────────────┘ │  ┌────────────────┐  │                      │
│                       │  │  Events Engine │  │                      │
│                       │  │  events.py     │  │                      │
│                       │  │  narrator.py   │  │                      │
│                       │  │  npc_responder │  │                      │
│                       │  └────────────────┘  │                      │
│                       └──────────────────────┘                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Episode Loop

```
MysteryGenerator ──► WorldState
                          │
    ┌─────────────────────▼──────────────────────────────────────────┐
    │  for step in range(num_time_steps):                            │
    │    process_weather_change(state, rng)       ◄── events.py      │
    │    process_npc_movement(state, rng)         ◄── Option A or B  │
    │    process_culprit_tampering(state, rng)    ◄── hidden events  │
    │    process_evidence_decay(state, rng)                          │
    │                                                                │
    │    obs = narrator.render(state, events)     ◄── partial obs.   │
    │    action, args = agent.decide_action(obs)                     │
    │    result = env.step(action, args)                             │
    │         │                                                      │
    │         ├── MOVE / EXAMINE_LOCATION                            │
    │         ├── EXAMINE_OBJECT / SEARCH_FOR_EVIDENCE               │
    │         ├── TALK_TO ──► NPCResponder (LLM or template)         │
    │         ├── ACCUSE ──► score & end episode                     │
    │         └── WAIT / CHECK_INVENTORY / TAKE_OBJECT               │
    └────────────────────────────────────────────────────────────────┘
                          │
                     ScoreRecord
```

### NPC Interview Flow (stateful, lying-aware)

```
Agent              world.py              npc_responder.py       vLLM / Together AI
  │                    │                       │                       │
  │── TALK_TO ────────►│                       │                       │
  │  (char, question)  │                       │                       │
  │                    │── build_npc_system_prompt(char, state) ──────►
  │                    │   [inject lying directive from ground truth]  │
  │                    │                       │                       │
  │                    │── NPCResponder.respond(char, state, q, hist) ►│
  │                    │   [append q to per-char history]              │
  │                    │                       │──── API call ────────►│
  │                    │                       │◄─── NPC response ─────│
  │                    │◄── response text ─────│                       │
  │◄── observation ────│                       │                       │
```

---

## Installation

```bash
git clone https://github.com/nguyentthong/mystery-benchmark.git
cd mystery-benchmark
uv sync
```

[uv](https://docs.astral.sh/uv/) is the recommended package manager. It reads `pyproject.toml` and creates a virtual environment automatically. If you don't have uv installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

All dependencies (`anthropic`, `openai`, `structlog`, etc.) are declared in `pyproject.toml` and installed by `uv sync`. To add a new dependency:

```bash
uv add <package>
```

### API Keys

Export whichever keys you need:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # Claude agents
export OPENAI_API_KEY="sk-..."          # ChatGPT agents
export GOOGLE_API_KEY="..."             # Gemini agents
```

---

## Quick Start

### 1. Generate a benchmark suite

```bash
uv run scripts/generate_benchmark.py \
    --levels TRIVIAL EASY MEDIUM \
    --instances-per-level 5 \
    --seed 42 \
    --output-dir data/benchmark_v1
```

Valid level names: `TRIVIAL`, `EASY`, `MEDIUM`, `HARD`, `EXPERT`.

Output structure:
```
data/benchmark_v1/
  level_1/
    instance_10042.json
    solution_10042.json
    ...
  level_2/
    ...
  manifest.json          ← index of all instances with solutions
```

### 2. Run evaluation

```bash
# Heuristic baseline (no API key needed)
uv run scripts/run_evaluation.py \
    --agent heuristic \
    --benchmark-dir data/benchmark_v1 \
    --output-dir results/heuristic

# Claude Sonnet
uv run scripts/run_evaluation.py \
    --agent claude \
    --model claude-sonnet-4-20250514 \
    --benchmark-dir data/benchmark_v1 \
    --output-dir results/claude_sonnet

# ChatGPT
uv run scripts/run_evaluation.py \
    --agent chatgpt \
    --model gpt-4o \
    --benchmark-dir data/benchmark_v1 \
    --output-dir results/chatgpt

# Gemini
uv run scripts/run_evaluation.py \
    --agent gemini \
    --model gemini-2.0-flash \
    --benchmark-dir data/benchmark_v1 \
    --output-dir results/gemini

# With LLM-powered NPCs (stateful interviews, lying-aware)
uv run scripts/run_evaluation.py \
    --agent claude \
    --npc-url http://localhost:8000/v1 \
    --npc-model Qwen/Qwen2.5-7B-Instruct \
    --npc-seed 123 \
    --benchmark-dir data/benchmark_v1 \
    --output-dir results/claude_llm_npcs
```

Each run produces per-episode JSON files and a `summary.json`:
```json
{
  "agent": "claude",
  "model": "claude-sonnet-4-20250514",
  "total_episodes": 15,
  "accuracy": 0.73,
  "partial_credit_mean": 0.81,
  "efficiency_mean": 0.64,
  "by_level": {
    "TRIVIAL": {"accuracy": 1.0, "n": 5},
    "EASY":    {"accuracy": 0.8, "n": 5},
    "MEDIUM":  {"accuracy": 0.4, "n": 5}
  }
}
```

### 3. Programmatic API

```python
from benchmark.generator import MysteryGenerator
from mystery_world import ComplexityLevel, COMPLEXITY_PRESETS
from mystery_world.world import MysteryEnvironment
from agents.llm_agent import LLMAgent

# Generate one world
config = COMPLEXITY_PRESETS[ComplexityLevel.MEDIUM]
gen = MysteryGenerator(config, seed=42)
world_state = gen.generate()

# Create environment + agent
env = MysteryEnvironment(world_state)
agent = LLMAgent(provider="anthropic", model="claude-sonnet-4-20250514")
agent.initialize(env, env.get_briefing())

# Run episode
obs = env.get_observation()
while not env.is_done():
    action, args = agent.decide_action(obs)
    obs, reward, done, info = env.step(action, args)

print(env.get_score())
```

---

## Module Reference

### `mystery_world/`

| File | Description |
|------|-------------|
| `__init__.py` | `ComplexityConfig` dataclass (all difficulty knobs), `ComplexityLevel` enum, `COMPLEXITY_PRESETS` dict, `AssetPool` name pools |
| `entities.py` | Core dataclasses: `Location`, `Character`, `Evidence`, `WorldObject`, `TimelineEntry`, `Relationship` |
| `world.py` | `WorldState` (all entities in one place), `MysteryEnvironment` (gym-like env with `step()`, `get_observation()`, action handlers), `AgentAction` enum |
| `events.py` | Simulation tick functions: `process_weather_change`, `process_npc_movement` (Option A random-walk / Option B routine-based), `process_culprit_tampering`, `process_evidence_decay` |
| `narrator.py` | Converts `WorldState` + event log into natural-language observations for the agent; enforces partial observability |
| `npc_responder.py` | `NPCResponder` — stateful LLM-powered NPC interviews via OpenAI-compatible endpoint; `build_npc_system_prompt` injects ground-truth lying directives |

### `benchmark/`

| File | Description |
|------|-------------|
| `generator.py` | `MysteryGenerator` — procedurally builds a `WorldState` from a `ComplexityConfig`; places culprit, lays clues, assigns motives and alibis |
| `serialiser.py` | `WorldState` ↔ JSON; used by generate/load scripts |
| `validator.py` | Sanity-checks a generated world (culprit reachable, evidence discoverable, solution unique) |

### `agents/`

| File | Description |
|------|-------------|
| `base_agent.py` | `BaseAgent` ABC + `BeliefState` dataclass (suspect/weapon/location probability dicts, known facts, reasoning trace) |
| `llm_agent.py` | `LLMAgent` — sends observation history to LLM, parses structured JSON output, updates beliefs; supports Anthropic / OpenAI / Google |
| `heuristic_agent.py` | `HeuristicAgent` — rule-based plan: examine locations → interview all suspects (two questions each) → search for evidence → accuse top-probability suspect |

### `evaluation/`

| File | Description |
|------|-------------|
| `runner.py` | `run_episode(env, agent)` → `ScoreRecord`; `run_benchmark(instances, agent_factory)` → list of `ScoreRecord` |
| `metrics.py` | `ScoreRecord` dataclass; `compute_accuracy`, `compute_partial_credit`, `compute_efficiency` |

### `scripts/`

| File | Description |
|------|-------------|
| `generate_benchmark.py` | CLI: generate N instances per level, write JSON + index |
| `run_evaluation.py` | CLI: load benchmark, run agent, write per-episode results + `summary.json` |

---

## ComplexityConfig Reference

| Field | Default | Description |
|-------|---------|-------------|
| `num_locations` | 5 | Distinct rooms / areas |
| `num_suspects` | 4 | Characters agent must consider as culprits |
| `num_innocents` | 2 | Extra non-suspect NPCs adding noise |
| `num_weapons` | 3 | Candidate murder weapons |
| `num_objects` | 8 | Total interactive objects (clues, props) |
| `num_red_herrings` | 2 | Deliberately misleading clue objects |
| `num_time_steps` | 12 | Simulation steps per episode |
| `evidence_decay_rate` | 0.1 | Per-step probability evidence degrades |
| `witness_memory_half_life` | 6 | Steps until witness recall = 50% |
| `weather_change_prob` | 0.15 | Per-step probability weather changes |
| `npc_move_prob` | 0.3 | Option A: per-step NPC relocation probability |
| `culprit_tamper_prob` | 0.2 | Probability culprit tampers with evidence per step |
| `reactive_events` | False | **True** = Option B routine-based NPC movement (HARD/EXPERT) |
| `alibi_complexity` | 2 | Number of alibi chains agent must verify |
| `motive_layers` | 1 | Depth of nested motive reasoning |
| `requires_deduction` | True | World solvable by strict logical deduction |
| `requires_abduction` | True | Agent must reason to best explanation |
| `evidence_ambiguity` | 0.0 | Probability a clue links to the wrong character |
| `evidence_difficulty_min` | 0.2 | Minimum discovery difficulty (0=obvious) |
| `evidence_difficulty_max` | 0.6 | Maximum discovery difficulty (1=very hidden) |
| `testimony_unreliability` | 0.0 | Probability a witness statement contains errors |
| `allow_suspect_corroborators` | False | Suspects can vouch for each other's alibis |
| `max_corroborators` | 1 | Maximum alibi witnesses per character |
| `culprit_alibi_weights` | (0.30, …, 1.00) | Probability weights over culprit alibi types |
| `max_agent_actions` | 30 | Action budget (episode ends if exceeded) |

### Preset Summary

| Level | Locations | Suspects | Steps | Red Herrings | Routine NPCs | Budget |
|-------|-----------|----------|-------|--------------|--------------|--------|
| TRIVIAL | 3 | 2 | 6 | 0 | No | 15 |
| EASY | 4 | 3 | 8 | 1 | No | 20 |
| MEDIUM | 5 | 4 | 12 | 2 | No | 30 |
| HARD | 7 | 5 | 12 | 3 | **Yes** | 40 |
| EXPERT | 10 | 7 | 24 | 5 | **Yes** | 60 |

---

## NPC Lying System

Lying is deterministically controlled from ground-truth flags — the LLM has no autonomy over deception:

| Condition | Instruction injected into NPC system prompt |
|-----------|---------------------------------------------|
| `char.is_culprit == True` | Deny any involvement; claim your alibi |
| `alibi_corroboration_is_genuine == False` | Confirm the culprit was with you (false alibi) |
| `char.alibi_has_gap == True` | (no special instruction — honest but incomplete) |
| Otherwise | (no instruction — fully truthful NPC) |

The LLM generates fluent natural-language dialogue within these constraints. The agent cannot read these flags and must infer deception from logical inconsistencies.

To use LLM-powered NPCs, start a vLLM server:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000
```
Then pass `--npc-url http://localhost:8000/v1` to `run_evaluation.py`.

---

## NPC Movement (Option A vs B)

Controlled by `reactive_events` in `ComplexityConfig`:

**Option A** (`reactive_events=False`, TRIVIAL/EASY/MEDIUM): Each NPC independently relocates with probability `npc_move_prob` per step to a random adjacent location. Simple, stochastic.

**Option B** (`reactive_events=True`, HARD/EXPERT): NPCs follow believable routines:
- Each NPC has a `home_location_id` (assigned on first step)
- State machine: **home** (dwell 2–4 steps) → **errand/social** (dwell 1–2 steps) → **home**
- Social visits: NPC moves to a location occupied by a character they like (positive relationship)
- Culprit follows the same visible routine as innocents; only deviates for probability-gated tamper runs
- Culprit tamper runs: triggered when at the evidence location; culprit leaves immediately after tampering (does not linger)
- Tamper events have `agent_visible=False` — agent sees consequences (evidence hidden/moved) but not cause

---

## Adding a New Agent

1. Create `agents/my_agent.py` subclassing `BaseAgent`:

```python
from agents.base_agent import BaseAgent
from mystery_world.world import AgentAction, MysteryEnvironment

class MyAgent(BaseAgent):
    def initialize(self, env: MysteryEnvironment, briefing: str) -> None:
        ...  # set up internal state

    def decide_action(self, observation: str) -> tuple[AgentAction, dict]:
        ...  # return (action, action_args)

    def update_beliefs(self, observation: str) -> None:
        ...  # called after each step
```

2. Register it in `scripts/run_evaluation.py` under `AGENT_CONFIGS`:

```python
AGENT_CONFIGS = {
    ...
    "my_agent": {"class": "agents.my_agent.MyAgent", "kwargs": {}},
}
```

3. Run:
```bash
python scripts/run_evaluation.py \
    --agent my_agent \
    --benchmark-dir data/benchmark_v1 \
    --output-dir results/my_agent
```

---

## Scoring

Each episode produces a `ScoreRecord`:

| Metric | Type | Description |
|--------|------|-------------|
| `correct` | bool | Right suspect **and** weapon **and** location |
| `partial_credit` | float 0–1 | Fraction of the three elements correctly identified |
| `efficiency` | float 0–1 | `1 − (actions_used / budget)` — higher = fewer actions needed |

The benchmark summary reports mean accuracy, mean partial credit, and mean efficiency broken down by complexity level.

---

## Reproducibility

Every benchmark instance is fully determined by:
- A **random seed** (controls all procedural generation)
- A **`ComplexityConfig`** (specifies all difficulty knobs)
- **Deterministic transition functions** (world state evolves identically given the same seed)
- **Logged event traces** (full JSON audit trail for post-hoc analysis)

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
