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
│  │  generate.py   │   │                      │   │  runner.py    │  │
│  │  verify.py     │   │  ┌────────────────┐  │   │  metrics.py   │  │
│  └────────────────┘   │  │   WorldState   │  │   └───────┬───────┘  │
│          │            │  │  locations     │  │           │          │
│  ┌───────▼──────────┐ │  │  characters    │  │   ┌───────▼───────┐  │
│  │  ComplexityConfig│ │  │  objects       │  │   │   agents/     │  │
│  │  5 presets:      │ │  │  evidence      │  │   │  LLMAgent     │  │
│  │  TRIVIAL → EXPERT│ │  └────────────────┘  │   │  HeuristicAgt │  │
│  └──────────────────┘ │                      │   │  OracleAgent  │  │
│                       │  ┌────────────────┐  │   └───────────────┘  │
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
    │         │                                                      │
    │         ├── MOVE / EXAMINE_LOCATION                            │
    │         ├── EXAMINE_OBJECT                                     │
    │         ├── ANALYZE / TRAVEL_TIME / CHECK_ROUTE                │
    │         ├── TALK_TO ──► NPCResponder (LLM or template)         │
    │         ├── ACCUSE ──► score & end episode                     │
    │         └── WAIT / CHECK_INVENTORY / TAKE_OBJECT               │
    └────────────────────────────────────────────────────────────────┘
                          │
                     EpisodeMetrics
```

### NPC Interview Flow (stateful, lying-aware)

```
Agent              world.py              npc_responder.py       vLLM endpoint
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

### 0. Play it yourself (human mode)

```bash
# Random MEDIUM case
uv run scripts/play.py --npc-url http://localhost:8200/v1 --npc-model Qwen/Qwen3.5-27B

# Choose difficulty and seed
uv run scripts/play.py --level TRIVIAL --seed 42 --npc-url http://localhost:8200/v1 --npc-model Qwen/Qwen3.5-27B

# Play a pre-generated benchmark example by ID
uv run scripts/play.py --example trivial_seed_0

# Load a saved world file
uv run scripts/play.py --load data/benchmark_v1/level_1/instance_10042.json
```

Commands in-game: `look`, `go <room>`, `examine <object>`, `search`, `talk <name>`, `take <object>`, `inventory`, `map`, `suspects`, `accuse`, `wait`, `hint`, `help`, `quit`.

The `hint` command runs the oracle agent against the current game state and prints its recommended next action.

### 1. Play the pre-generated benchmark examples

Twenty curated cases (4 per difficulty level) are included in `examples/`. Each records the ground-truth answer and the oracle's full action sequence, making them ideal for benchmarking agents or comparing scores across players.

```bash
# List all available examples
python scripts/list_examples.py

# Filter by level
python scripts/list_examples.py --level EASY

# Show full details + human-readable oracle walkthrough for one example
python scripts/list_examples.py --show easy_seed_0

# Play a specific example
uv run scripts/play.py --example trivial_seed_0     # easiest warmup
uv run scripts/play.py --example medium_seed_0
uv run scripts/play.py --example expert_seed_3      # hardest
```

Example listing:
```
ID                         LEVEL      SEED  MULTI  CULPRIT                         COMPOSITE
--------------------------------------------------------------------------------------------
trivial_seed_0             TRIVIAL       0    yes  Rosalind Iverson                   1.0000
trivial_seed_1             TRIVIAL       1    yes  Fern Iverson                       1.0000
trivial_seed_2             TRIVIAL       2    yes  Adrian Iverson                     1.0000
trivial_seed_3             TRIVIAL       3    yes  Silas Elsworth                     1.0000
easy_seed_0                EASY          0    yes  Silas Prescott                     1.0000
easy_seed_1                EASY          1    yes  Silas Elsworth                     1.0000
easy_seed_2                EASY          2    yes  Rosalind Blackwood                 0.9611
easy_seed_3                EASY          3    yes  Petra Harlow                       0.9611
medium_seed_0              MEDIUM        0    yes  Fern Montague                      0.9500
medium_seed_1              MEDIUM        1     no  Nadia Oakley                       0.9500
medium_seed_2              MEDIUM        2    yes  Beatrix Greystone                  0.9111
medium_seed_22             MEDIUM       22     no  Nadia Ashworth                     0.9000
hard_seed_84               HARD         84     no  Adrian Crane                       0.8875
hard_seed_0                HARD          0    yes  Petra Iverson                      0.6133
hard_seed_1                HARD          1    yes  Petra Iverson                      0.6133
hard_seed_83               HARD         83     no  Rosalind Juno                      0.6042
expert_seed_0              EXPERT        0    yes  Petra Quinlan                      0.7000
expert_seed_1              EXPERT        1    yes  Thea Blackwood                     0.6944
expert_seed_2              EXPERT        2    yes  Thea Blackwood                     0.6944
expert_seed_3              EXPERT        3    yes  Orson Juno                         0.6528
```

`MULTI=yes` means at least one Locard triangle edge has multiple valid evidence pieces — earning full triangle credit requires citing the right IDs.

Each example JSON (`examples/<id>.json`) contains:

| Field | Description |
|-------|-------------|
| `ground_truth` | Culprit name, weapon, and murder location |
| `alibi_claims` | Exact alibi text the culprit will claim |
| `eliminations` | Which innocents have SUSPECT_ELSEWHERE evidence and who corroborates them |
| `oracle_plan` | Evidence IDs per triangle edge, alibi type, alibi contradiction |
| `oracle_action_sequence` | Every MOVE / EXAMINE / TALK / ACCUSE step the oracle takes |
| `oracle_scores` | Composite, triangle, alibi, elimination, and per-edge scores |

To regenerate or extend the set:

```bash
python scripts/generate_examples.py    # writes examples/<id>.json
python scripts/test_examples.py        # replays every oracle sequence, verifies scores match
```

### 2. Generate a benchmark suite

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

### 3. Run evaluation

```bash
# Heuristic baseline (no API key needed)
uv run scripts/run_evaluation.py \
    --agent heuristic \
    --benchmark-dir data/benchmark_v1 \
    --output-dir results/heuristic \
    --npc-url http://localhost:8200/v1 \
    --npc-model Qwen/Qwen3.5-27B

# Claude Sonnet
uv run scripts/run_evaluation.py \
    --agent claude \
    --model claude-sonnet-4-6 \
    --benchmark-dir data/benchmark_v1 \
    --output-dir results/claude_sonnet \
    --npc-url http://localhost:8200/v1 \
    --npc-model Qwen/Qwen3.5-27B

# ChatGPT
uv run scripts/run_evaluation.py \
    --agent chatgpt \
    --model gpt-4o \
    --benchmark-dir data/benchmark_v1 \
    --output-dir results/chatgpt \
    --npc-url http://localhost:8200/v1 \
    --npc-model Qwen/Qwen3.5-27B

# Gemini
uv run scripts/run_evaluation.py \
    --agent gemini \
    --model gemini-2.0-flash \
    --benchmark-dir data/benchmark_v1 \
    --output-dir results/gemini \
    --npc-url http://localhost:8200/v1 \
    --npc-model Qwen/Qwen3.5-27B
```

Each run produces per-episode JSON files and a `summary.json`:
```json
{
  "agent": "claude",
  "model": "claude-sonnet-4-6",
  "total_instances": 15,
  "solved": 11,
  "solve_rate": 0.73,
  "avg_partial_score": 0.81,
  "npc_model": "Qwen/Qwen3.5-27B"
}
```

### 4. Run the oracle (calibration upper bound)

The oracle knows the full ground truth and executes the cheapest legal proof:
one clue per Locard triangle edge + alibi contradiction, via the shortest route.
It cannot skip discovery — it must call the game API to find each clue before citing it.

```python
from mystery_world import ComplexityLevel, COMPLEXITY_PRESETS
from mystery_world.generator import generate_mystery
from mystery_world.narrator import render_initial_briefing
from mystery_world.world import MysteryEnvironment
from agents.oracle_agent import OracleAgent

config = COMPLEXITY_PRESETS[ComplexityLevel.MEDIUM]
state  = generate_mystery(seed=42, config=config)
env    = MysteryEnvironment(state)
agent  = OracleAgent()

result = agent.run(env, render_initial_briefing(env))
print(result["accusation_correct"])   # True
print(result["actions_taken"])        # minimal action count
print(result["plan_summary"])         # which evidence was used per edge
```

To sweep all seeds and levels:

```python
from mystery_world import ComplexityLevel, COMPLEXITY_PRESETS
from mystery_world.generator import generate_mystery
from mystery_world.narrator import render_initial_briefing
from mystery_world.world import MysteryEnvironment
from agents.oracle_agent import OracleAgent

for level in ComplexityLevel:
    config = COMPLEXITY_PRESETS[level]
    scores = []
    for seed in range(20):
        state  = generate_mystery(seed=seed, config=config)
        env    = MysteryEnvironment(state)
        agent  = OracleAgent()
        result = agent.run(env, render_initial_briefing(env))
        scores.append(result["accusation_correct"])
    print(f"{level.name:8s}  solve_rate={sum(scores)/len(scores):.2f}")
```

### 5. Programmatic API

```python
from mystery_world import ComplexityLevel, COMPLEXITY_PRESETS
from mystery_world.generator import generate_mystery
from mystery_world.narrator import render_initial_briefing, render_step_observation
from mystery_world.world import AgentAction, MysteryEnvironment
from agents.llm_agent import LLMAgent

# Generate one world
config = COMPLEXITY_PRESETS[ComplexityLevel.MEDIUM]
state  = generate_mystery(seed=42, config=config)

# Create environment + agent
env     = MysteryEnvironment(state)
agent   = LLMAgent(provider="anthropic", model="claude-sonnet-4-6")
briefing = render_initial_briefing(env)
agent.initialize(env, briefing)

# Run episode
obs = briefing
while not env.is_solved and env.budget_remaining > 0:
    action, kwargs = agent.decide_action(obs)
    result = env.step(action, **kwargs)
    obs = render_step_observation(env, result.observation)
    agent.update_beliefs(obs)

print(env.get_episode_summary())
```

---

## Module Reference

### `mystery_world/`

| File | Description |
|------|-------------|
| `__init__.py` | `ComplexityConfig` dataclass (all difficulty knobs), `ComplexityLevel` enum, `COMPLEXITY_PRESETS` dict, `AssetPool` name pools |
| `entities.py` | Core dataclasses: `Location`, `Character`, `Evidence`, `WorldObject`, `EdgeRelevance`, `AlibiClaim`, `TimelineEntry` |
| `world.py` | `WorldState` (ground truth), `MysteryEnvironment` (agent-facing `step()` API, action handlers, Locard triangle scorer), `AgentAction` enum |
| `generator.py` | `generate_mystery(seed, config)` — procedurally builds a full `WorldState`; places culprit, lays clues, assigns motives and alibis |
| `events.py` | Simulation tick functions: `process_weather_change`, `process_npc_movement` (Option A random-walk / Option B routine-based), `process_culprit_tampering`, `process_evidence_decay` |
| `narrator.py` | Converts `WorldState` + event log into natural-language observations; enforces partial observability; `render_initial_briefing`, `render_step_observation` |
| `npc_responder.py` | `NPCResponder` — stateful LLM-powered NPC interviews via OpenAI-compatible endpoint; `build_npc_system_prompt` injects ground-truth lying directives |

### `benchmark/`

| File | Description |
|------|-------------|
| `generate.py` | `generate_benchmark_suite()` — generates N instances per level, saves JSON + `manifest.json`; `load_benchmark_suite()` for loading |
| `verify.py` | `check_structural_consistency`, `check_solvability`, `check_locard_solvability` — sanity checks; `export_annotation_sheet` for human review |

### `agents/`

| File | Description |
|------|-------------|
| `base_agent.py` | `BaseAgent` ABC + `BeliefState` dataclass (suspect/weapon/location probability dicts, known facts, reasoning trace) |
| `llm_agent.py` | `LLMAgent` — sends observation history to LLM, parses structured JSON output, updates beliefs; supports Anthropic / OpenAI / Google |
| `heuristic_agent.py` | `HeuristicAgent` — rule-based plan: examine locations → interview all suspects → search for evidence → accuse top-probability suspect |
| `symbolic_agent.py` | `SymbolicAgent` — maintains an explicit logical fact base; eliminates suspects/weapons by contradiction before accusing |
| `oracle_agent.py` | `OracleAgent` — calibration upper bound; reads ground truth, plans the cheapest legal proof (one clue per Locard edge + alibi), executes through the normal game API |

### `evaluation/`

| File | Description |
|------|-------------|
| `runner.py` | `run_episode(agent, world_state)` → `EpisodeResult`; `run_benchmark(agent_factory, instances, output_dir)` → list of results |
| `metrics.py` | `EpisodeMetrics` dataclass; `compute_episode_metrics`; `aggregate_metrics` by complexity level |

### `scripts/`

| File | Description |
|------|-------------|
| `play.py` | **Human-player mode** — interactive CLI; supports `--example <id>`, `--level`, `--seed`, `--load` |
| `generate_examples.py` | Generate the 20 curated benchmark examples (4 per level) into `examples/` |
| `list_examples.py` | Print a table of all examples; `--show <id>` prints full details and a human-readable oracle walkthrough |
| `test_examples.py` | Replay each example's oracle action sequence and verify scores match (regression test) |
| `generate_benchmark.py` | CLI: generate N instances per level, write JSON + index |
| `run_evaluation.py` | CLI: load benchmark, run agent, write per-episode results + `summary.json` |
| `analyze_results.py` | Post-hoc analysis and plots over saved results |
| `demo_pipeline.py` | End-to-end demo: generate → run oracle → run LLM agent → compare |

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
| `step_duration_minutes` | 30 | Real-world minutes per step (used for clock timestamps) |
| `world_start_hour` | 20 | Hour the evening begins (20 = 8:00 PM) |
| `freshness_threshold` | 2.0 | Steps within which evidence is considered "fresh" for Locard scoring |
| `num_route_constraints` | 1 | Passages that were blocked during the murder window |
| `evidence_decay_rate` | 0.1 | Per-step probability evidence degrades |
| `witness_memory_half_life` | 6 | Steps until witness recall = 50% |
| `weather_change_prob` | 0.15 | Per-step probability weather changes |
| `npc_move_prob` | 0.3 | Option A: per-step NPC relocation probability |
| `culprit_tamper_prob` | 0.2 | Probability culprit tampers with evidence per step |
| `reactive_events` | False | **True** = Option B routine-based NPC movement (HARD/EXPERT) |
| `evidence_ambiguity` | 0.0 | Probability a clue links to the wrong character |
| `evidence_difficulty_min` | 0.2 | Minimum discovery difficulty (0=obvious) |
| `evidence_difficulty_max` | 0.6 | Maximum discovery difficulty (1=very hidden) |
| `testimony_unreliability` | 0.0 | Probability a witness statement contains errors |
| `allow_suspect_corroborators` | False | Suspects can vouch for each other's alibis |
| `max_agent_actions` | 30 | Action budget (episode ends if exceeded) |

### Preset Summary

| Level | Locations | Suspects | Budget | Freshness | Route Constraints |
|-------|-----------|----------|--------|-----------|-------------------|
| TRIVIAL | 3 | 2 | 15 | 3.0 | 0 |
| EASY | 4 | 3 | 20 | 2.5 | 0 |
| MEDIUM | 5 | 4 | 30 | 2.0 | 1 |
| HARD | 7 | 5 | 40 | 1.5 | 2 |
| EXPERT | 10 | 7 | 60 | 1.0 | 3 |

---

## Scoring

Each accusation is scored on three dimensions:

### 1. Accusation score (0–1)
Fraction of the three elements (suspect, weapon, location) correctly identified.

### 2. Locard triangle score (0–3)
Each edge is scored with **F1** (harmonic mean of precision and recall) against the set of valid evidence IDs for that edge:

| Edge | What it requires |
|------|-----------------|
| `SUSPECT_WEAPON` | Evidence linking culprit ↔ murder weapon |
| `WEAPON_VICTIM` | Evidence linking murder weapon ↔ victim |
| `SUSPECT_ROOM` | Evidence linking culprit ↔ murder location |

Evidence counts as valid if it is non-red-herring, its contact timestamp falls within `freshness_threshold` steps of the murder, and it is linked to the correct ground-truth entities. Citing a superset of valid IDs earns full recall; missing any valid ID reduces recall proportionally.

### 3. Alibi score (0–1)
Awarded for correctly citing the culprit's alibi claim and a valid contradiction. Two alibi types:
- **Type A** — culprit claims a location they could not have been at during the murder window
- **Type B** — bracketing location claims whose only connecting route passes through the crime scene

### 4. Elimination score (0–1)
Awarded for correctly clearing innocent suspects using `SUSPECT_ELSEWHERE` evidence. Formula:
```
elimination = max(0, (correct_eliminations − 2 × incorrect_eliminations) / total_innocents)
```
Characters who appear only as alibi corroborators (never as alibi targets) are excluded from `total_innocents`.

### Composite score
```
base      = 0.35 × accusation + 0.35 × (triangle / 3) + 0.15 × alibi + 0.15 × elimination
composite = base × (0.8 + 0.2 × examine_efficiency)
```
`examine_efficiency` is the fraction of examined objects that were relevant evidence (rewards focused investigation).

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
CUDA_VISIBLE_DEVICES=1 uv run vllm serve Qwen/Qwen3.5-27B --port 8200
```
Then pass `--npc-url http://localhost:8200/v1` to `run_evaluation.py` or `play.py`.

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
        ...  # return (action, action_kwargs)

    def update_beliefs(self, observation: str) -> None:
        ...  # called after each step
```

2. Register it in `scripts/run_evaluation.py` under `AGENT_CONFIGS`:

```python
AGENT_CONFIGS = {
    ...
    "my_agent": {"provider": None, "model": None, "description": "My custom agent"},
}
```

3. Run:
```bash
uv run scripts/run_evaluation.py \
    --agent my_agent \
    --benchmark-dir data/benchmark_v1 \
    --output-dir results/my_agent \
    --npc-url http://localhost:8200/v1 \
    --npc-model Qwen/Qwen3.5-27B
```

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
