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

### 0. Play it yourself (human mode)

```bash
# Random MEDIUM case
uv run scripts/play.py --npc-url http://localhost:8123/v1 --npc-model Qwen/Qwen3.5-27B

# Choose difficulty and seed
uv run scripts/play.py --level EASY --seed 42 --npc-url http://localhost:8200/v1 --npc-model Qwen/Qwen3.5-27B

# Load a saved world file
uv run scripts/play.py --load data/benchmark_v1/level_1/instance_10042.json
```

Commands in-game: `look`, `go <room>`, `examine <object>`, `search`, `talk <name>`, `take <object>`, `inventory`, `map`, `suspects`, `accuse`, `wait`, `help`, `quit`.

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
    --output-dir results/heuristic \
    --npc-url http://localhost:8123/v1 \
    --npc-model Qwen/Qwen3.5-27B

# Claude Sonnet
uv run scripts/run_evaluation.py \
    --agent claude \
    --model claude-sonnet-4-20250514 \
    --benchmark-dir data/benchmark_v1 \
    --output-dir results/claude_sonnet \
    --npc-url http://localhost:8123/v1 \
    --npc-model Qwen/Qwen3.5-27B

# ChatGPT
uv run scripts/run_evaluation.py \
    --agent chatgpt \
    --model gpt-4o \
    --benchmark-dir data/benchmark_v1 \
    --output-dir results/chatgpt \
    --npc-url http://localhost:8123/v1 \
    --npc-model Qwen/Qwen3.5-27B

# Gemini
uv run scripts/run_evaluation.py \
    --agent gemini \
    --model gemini-2.0-flash \
    --benchmark-dir data/benchmark_v1 \
    --output-dir results/gemini \
    --npc-url http://localhost:8123/v1 \
    --npc-model Qwen/Qwen3.5-27B


# With LLM-powered NPCs (stateful interviews, lying-aware)
uv run scripts/run_evaluation.py \
    --agent claude \
    --npc-url http://localhost:8123/v1 \
    --npc-model Qwen/Qwen2.5-7B-Instruct \
    --npc-seed 123 \
    --benchmark-dir data/benchmark_v1 \
    --output-dir results/claude_llm_npcs \
    --npc-url http://localhost:8123/v1 \
    --npc-model Qwen/Qwen3.5-27B
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
| `play.py` | **Human-player mode** — interactive CLI to play a mystery case yourself |
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
CUDA_VISIBLE_DEVICES=1 uv run vllm serve Qwen/Qwen3.5-27B --port 8123
```
Then pass `--npc-url http://localhost:8123/v1` to `run_evaluation.py` or `play.py`.

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




======


I am designing a murder-mystery benchmark for evaluating LLM-based detective agents. One component of difficulty is determining WHERE the murder actually happened, which is distinct from where the body was found (the culprit moved the body).
   
I want the crime scene location to require genuine reasoning, not just exhaustive room-by-room search. I am considering planting multiple weak, indirect clues that must be aggregated — no single clue is conclusive alone. The agent must combine evidence from different sources to confidently identify the true crime scene.
   
The three clue types I am considering:    
   
1. Victim body trace — The victim's body carries a physical material from the crime scene (e.g. fireplace ash on clothing, sawdust on shoes, paint flakes on hands). The agent examines the body and gets one ambiguous material description. They must then find which room matches that material.
2. Culprit movement trail — The culprit dragged the body through one or more intermediate rooms, leaving partial traces (a smear, a dropped object, disturbed dust). The agent finds these en route but each is inconclusive alone.
3. NPC testimony — An innocent NPC heard or saw something near the true crime scene at roughly the right time (a sound, movement, a glimpse of the culprit from a distance). The agent must interview this NPC and cross-reference their testimony with the physical evidence.
   
My design questions — please discuss each:
   
1. Sufficiency: Is combining any two of these three clue types enough for a capable agent to confidently deduce the crime scene, or do all
 three need to be present? What is the minimum combination that is both challenging and fair?   
2. Ambiguity calibration: How ambiguous should each clue be? For example, should the victim body trace name a material (e.g. "ash") that 
appears in only one room, or in two rooms so the agent must use a second clue to disambiguate? What is the right level of ambiguity per   
difficulty level?    
3. NPC testimony risk: If the NPC witness is the only path to the crime scene and the NPC gives imprecise or wrong information (due to    
unreliable testimony at harder difficulty levels), the puzzle could become unsolvable. How should I handle this — guarantee one reliable  
witness, or ensure the physical clues alone are sufficient as a fallback?
4. Agent strategy implications: A naive agent doing exhaustive search will still find the crime scene eventually. What properties must the
 clue design have to reward reasoning over exhaustive search? For example, should searching a room without having prior clues yield
nothing, while examining specific objects triggered by prior clues reveals the evidence?
5. Benchmark validity: For a research paper, how do I argue that the crime scene deduction task is neither too easy (solvable by search   
alone) nor too hard (requires lucky NPC interviews)? What formal or empirical argument would a reviewer accept?      
6. Related work: Are there existing puzzle design frameworks, interactive fiction systems, or detective game engines that have solved this
 "convergent evidence" design problem? What should I read?   
   
Please be critical and suggest concrete design decisions, not just tradeoffs.

What about this?

1) Trivial: 

Good for tutorial or sanity-check baselines.

Body trace alone points to the correct room
Movement trail is obvious
Witness is clear and specific
Searching rooms directly reveals key evidence immediately

Example:

victim has ash
only one room has ash
witness says they heard something near that room

This is probably too easy for your real benchmark.

2) Easy

Good for beginner agents.

Body trace narrows to 2 rooms
Movement trail clearly favors 1 of those 2
Witness is truthful and fairly specific
Any 2 clues are enough
Unguided room search still works, but is slower

Example:

white dust on victim matches studio or storage room
drag marks pass through corridor connected to studio side
witness heard noise near studio wing

This is a fair starter level.

3) Medium

Probably the best default benchmark level.

Body trace narrows to 2–3 rooms
Movement trail is partial, not complete
Witness is truthful but somewhat vague
Body clue + one other clue is enough
No single clue solves it alone
Blind room search gives only generic descriptions
Targeted inspection reveals the real evidence

Example:

fine black residue matches library fireplace, boiler room, or old workshop
trail suggests body moved through service hallway
witness heard a metallic crash near the lower east side

This level rewards actual clue combination.

4) Hard

Strong reasoning required, but still fair.

Body trace narrows to 3–4 rooms
Movement trail has gaps or decoys
Witness is truthful but low-resolution
Physical clues alone should still be enough as fallback
Search without a hypothesis is inefficient and often inconclusive
Several rooms partially fit, but only one fits all evidence together

Example:

residue is “dark powder with grit,” which could fit multiple utility rooms
trail includes misleading intermediate rooms
witness only knows “somewhere near the service wing”

This is good for testing multi-step deduction.

5) Expert

Use this only if you really want a difficult research challenge.

Each clue alone is quite weak
Body trace points to a material family, not a specific substance
Movement trail includes noise, overlap, or contamination
Witness is truthful but very incomplete
The answer comes from combining:
material match
route logic
timing
room function
Exhaustive search is strongly discouraged by action limits or shallow observations
Scoring should require not just the right room, but the right reasoning chain

Example:

victim has pale mineral dust that could come from several rooms
trail passes through rooms where the body was only moved, not killed
witness only reports a heavy sound from one side of the house at about the right time

This level should feel like real detective reasoning, not clue collection.


- Best final design policy

If you want one clean summary, use this:

1) Trivial

One clue solves it.

2) Easy

Two clues solve it; witness is fairly clear.

3) Medium

Two clues solve it; no clue alone solves it; witness is vague but useful.

4) Hard

Two clues still solve it, but only with real reasoning; physical clues are sufficient even if witness is weak.

5) Expert

Requires combining weak clues, route reasoning, and timing under search limits.

Tell me precisely which line and file to fix. Give me exact code to write or modify. I will copy and paste by myself.

======


uv run scripts/generate_benchmark.py --levels TRIVIAL EASY MEDIUM --instances-per-level 5 --seed 42 --output-dir data/benchmark_v1


There is one issue: in our paper, we must persuade that the solvability is satisfied. How do we know our solvability conditions are adequate? Write me a prompt to discuss with chatgpt and gemini too


I am writing a research paper on a procedurally generated murder-mystery benchmark for evaluating LLM-based detective agents. I need your help critically evaluating whether our solvability conditions are adequate — i.e., whether they guarantee that every benchmark instance can, in principle, be solved by a sufficiently capable agent.
           
What solvability means in our benchmark:         
Each mystery instance specifies a unique culprit, weapon, and murder location. An instance is "solvable" if a rational agent with full access to all observable evidence can logically deduce the correct answer. We currently declare an instance solvable if ALL of the following hold:      
1. At least 2 usable (non-destroyed, non-red-herring) pieces of evidence are linked to the culprit
2. At least one of those pieces is of type PHYSICAL (e.g. fingerprint, bloodstain)
3. The culprit's alibi is "breakable" — meaning one of: (a) no alibi, (b) alibi has no corroborator, (c) corroborator is lying (false alibi), or (d) corroborator is honest but has a time gap       
4. The murder weapon is physically present in some reachable location             
           
Confounding factors our benchmark includes:      
- Red herrings: evidence deliberately linked to innocent suspects   
- Evidence ambiguity: some clues point to the wrong person with a configurable probability        
- Evidence decay: evidence degrades or is destroyed stochastically over time steps
- Culprit tampering: the culprit actively hides or moves evidence (with some probability per step)
- Unreliable testimony: NPC witnesses may give inaccurate statements with configurable probability
- False alibis: a corroborator may lie to cover for the culprit
           
My concern:
Our solvability check is static — it is evaluated at world-generation time, before any simulation steps run. But evidence can decay or be tampered with during the episode. So a world that passes our check at generation time may become unsolvable mid-episode.
Additionally, our check does not verify solution uniqueness — it is possible that a non-culprit suspect also has 2+ pieces of physical evidence pointing to them (e.g. via red herrings), making the solution ambiguous. 
           
Questions I want to discuss:     
1. Are our four solvability conditions necessary and sufficient? What conditions are we missing?
2. How should we handle dynamic unsolvability — evidence that gets destroyed before the agent can find it? Should we guarantee a minimum number of evidence pieces survive to episode end?
3. How do we formally define and check solution uniqueness — that no innocent suspect is equally or more strongly implicated than the culprit given the available evidence?            
4. What is the standard approach in AI benchmark papers for proving or arguing solvability? Are there formal methods, simulation-based arguments, or human annotation protocols we should use?        
5. Are there related benchmarks (e.g. Cluedo, ZorkGPT, MYSTERY, detective games) whose solvability arguments we should cite or borrow from?      
         
Please critique our current conditions and suggest concrete improvements, ideally ones we can implement as automated checks in code.



There is a case that in /mnt/ssd3/thong/mystery_benchmark/sessions/seed42_20260413_141604/world.json that the witness Dahlia Oakley did not witness Silas Greystone stayed with her in the Greenhouse, even though it was. This made the innocent suspcet's alibi not corroborated in the trivial case. Should we fix the engine? Tell me precisely which line and file to fix. Give me exact code to write or modify. I will copy and paste by myself.


In these cases, is the place the culprit committed the murder always the place where the weapon was found?


But this is too easy, right? The agent just searches for the bloodstain and signs of struggle. Is there a way to add hints on victims or culprits too? Discuss first, we decide the implementation later.