# MysteryArena: A Procedural Multi-Agent Benchmark for Evidence-Grounded Investigation and Strategic Deception

## Abstract

Language-model agents are increasingly evaluated in long-horizon interactive settings, yet many benchmarks still test a single agent against a static task or a mostly passive environment. MysteryArena is a procedural multi-agent benchmark for evaluating agents in adversarial murder-mystery games. In each game, a detective agent must gather evidence, infer the culprit, weapon, and crime location, and make a final accusation. A culprit agent observes and acts in the same world while attempting to avoid exposure without making the case unsolvable. The benchmark therefore tests grounded exploration, evidence synthesis, theory revision, strategic deception, and robustness to an adaptive opponent.

MysteryArena generates structured cases with hidden ground truth, text observations, NPC interactions, object evidence, alibis, and a solvability guard that prevents the adversary from destroying the logical path to the solution. Arena matches produce replayable trajectories and two role-specific payoffs: a detective payoff based on accusation accuracy and evidence support, and a culprit payoff based on avoiding exposure. The culprit payoff is not defined as one minus the detective payoff, which prevents strategic loopholes where the culprit benefits from reducing auxiliary detective metrics without actually remaining hidden. We report a preliminary public snapshot of 1,759 matches across five difficulty levels and nine detective and culprit policies. The results show that even strong language models remain brittle in high-difficulty cases, with zero full solves observed on HARD and EXPERT in the current snapshot. MysteryArena is released with a local runner, public result artifacts, and a replay UI to support reproducible agent comparison and qualitative error analysis.

## 1. Introduction

Interactive agent benchmarks have moved beyond single-turn question answering toward tasks that require state tracking, tool use, and long-horizon decision making. Text-game environments such as TextWorld and Jericho test parser-based exploration and planning. ALFWorld and ScienceWorld extend this framing to household and scientific tasks. WebArena and SWE-bench evaluate agents in web and software engineering workflows. These benchmarks have clarified a central failure mode of modern language models: models can often explain what should be done, yet still fail when they must repeatedly act, observe consequences, update beliefs, and recover from mistakes.

Mystery-solving games expose a different but complementary set of capabilities. A detective must search a space, collect clues, reconcile testimony, eliminate impossible suspects, and make a final accusation. The task is not solved by reaching a location or completing a known checklist. It requires forming a latent causal theory from partial observations. At the same time, the environment has a natural adversarial role. A culprit can move, speak, or manipulate the board state to reduce exposure, but should not be allowed to invalidate the puzzle entirely. This creates a benchmark for both sides of an information game: the detective is evaluated on evidence-grounded inference, while the culprit is evaluated on strategic concealment under a solvability constraint.

MysteryArena contributes:

1. A procedural murder-mystery environment with hidden ground truth over suspects, weapons, locations, evidence links, alibis, and NPC state.
2. A two-role Arena protocol in which a detective agent and a culprit agent act in the same world and are evaluated separately.
3. A scoring design that separates detective quality from culprit concealment, avoiding complement-payoff loopholes.
4. Replayable JSONL trajectories and aggregate leaderboards for qualitative and quantitative analysis.
5. A preliminary public result snapshot with 1,759 matches across five difficulty levels and multiple contemporary LLM policies.

The benchmark is intended to measure agent behavior rather than static language ability. It emphasizes whether a model can use observations from the world, not merely whether it can produce plausible mystery-fiction narration.

## 2. Benchmark Overview

Each MysteryArena episode is a generated murder mystery. The hidden solution is a tuple:

```
culprit, weapon, crime_location
```

The environment also contains suspects, rooms, objects, evidence edges, alibis, and narrative observations. The detective receives text observations through actions such as moving, examining locations, examining objects, talking to NPCs, taking objects, checking inventory, waiting, and accusing. The culprit is assigned to one suspect identity and can act in the same world through a role-specific stepping interface. NPCs are controlled separately and are not ranked on the Arena leaderboard.

The core loop is:

1. Generate a seeded case.
2. Initialize a detective policy and a culprit policy.
3. Alternate or schedule actions from the detective, culprit, and NPC responders.
4. Record observations, world-state changes, guard events, and final accusation.
5. Score detective and culprit payoffs.
6. Save both a compact match record and a replayable trajectory.

The Arena view then aggregates matches into role-specific leaderboards, model-by-model matrices, and trajectory replays. This design supports both benchmark-style comparison and detailed debugging of why a model solved or failed a case.

## 3. Environment

### 3.1 Case State

A generated case contains both public state and hidden state. Public state includes room descriptions, visible objects, suspect names, and dialogue observations that become available through actions. Hidden state includes the true culprit, true weapon, true crime location, and the intended evidence graph linking these facts. The agent only accesses hidden state through environment observations.

The target inference is deliberately structured. The detective is not only asked to guess a suspect, but to recover the triangle:

```
suspect -> weapon
weapon -> victim
suspect -> room
```

This makes it possible to distinguish lucky final guesses from evidence-grounded solutions. A detective can receive partial credit for correctly reconstructing parts of the causal chain even when the final accusation is incomplete.

### 3.2 Actions and Observations

The primary detective action space is:

| Action | Purpose |
| --- | --- |
| `MOVE` | Navigate between locations. |
| `EXAMINE_LOCATION` | Inspect the current room for clues. |
| `EXAMINE_OBJECT` | Inspect an object for evidence. |
| `TALK_TO` | Query an NPC or suspect. |
| `TAKE_OBJECT` | Add an object to inventory when allowed. |
| `CHECK_INVENTORY` | Review carried items. |
| `WAIT` | Advance time without changing location. |
| `ACCUSE` | Submit the final suspect, weapon, and location. |

The world implementation also supports optional advanced actions such as `ANALYZE`, `TRAVEL_TIME`, and `CHECK_ROUTE` for richer investigation modes. Policies interact through JSON-like actions with explicit arguments rather than free-form parser commands, which reduces parser ambiguity while retaining long-horizon decision making.

Observations are textual and local. The agent must remember prior observations, decide which leads to pursue, and avoid premature accusations. The same world can be replayed from saved trajectories for visual inspection.

### 3.3 Difficulty Levels

MysteryArena uses five named difficulty levels:

| Level | Intended stressor |
| --- | --- |
| TRIVIAL | Short cases with clearer clues and lower branching. |
| EASY | More distractors and mild ambiguity. |
| MEDIUM | Larger search space and more opportunities for false leads. |
| HARD | Sparse or indirect evidence requiring stronger synthesis. |
| EXPERT | Long-horizon investigation with high branching and weak direct cues. |

The exact generation knobs are implementation details of the case generator, but the levels are designed to monotonically increase the burden on exploration, memory, and evidence integration. The public snapshot confirms that the levels separate current agents sharply: solve rates fall from 57.4 percent on TRIVIAL to zero observed full solves on HARD and EXPERT.

### 3.4 Culprit and NPC Dynamics

The Arena match distinguishes three actor classes:

1. Detective: the ranked investigator policy.
2. Culprit: the ranked adversarial policy assigned to the true culprit identity.
3. NPCs: fixed responder policies used to populate the world and preserve game dynamics.

The culprit is not merely a label in the hidden solution. It can act through an actor-specific environment step, producing changes that are recorded in the trajectory. The culprit identity is randomized through seeded case generation so that repeated games do not collapse into a fixed suspect-index bias.

NPCs are excluded from leaderboard scoring. This avoids conflating a model's detective or culprit skill with the quality of background dialogue generation. NPC policies can be swapped for richer simulations, but the primary Arena comparison is between the detective and culprit roles.

### 3.5 Solvability Guard

Adversarial environments are vulnerable to a trivial failure mode: the adversary can destroy or hide essential evidence, making the task impossible rather than strategically difficult. MysteryArena therefore includes a solvability guard. After a successful non-accusation action that changes the world, the environment checks whether the case remains solvable. If an action would make the solution unreachable or logically unsupported, the change can be rolled back, blocked, or suppressed.

Guard events are diagnostic rather than reward signals. The guard protects benchmark validity; it is not intended to teach the culprit how to win. A strong culprit should reduce exposure while preserving a coherent path to the truth.

## 4. Arena Protocol

### 4.1 Match Definition

A match is identified by:

```
detective_policy, culprit_policy, difficulty_level, seed
```

For each match, the runner initializes the same generated case for both roles. The detective and culprit receive role-appropriate observations and submit actions until the episode ends by accusation, timeout, or step limit. The runner records the full trajectory as compressed JSONL and writes a compact match summary for aggregation.

### 4.2 Leaderboards

MysteryArena reports separate detective and culprit leaderboards. This is important because the two roles are not symmetric. A model that is excellent at evidence synthesis may be poor at concealment, and a model that is difficult to expose may not be a good detective.

The primary leaderboard statistic is mean role payoff. The Arena also computes a role-specific TrueSkill-style rating as an auxiliary ranking signal. TrueSkill is useful for pairwise comparison under non-uniform schedules, but mean payoff remains the most interpretable public metric.

### 4.3 Matrix Evaluation

The benchmark supports matrix evaluation, where each detective policy is paired with each culprit policy across levels and seeds. This exposes matchup-specific behavior. For example, a detective may have high average performance because it performs well against a passive culprit, yet fail against active culprits that alter the board state. Conversely, a culprit may only be effective against detectives that overfit to direct object evidence.

### 4.4 Artifacts and Replay

Each published run contains:

- Match records in JSONL or compressed JSONL.
- Trajectory files with action and observation traces.
- Aggregated leaderboards.
- Matrix summaries.
- An index consumed by the public Arena UI.

The replay UI is part of the benchmark, not only a visualization layer. It allows researchers to inspect whether a score came from correct evidence use, accidental guessing, failure to explore, or a brittle interaction with the culprit policy.

## 5. Scoring

MysteryArena uses role-specific payoffs. The detective is rewarded for solving and supporting the case. The culprit is rewarded for avoiding exposure. These are related but not exact complements.

### 5.1 Detective Payoff

The detective submits a final accusation containing a suspect, weapon, and room. Let:

```
accusation_score = mean(correct_suspect, correct_weapon, correct_room)
```

The environment also evaluates whether the detective recovered the intended evidence triangle:

```
triangle_score =
  F1(SUSPECT_WEAPON)
+ F1(WEAPON_VICTIM)
+ F1(SUSPECT_ROOM)
```

Additional terms reward valid alibi reasoning and correct elimination of innocent suspects:

```
base =
    0.35 * accusation_score
  + 0.35 * (triangle_score / 3.0)
  + 0.15 * alibi_score
  + 0.15 * elimination_score
```

Finally, an examination-efficiency multiplier mildly rewards solving with fewer wasted examinations:

```
detective_payoff = base * (0.8 + 0.2 * examine_efficiency)
```

This design gives partial credit for grounded progress while still prioritizing the final solution and the core causal evidence.

### 5.2 Culprit Payoff

The culprit payoff is based on exposure, not on the detective's full composite score. Let:

```
culprit_exposure =
    0.70 * correct_suspect
  + 0.15 * correct_weapon
  + 0.15 * correct_room
```

Then:

```
culprit_payoff = 1.0 - culprit_exposure
```

This avoids a payoff loophole. If the culprit payoff were simply `1 - detective_payoff`, then a culprit could benefit from reducing auxiliary detective metrics such as evidence-triangle completeness or efficiency, even when the detective still correctly identifies the culprit. MysteryArena instead asks whether the culprit remained unexposed. If the detective names the correct suspect, most of the exposure penalty is incurred regardless of whether the detective's supporting explanation is perfect.

### 5.3 Full Solve

The binary solve rate reports whether the final accusation exactly matches the culprit, weapon, and room. Solve rate is intentionally stricter than detective payoff. Payoff captures partial evidence-grounded progress; solve rate captures complete task success.

## 6. Experimental Setup

We report a preliminary public snapshot of the benchmark artifacts available as of May 28, 2026. The snapshot contains 1,759 match records across five difficulty levels:

| Level | Matches |
| --- | ---: |
| TRIVIAL | 470 |
| EASY | 101 |
| MEDIUM | 405 |
| HARD | 378 |
| EXPERT | 405 |
| Total | 1,759 |

The public snapshot includes nine detective policies and nine culprit policies:

| Detective policies | Culprit policies |
| --- | --- |
| `deepseek-v4-pro` | `deepseek-v4-pro` |
| `glm-4.7` | `glm-4.7` |
| `glm-5` | `glm-5` |
| `glm-5.1` | `glm-5.1` |
| `gpt-5.4-ptu` | `gpt-5.4-ptu` |
| `gpt-5.5` | `gpt-5.5` |
| `heuristic` | `kimi-k2.5` |
| `kimi-k2.5` | `minimax-m2.7` |
| `minimax-m2.7` | `passive` |

The snapshot is not yet a perfectly balanced full matrix. It includes historical runs and incremental fills, so we treat the following results as diagnostic rather than final leaderboard claims. A camera-ready evaluation should rerun a balanced matrix with fixed model versions, fixed inference settings, and equal seeds per matchup.

## 7. Results

### 7.1 Difficulty Scaling

Table 1 reports aggregate performance by difficulty. Solve rate drops sharply as cases become more complex. Detective payoff also decreases, while culprit payoff increases, indicating that current detectives fail to expose culprits in harder settings.

Table 1: Public snapshot performance by difficulty.

| Level | Matches | Solve rate | Detective payoff | Culprit payoff |
| --- | ---: | ---: | ---: | ---: |
| TRIVIAL | 470 | 0.5745 | 0.4028 | 0.5972 |
| EASY | 101 | 0.1089 | 0.2801 | 0.7199 |
| MEDIUM | 405 | 0.2074 | 0.1878 | 0.8122 |
| HARD | 378 | 0.0000 | 0.0780 | 0.9220 |
| EXPERT | 405 | 0.0000 | 0.0298 | 0.9702 |

The non-monotonic solve rate between EASY and MEDIUM reflects the preliminary and non-uniform composition of the current public snapshot. The broader trend is clear: high-difficulty procedural mysteries remain unsolved by the evaluated agents.

### 7.2 Detective Leaderboard

Table 2 reports detective policies ranked by mean detective payoff in the public snapshot.

Table 2: Detective performance by mean payoff.

| Rank | Detective | Matches | Payoff | Solve rate | Mean actions |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `heuristic` | 225 | 0.3529 | 0.0400 | 45.7 |
| 2 | `gpt-5.5` | 185 | 0.1845 | 0.2919 | 85.5 |
| 3 | `glm-5.1` | 196 | 0.1834 | 0.3061 | 75.5 |
| 4 | `glm-5` | 196 | 0.1712 | 0.2755 | 81.4 |
| 5 | `deepseek-v4-pro` | 193 | 0.1652 | 0.0881 | 88.1 |
| 6 | `kimi-k2.5` | 188 | 0.1621 | 0.2181 | 85.7 |
| 7 | `gpt-5.4-ptu` | 193 | 0.1575 | 0.2642 | 84.7 |
| 8 | `minimax-m2.7` | 186 | 0.1557 | 0.2097 | 84.0 |
| 9 | `glm-4.7` | 197 | 0.1547 | 0.2030 | 84.6 |

The heuristic policy ranks first by payoff but has a low full-solve rate. This illustrates why the benchmark reports both continuous payoff and binary solve rate. The heuristic appears to collect or reconstruct partial evidence efficiently, while LLM policies more often produce complete accusations in easier cases. A final benchmark report should include per-level and per-matchup breakdowns to separate these effects.

### 7.3 Culprit Leaderboard

Table 3 reports culprit policies ranked by mean culprit payoff.

Table 3: Culprit performance by mean payoff.

| Rank | Culprit | Matches | Payoff | Detective solve rate | Mean actions |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `passive` | 225 | 0.9583 | 0.0044 | 79.4 |
| 2 | `gpt-5.5` | 189 | 0.7971 | 0.2275 | 78.7 |
| 3 | `glm-5.1` | 193 | 0.7905 | 0.2487 | 78.6 |
| 4 | `glm-5` | 192 | 0.7905 | 0.2292 | 79.0 |
| 5 | `minimax-m2.7` | 190 | 0.7901 | 0.2316 | 78.6 |
| 6 | `kimi-k2.5` | 192 | 0.7886 | 0.2448 | 78.7 |
| 7 | `glm-4.7` | 194 | 0.7875 | 0.2371 | 78.7 |
| 8 | `deepseek-v4-pro` | 190 | 0.7807 | 0.2579 | 79.1 |
| 9 | `gpt-5.4-ptu` | 194 | 0.7760 | 0.2216 | 78.4 |

The passive culprit baseline performs strongly in the preliminary snapshot. This should not be interpreted as proof that passivity is intrinsically optimal. It may indicate that current detectives often fail to solve cases without needing an active adversary, especially at higher difficulty. It also motivates matchup-conditioned analysis: active culprit skill matters most when detectives are strong enough to exploit static evidence.

## 8. Analysis

### 8.1 Evidence Grounding Versus Final Guessing

MysteryArena separates exact solves from evidence-grounded progress. This distinction matters because LLMs can produce plausible final accusations without having collected decisive evidence. Conversely, an agent may gather the right evidence but fail to format or time the final accusation correctly. The composite detective payoff is designed to make these differences measurable.

Replay analysis is particularly important here. A high-scoring trajectory should show a coherent chain: the detective discovers object or location evidence, compares testimony, eliminates alternatives, and accuses based on the accumulated record. A low-quality lucky solve should be visible as a premature or unsupported accusation.

### 8.2 Strategic Deception Under Constraints

The culprit role tests a constrained form of deception. The objective is not to maximize chaos. A culprit that makes the case unsolvable violates the benchmark contract and is blocked by the solvability guard. The intended skill is to remain hidden while preserving enough world consistency for a fair detective.

This is closer to adversarial game play than to unrestricted harmful deception. The benchmark domain is fictional, structured, and auditable. All actions are logged, and the environment enforces constraints on destructive behavior.

### 8.3 Why Not Use Complementary Payoffs?

The initial design question for a two-role benchmark is whether to define:

```
culprit_payoff = 1 - detective_payoff
```

MysteryArena avoids this because detective payoff includes auxiliary quality terms. A detective can correctly expose the culprit while losing points for incomplete alibi discussion, weak edge reconstruction, or inefficient exploration. If the culprit received the exact complement, it would be rewarded for those auxiliary failures even after being identified. The current exposure-based culprit payoff instead aligns the role objective with the semantic goal: avoid being named as the culprit, with smaller penalties for weapon and room exposure.

### 8.4 Difficulty Is Not Just More Steps

The hard cases are not merely longer. They require the agent to preserve a structured belief state under uncertainty. Current LLM policies often show one or more of the following failure modes:

- Over-indexing on the most recently observed clue.
- Treating an unverified NPC statement as decisive.
- Failing to revisit locations after new evidence appears.
- Accusing before connecting suspect, weapon, and room.
- Losing track of which objects were examined versus merely mentioned.
- Producing a plausible explanation that is inconsistent with the hidden evidence graph.

These failure modes are difficult to observe in static QA benchmarks, but become visible in replayed trajectories.

## 9. Reproducibility

The benchmark repository contains the environment, runner, aggregation code, and UI. The public Arena artifacts are organized so that a run can be inspected without rerunning model APIs:

- Compact match records describe policies, seeds, levels, scores, and file paths.
- Compressed trajectory files store the step-by-step interaction.
- Aggregation scripts compute leaderboards and matrices.
- A Streamlit Space renders the public leaderboard and replay view.

A typical local evaluation uses the Arena runner with a fixed set of detectives, culprits, levels, and seeds. For example:

```bash
uv run python scripts/arena_run.py \
  --mode matrix \
  --detectives heuristic gpt-5.5 \
  --culprits passive minimax-m2.7 \
  --levels TRIVIAL EASY MEDIUM \
  --seeds 0 1 2 \
  --workers 4 \
  --out arena/results/example_run
```

The exact command for a final paper should pin model versions, API parameters, prompt templates, and repository commit hash. Public result artifacts are available through the associated Hugging Face dataset, and the replay UI is available through the MysteryArena Space.

## 10. Limitations

The current benchmark has several limitations.

First, the public snapshot reported here is preliminary and not a balanced full matrix. It contains historical and incremental runs. Final claims should use a clean matrix with equal seeds and fixed model configurations.

Second, model APIs can change over time. A robust benchmark release should record provider, model version, decoding parameters, prompt revision, and date of evaluation. Where possible, raw trajectories should be preserved so qualitative conclusions remain inspectable.

Third, the current environment is text-centric. This isolates language-based investigation and strategic reasoning, but does not evaluate visual perception or embodied spatial control.

Fourth, the solvability guard protects benchmark validity but may also constrain some natural adversarial strategies. Future work should report guard-trigger rates and analyze whether strong culprits are being over-constrained.

Fifth, the current hard and expert levels may be too difficult for the evaluated agents. This is useful as a stress test, but leaderboard sensitivity benefits from levels where models are neither saturated nor uniformly failing.

Finally, mystery generation should be validated for narrative diversity, clue fairness, and absence of unintended shortcuts. Procedural benchmarks can accidentally encode artifacts that agents exploit.

## 11. Ethics and Safety

MysteryArena evaluates deception in a fictional, game-like setting. The culprit role is constrained to a generated murder-mystery world and is scored on avoiding exposure inside that world. The benchmark should not be framed as training models for real-world deception. Several design choices reduce risk:

- The domain is synthetic and auditable.
- The action space is structured and limited.
- The solvability guard blocks destructive invalidation of the task.
- Replays expose deceptive actions rather than hiding them.
- Public artifacts should omit API secrets and private provider metadata.

At the same time, adversarial-agent benchmarks should be discussed carefully. Reports should emphasize constrained strategic behavior, not real-world manipulation.

## 12. Future Work

Future versions of MysteryArena should add:

- A balanced public matrix with fixed seeds per matchup.
- Human detective and human culprit baselines.
- Per-level prompt ablations and memory ablations.
- Guard-trigger diagnostics for each culprit policy.
- Stronger analysis of replay failure modes.
- Case-generation audits for shortcut artifacts.
- Confidence intervals or bootstrap intervals for leaderboard metrics.
- Additional role variants, such as cooperative detective teams or noisy witnesses.

The most important next step is a clean, fully balanced evaluation run. The current snapshot is sufficient to demonstrate the benchmark and expose major failure modes, but a paper-ready leaderboard should remove historical imbalance.

## 13. Conclusion

MysteryArena is a procedural benchmark for evaluating language-model agents in adversarial mystery games. It tests capabilities that are underrepresented in static QA and single-agent task completion: evidence-grounded investigation, belief-state maintenance, strategic concealment, and fair competition under a solvability constraint. The preliminary public snapshot shows that current agents remain far from reliable on harder generated mysteries. By releasing replayable trajectories, role-specific scoring, and public aggregation tools, MysteryArena provides a practical platform for studying both how agents solve cases and how they fail.

## References

Cote, M.-A., Kadar, A., Yuan, X., Kybartas, B., Barnes, T., Fine, E., Moore, J., Hausknecht, M., Asri, L. E., Adada, M., Tay, W., and Trischler, A. 2018. TextWorld: A Learning Environment for Text-based Games. arXiv:1806.11532. https://arxiv.org/abs/1806.11532

Hausknecht, M., Ammanabrolu, P., Cote, M.-A., and Yuan, X. 2020. Interactive Fiction Games: A Colossal Adventure. arXiv:1909.05398. https://arxiv.org/abs/1909.05398

Herbrich, R., Minka, T., and Graepel, T. 2006. TrueSkill: A Bayesian Skill Rating System. NeurIPS 2006. https://proceedings.neurips.cc/paper/2006/file/f44ee263952e65b3610b8ba51229d1f9-Paper.pdf

Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. 2023. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? arXiv:2310.06770. https://arxiv.org/abs/2310.06770

Liu, X., Yu, H., Zhang, H., Xu, Y., Lei, X., Lai, H., Gu, Y., Ding, H., Men, K., Yang, K., Zhang, S., Deng, X., Zeng, A., Du, Z., Zhang, C., Shen, S., Zhang, T., Su, Y., Sun, H., Huang, M., Dong, Y., and Tang, J. 2023. AgentBench: Evaluating LLMs as Agents. arXiv:2308.03688. https://arxiv.org/abs/2308.03688

Shridhar, M., Yuan, X., Cote, M.-A., Bisk, Y., Trischler, A., and Hausknecht, M. 2020. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. arXiv:2010.03768. https://arxiv.org/abs/2010.03768

Wang, R., Jansen, P., Cote, M.-A., and Ammanabrolu, P. 2022. ScienceWorld: Is your Agent Smarter than a 5th Grader? arXiv:2203.07540. https://arxiv.org/abs/2203.07540

Zhou, S., Xu, F. F., Zhu, H., Zhou, X., Lo, R., Sridhar, A., Cheng, X., Ou, T., Bisk, Y., Fried, D., Alon, U., and Neubig, G. 2023. WebArena: A Realistic Web Environment for Building Autonomous Agents. arXiv:2307.13854. https://arxiv.org/abs/2307.13854

MysteryArena public Space. https://huggingface.co/spaces/Elfsong/Mystery_Arena/tree/main

MysteryArena public results dataset. https://huggingface.co/datasets/Elfsong/Mystery_Arena_Results
