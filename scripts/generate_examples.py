"""
Generate 20 benchmark examples across all complexity levels.

For each example, records:
  - ground truth (culprit / weapon / location)
  - oracle plan (which evidence IDs per edge, alibi, eliminations)
  - oracle action sequence (every MOVE / EXAMINE / TALK / ACCUSE step)
  - final scores

At least 2 examples per level have multiple evidence pieces on at least one edge.
Output: examples/<id>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mystery_world import COMPLEXITY_PRESETS, ComplexityLevel
from mystery_world.generator import generate_mystery
from mystery_world.narrator import render_initial_briefing
from mystery_world.world import AgentAction, MysteryEnvironment
from agents.oracle_agent import OracleAgent

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLES_DIR.mkdir(exist_ok=True)

# ── Selection criteria ────────────────────────────────────────────────────────
# For each level, try seeds 0..99 and pick:
#   • 2 examples where ≥1 edge has multiple evidence IDs (multi_evidence=True)
#   • 2 more where any correct accusation is fine
LEVELS = [
    ComplexityLevel.TRIVIAL,
    ComplexityLevel.EASY,
    ComplexityLevel.MEDIUM,
    ComplexityLevel.HARD,
    ComplexityLevel.EXPERT,
]
TARGET_PER_LEVEL = 4          # total examples per level
MULTI_EV_PER_LEVEL = 2        # minimum that must have multi-evidence on ≥1 edge


def _run_oracle(level: ComplexityLevel, seed: int) -> dict | None:
    """Run oracle and return a full example dict, or None if accusation fails."""
    config = COMPLEXITY_PRESETS[level]
    state  = generate_mystery(seed=seed, config=config)
    env    = MysteryEnvironment(state)
    agent  = OracleAgent()
    result = agent.run(env, render_initial_briefing(env))

    if not result["accusation_correct"]:
        return None

    summary = env.get_episode_summary()
    score   = summary.get("score_result", {}) or {}
    plan    = result["plan_summary"]

    # Capture full action sequence
    action_sequence = []
    for i, record in enumerate(env.action_history, 1):
        action_sequence.append({
            "step":   i,
            "action": record["action"],
            "kwargs": record["kwargs"],
        })

    # Ground truth
    culprit    = state.get_culprit()
    weapon_obj = state.objects.get(state.murder_weapon_id)
    murder_loc = state.locations.get(state.murder_location_id)

    # Alibi claims
    alibi_claims = []
    if culprit:
        for c in culprit.alibi_claims:
            alibi_claims.append({"location": c.location_name, "time": c.clock_time_str})

    # Elimination ground truth
    from mystery_world.entities import EdgeType
    eliminations_gt = []
    for ev in state.evidence.values():
        if (
            ev.relevance is not None
            and ev.relevance.edge_type == EdgeType.SUSPECT_ELSEWHERE
            and not ev.is_red_herring
            and ev.linked_character_id
            and ev.corroborator_id
        ):
            innocent = state.characters.get(ev.linked_character_id)
            corr     = state.characters.get(ev.corroborator_id)
            if innocent and corr and not innocent.is_culprit:
                eliminations_gt.append({
                    "suspect":      innocent.full_name,
                    "evidence_id":  ev.id,
                    "corroborator": corr.full_name,
                })

    sw = plan.get("sw_evidence", [])
    wv = plan.get("wv_evidence", [])
    sr = plan.get("sr_evidence", [])
    multi_evidence = any(len(ids) > 1 for ids in [sw, wv, sr])

    return {
        "id":              f"{level.name.lower()}_seed_{seed}",
        "complexity":      level.name,
        "seed":            seed,
        "multi_evidence":  multi_evidence,
        "ground_truth": {
            "culprit":  culprit.full_name  if culprit    else "",
            "weapon":   weapon_obj.name    if weapon_obj else "",
            "location": murder_loc.name    if murder_loc else "",
        },
        "alibi_claims": alibi_claims,
        "eliminations": eliminations_gt,
        "oracle_plan": {
            "sw_evidence":        sw,
            "wv_evidence":        wv,
            "sr_evidence":        sr,
            "alibi_type":         plan.get("alibi_type", "none"),
            "alibi_contradiction": (
                env.action_history[-1].get("kwargs", {}).get("alibi_contradiction")
                if env.action_history else None
            ),
        },
        "oracle_action_sequence": action_sequence,
        "oracle_scores": {
            "composite":  round(score.get("composite_score",  0.0), 4),
            "triangle":   round(score.get("triangle_score",   0.0), 4),
            "alibi":      round(score.get("alibi_score",      0.0), 4),
            "elimination":round(score.get("elimination_score",0.0), 4),
            "sw":         round(score.get("suspect_weapon_score", 0.0), 4),
            "wv":         round(score.get("weapon_victim_score",  0.0), 4),
            "sr":         round(score.get("suspect_room_score",   0.0), 4),
        },
    }


def generate() -> list[dict]:
    chosen: list[dict] = []

    for level in LEVELS:
        level_examples: list[dict] = []
        multi_count = 0

        for seed in range(100):
            if len(level_examples) >= TARGET_PER_LEVEL:
                break
            ex = _run_oracle(level, seed)
            if ex is None:
                continue
            # Prioritise multi-evidence examples to fill quota first
            if ex["multi_evidence"] and multi_count < MULTI_EV_PER_LEVEL:
                level_examples.append(ex)
                multi_count += 1
            elif not ex["multi_evidence"] and (len(level_examples) - multi_count) < (TARGET_PER_LEVEL - MULTI_EV_PER_LEVEL):
                level_examples.append(ex)

        # If we still haven't filled the multi-evidence quota, accept any correct
        for seed in range(100):
            if len(level_examples) >= TARGET_PER_LEVEL:
                break
            if any(e["seed"] == seed and e["complexity"] == level.name for e in level_examples):
                continue
            ex = _run_oracle(level, seed)
            if ex:
                level_examples.append(ex)

        chosen.extend(level_examples[:TARGET_PER_LEVEL])
        print(
            f"{level.name}: {len(level_examples[:TARGET_PER_LEVEL])} examples "
            f"({sum(e['multi_evidence'] for e in level_examples[:TARGET_PER_LEVEL])} multi-evidence)"
        )

    return chosen


def save(examples: list[dict]) -> None:
    for ex in examples:
        path = EXAMPLES_DIR / f"{ex['id']}.json"
        path.write_text(json.dumps(ex, indent=2))
    print(f"\nSaved {len(examples)} examples to {EXAMPLES_DIR}/")


if __name__ == "__main__":
    print("Generating examples …\n")
    examples = generate()
    save(examples)
    print(f"\nTotal: {len(examples)} examples")
    multi_total = sum(e["multi_evidence"] for e in examples)
    print(f"Multi-evidence examples: {multi_total}/{len(examples)}")
