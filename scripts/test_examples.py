"""
Replay each example's oracle action sequence and verify scores match.

Usage:
    python scripts/test_examples.py [--examples-dir examples/]
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

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
TOLERANCE = 1e-3  # score comparison tolerance


def _level_from_name(name: str) -> ComplexityLevel:
    return ComplexityLevel[name.upper()]


def _replay_example(ex: dict) -> dict:
    """Replay oracle_action_sequence against a fresh env; return result dict."""
    level  = _level_from_name(ex["complexity"])
    config = COMPLEXITY_PRESETS[level]
    state  = generate_mystery(seed=ex["seed"], config=config)
    env    = MysteryEnvironment(state)
    _      = render_initial_briefing(env)  # initialise env (records start)

    for step_rec in ex["oracle_action_sequence"]:
        action_name = step_rec["action"]
        kwargs      = step_rec["kwargs"]
        action      = AgentAction[action_name]
        env.step(action, **kwargs)

    summary  = env.get_episode_summary()
    score    = summary.get("score_result", {}) or {}
    correct  = summary.get("accusation_correct", False)

    replayed_scores = {
        "composite":   round(score.get("composite_score",        0.0), 4),
        "triangle":    round(score.get("triangle_score",         0.0), 4),
        "alibi":       round(score.get("alibi_score",            0.0), 4),
        "elimination": round(score.get("elimination_score",      0.0), 4),
        "sw":          round(score.get("suspect_weapon_score",   0.0), 4),
        "wv":          round(score.get("weapon_victim_score",    0.0), 4),
        "sr":          round(score.get("suspect_room_score",     0.0), 4),
    }
    return {
        "id":              ex["id"],
        "accusation_correct": correct,
        "replayed_scores": replayed_scores,
        "expected_scores": ex["oracle_scores"],
    }


def _check(result: dict) -> tuple[bool, list[str]]:
    """Return (passed, list_of_failures)."""
    failures: list[str] = []
    if not result["accusation_correct"]:
        failures.append("accusation_correct=False")
    for key, expected in result["expected_scores"].items():
        got = result["replayed_scores"].get(key, 0.0)
        if abs(got - expected) > TOLERANCE:
            failures.append(f"{key}: expected={expected:.4f} got={got:.4f}")
    return (len(failures) == 0), failures


def run_all(examples_dir: Path = EXAMPLES_DIR) -> int:
    paths = sorted(examples_dir.glob("*.json"))
    if not paths:
        print(f"No example files found in {examples_dir}")
        return 1

    passed = failed = 0
    for path in paths:
        ex     = json.loads(path.read_text())
        result = _replay_example(ex)
        ok, failures = _check(result)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {result['id']}")
        if not ok:
            for f in failures:
                print(f"       ! {f}")
            failed += 1
        else:
            passed += 1

    print(f"\n{passed}/{passed+failed} examples passed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test benchmark examples")
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=EXAMPLES_DIR,
        help="Directory containing example JSON files",
    )
    args = parser.parse_args()
    sys.exit(run_all(args.examples_dir))
