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
from mystery_world.world import AgentAction, MysteryEnvironment, WorldState

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
TOLERANCE = 1e-3  # score comparison tolerance


def _level_from_name(name: str) -> ComplexityLevel:
    return ComplexityLevel[name.upper()]


def _ground_truth_tuple(state: WorldState) -> tuple[str, str, str]:
    culprit = state.get_culprit()
    weapon = state.objects.get(state.murder_weapon_id)
    location = state.locations.get(state.murder_location_id)
    return (
        culprit.full_name if culprit else "",
        weapon.name if weapon else "",
        location.name if location else "",
    )


def _expected_ground_truth_tuple(ex: dict) -> tuple[str, str, str]:
    ground_truth = ex.get("ground_truth", {})
    return (
        ground_truth.get("culprit", ""),
        ground_truth.get("weapon", ""),
        ground_truth.get("location", ""),
    )


def _state_matches_example(state: WorldState, ex: dict) -> bool:
    return _ground_truth_tuple(state) == _expected_ground_truth_tuple(ex)


def _state_for_example(ex: dict, config) -> tuple[WorldState, int]:
    """Reconstruct the world state represented by an example fixture.

    Older fixtures only recorded the input seed. Newer generator versions may
    retry with seed+attempt before returning a solvable world, so replaying an
    old fixture against bare ``seed`` can produce a different case. Prefer the
    recorded attempt when available; otherwise match by recorded ground truth.
    """
    seed = int(ex["seed"])
    if "generation_attempt" in ex:
        attempt = int(ex["generation_attempt"])
        expected_state_seed = ex.get("state_seed")
        candidates = [
            generate_mystery(seed=seed, config=config, max_retries=attempt + 1)
        ]
        if attempt:
            candidates.append(
                generate_mystery(seed=seed + attempt, config=config, max_retries=1)
            )
        for state in candidates:
            if expected_state_seed is not None and state.seed != expected_state_seed:
                continue
            if _state_matches_example(state, ex):
                return state, attempt
        actual = [_ground_truth_tuple(state) for state in candidates]
        expected = _expected_ground_truth_tuple(ex)
        raise AssertionError(
            f"{ex['id']}: recorded generation_attempt={attempt} produced "
            f"{actual}, expected {expected}"
        )

    state = generate_mystery(seed=seed, config=config)
    if _state_matches_example(state, ex):
        return state, max(0, state.seed - seed)

    for attempt in range(10):
        state = generate_mystery(seed=seed + attempt, config=config, max_retries=1)
        if _state_matches_example(state, ex):
            return state, attempt

    raise AssertionError(
        f"{ex['id']}: could not reconstruct recorded ground truth "
        f"{_expected_ground_truth_tuple(ex)} from seed {seed}"
    )


def _replay_example(ex: dict) -> dict:
    """Replay oracle_action_sequence against a fresh env; return result dict."""
    level  = _level_from_name(ex["complexity"])
    config = COMPLEXITY_PRESETS[level]
    state, generation_attempt = _state_for_example(ex, config)
    env    = MysteryEnvironment(state)
    # Recorded sequences are the oracle's exact proof; replay them with the
    # stochastic perception layer bypassed, exactly as OracleAgent does.
    env._perception_disabled = True
    _      = render_initial_briefing(env)  # initialise env (records start)

    failed_actions = []
    for step_rec in ex["oracle_action_sequence"]:
        action_name = step_rec["action"]
        kwargs      = step_rec["kwargs"]
        action      = AgentAction[action_name]
        actor_id    = step_rec.get("actor_id", "detective")
        if actor_id == "detective":
            result = env.step(action, **kwargs)
        else:
            result = env.step_for_actor(actor_id, action, **kwargs)
        if not result.success:
            failed_actions.append({
                "step": step_rec["step"],
                "action": action_name,
                "kwargs": kwargs,
                "observation": result.observation,
            })

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
        "state_seed":      state.seed,
        "generation_attempt": generation_attempt,
        "accusation_correct": correct,
        "failed_actions":  failed_actions,
        "replayed_scores": replayed_scores,
        "expected_scores": ex["oracle_scores"],
    }


def _check(result: dict) -> tuple[bool, list[str]]:
    """Return (passed, list_of_failures)."""
    failures: list[str] = []
    for failed in result["failed_actions"]:
        failures.append(
            f"step {failed['step']} {failed['action']} failed: "
            f"{failed['observation']}"
        )
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
