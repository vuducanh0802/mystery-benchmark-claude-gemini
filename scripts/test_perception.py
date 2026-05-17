"""Smoke + invariant tests for the stochastic-discovery (perception) layer.

Run: uv run scripts/test_perception.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mystery_world import COMPLEXITY_PRESETS, ComplexityLevel
from mystery_world.entities import EvidenceState
from mystery_world.generator import generate_mystery
from mystery_world.world import MysteryEnvironment, _perception_roll


def _find_evidence_object(state):
    """Return (location_id, object) for the first usable evidence-bearing object."""
    for loc in state.locations.values():
        for oid in loc.objects_here:
            obj = state.objects.get(oid)
            if obj and obj.evidence_id:
                ev = state.evidence.get(obj.evidence_id)
                if ev and ev.state not in (EvidenceState.HIDDEN, EvidenceState.DESTROYED):
                    return loc.id, obj, ev
    return None, None, None


def test_keyed_roll_is_pure():
    a = _perception_roll(42, "abc123", 0)
    b = _perception_roll(42, "abc123", 0)
    c = _perception_roll(42, "abc123", 1)
    d = _perception_roll(43, "abc123", 0)
    assert a == b, "same key must give same roll"
    assert a != c and a != d, "different key must (almost surely) differ"
    print(f"[ok] keyed roll pure & key-sensitive ({a:.4f})")


def _run_examines(seed: int, level: ComplexityLevel, n: int):
    cfg = COMPLEXITY_PRESETS[level]
    state = generate_mystery(cfg, seed)
    env = MysteryEnvironment(state)
    loc_id, obj, ev = _find_evidence_object(state)
    assert obj is not None, "no evidence object found"
    env.agent_location_id = loc_id
    from mystery_world.world import AgentAction
    outcomes = []
    for _ in range(n):
        r = env.step(AgentAction.EXAMINE_OBJECT, object_name=obj.name)
        outcomes.append(bool(r.evidence_found))
    return outcomes, env, ev.id


def test_reproducible_across_runs():
    o1, _, _ = _run_examines(7, ComplexityLevel.HARD, 6)
    o2, _, _ = _run_examines(7, ComplexityLevel.HARD, 6)
    assert o1 == o2, f"non-reproducible: {o1} vs {o2}"
    print(f"[ok] reproducible miss/hit sequence: {o1}")


def test_decay_eventually_finds():
    # HARD has detective_miss_base=0.20; decay 0.5 → miss prob collapses fast.
    o, env, eid = _run_examines(7, ComplexityLevel.HARD, 8)
    assert any(o), f"never found evidence in 8 tries (decay broken?): {o}"
    s = env.get_episode_summary()
    assert s["examine_present"] >= 1
    print(f"[ok] decay-retry converges; misses={len(s['perception_misses'])}, seq={o}")


def test_perception_disabled_is_deterministic():
    cfg = COMPLEXITY_PRESETS[ComplexityLevel.EXPERT]
    state = generate_mystery(cfg, 3)
    env = MysteryEnvironment(state)
    env._perception_disabled = True
    loc_id, obj, ev = _find_evidence_object(state)
    env.agent_location_id = loc_id
    from mystery_world.world import AgentAction
    r = env.step(AgentAction.EXAMINE_OBJECT, object_name=obj.name)
    assert r.evidence_found == [ev.id], "perception_disabled must always reveal"
    print("[ok] perception_disabled → guaranteed discovery (oracle invariant)")


def test_concealment_preserves_solvability():
    from benchmark.verify import check_solvability
    cfg = COMPLEXITY_PRESETS[ComplexityLevel.EXPERT]  # culprit_conceal_prob=0.45
    state = generate_mystery(cfg, 11)
    concealed = [
        e for e in state.evidence.values()
        if e.concealment_prob > 0.0
    ]
    # at least usable() must hold for concealed evidence (state not flipped)
    for e in concealed:
        assert e.is_usable(), "concealment must not flip evidence to HIDDEN"
        assert e.discovery_difficulty <= 1.0
    assert check_solvability(state)["solvable"], "concealment broke solvability"
    print(f"[ok] concealment preserves solvability; {len(concealed)} evidence pre-concealed")


def test_miss_is_indistinguishable_and_recoverable():
    """Force a real perceptual miss; the miss observation must be byte-identical
    to examining the same object when it carries no evidence, and a retry must
    still be able to recover it (decay path)."""
    from mystery_world.world import AgentAction, _perception_roll

    cfg = COMPLEXITY_PRESETS[ComplexityLevel.EXPERT]  # miss_base 0.30
    found_miss = False
    for seed in range(60):
        state = generate_mystery(cfg, seed)
        loc_id, obj, ev = _find_evidence_object(state)
        if obj is None:
            continue
        miss_p0 = cfg.detective_miss_base * ev.discovery_difficulty
        if _perception_roll(state.seed, ev.id, 0) >= miss_p0:
            continue  # this seed/object would hit on first try — keep scanning

        env = MysteryEnvironment(state)
        env.agent_location_id = loc_id
        r_miss = env.step(AgentAction.EXAMINE_OBJECT, object_name=obj.name)
        assert not r_miss.evidence_found, "expected a miss"
        # Byte-identical to a no-evidence examine of the same object:
        expected = f"You examine the {obj.name}. {obj.description}"
        assert r_miss.observation == expected, (
            f"miss leaks information:\n{r_miss.observation!r}\nvs\n{expected!r}"
        )
        s = env.get_episode_summary()
        assert s["examine_present"] == 1 and s["examine_hit"] == 0
        assert len(s["perception_misses"]) == 1
        # Keep retrying — decay must let it be recovered within the budget.
        recovered = any(
            env.step(AgentAction.EXAMINE_OBJECT, object_name=obj.name).evidence_found
            for _ in range(10)
        )
        assert recovered, "decay-retry never recovered a missed clue"
        print(f"[ok] forced miss @seed={seed} indistinguishable + recovered")
        found_miss = True
        break
    assert found_miss, "could not construct a miss in 60 seeds (miss path untested!)"


def test_serialization_roundtrip():
    state = generate_mystery(COMPLEXITY_PRESETS[ComplexityLevel.MEDIUM], 5)
    from mystery_world.world import WorldState
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "w.json")
        state.save(p)
        r = WorldState.load(p)
    for k, ev in state.evidence.items():
        assert r.evidence[k].concealment_prob == ev.concealment_prob
        assert r.evidence[k].examine_attempts == ev.examine_attempts
    assert r.config.detective_miss_base == state.config.detective_miss_base
    print("[ok] new fields round-trip through save/load")


if __name__ == "__main__":
    test_keyed_roll_is_pure()
    test_reproducible_across_runs()
    test_decay_eventually_finds()
    test_perception_disabled_is_deterministic()
    test_concealment_preserves_solvability()
    test_miss_is_indistinguishable_and_recoverable()
    test_serialization_roundtrip()
    print("\nAll perception-layer tests passed.")
