from __future__ import annotations

import unittest
from dataclasses import replace

from mystery_world import (
    COMPLEXITY_PRESETS,
    ComplexityLevel,
    classify_weapon,
    wound_for_weapon,
)
from mystery_world.entities import EdgeType, EvidenceState, WorldObject
from mystery_world.generator import (
    anchored_triangle_coverage,
    generate_mystery,
    murder_weapon_class_unique,
    verify_solvability,
)
from mystery_world.narrator import render_initial_briefing
from mystery_world.world import AgentAction, MysteryEnvironment
from agents.oracle_agent import OracleAgent


def _free_culprit_env(seed: int = 0, level: ComplexityLevel = ComplexityLevel.TRIVIAL):
    config = replace(COMPLEXITY_PRESETS[level], free_culprit_actions=True)
    state = generate_mystery(config=config, seed=seed)
    env = MysteryEnvironment(state)
    env.enable_free_culprit()
    return env


def _first_portable_evidence_object(state) -> WorldObject:
    for obj in state.objects.values():
        if (
            obj.evidence_id
            and obj.portable
            and obj.location_id in state.locations
            and state.evidence[obj.evidence_id].state != EvidenceState.HIDDEN
        ):
            return obj
    raise AssertionError("expected a visible portable evidence object")


def _portable_evidence_objects(state) -> list[WorldObject]:
    return [
        obj for obj in list(state.objects.values())
        if obj.evidence_id and obj.portable and obj.location_id in state.locations
    ]


def _run_oracle(env) -> None:
    """Drive a max-score oracle detective to its accusation."""
    oracle = OracleAgent(mode="max_score")
    oracle.initialize(env, render_initial_briefing(env))
    for _ in range(env.state.config.max_agent_actions + 5):
        if env.is_solved:
            break
        obs = env.observe_location()
        action, kwargs = oracle.decide_action(obs)
        env.step(action, **kwargs)
        if action == AgentAction.ACCUSE:
            break


class CulpritTakeTests(unittest.TestCase):
    """The culprit may now hide evidence; solvability is protected elsewhere."""

    def test_culprit_can_take_evidence_bearing_object(self) -> None:
        env = _free_culprit_env()
        state = env.state
        obj = _first_portable_evidence_object(state)
        culprit = state.get_culprit()
        assert culprit is not None

        env._set_actor_location(culprit.id, obj.location_id)
        source_loc = obj.location_id

        result = env.step_for_actor(
            culprit.id, AgentAction.TAKE_OBJECT,
            object_name=obj.name, advance_world=False,
        )

        self.assertTrue(result.success)
        self.assertIn(obj.id, culprit.inventory)
        self.assertNotIn(obj.id, state.locations[source_loc].objects_here)
        self.assertEqual(obj.location_id, f"inventory:{culprit.id}")

    def test_take_targets_include_evidence_objects(self) -> None:
        env = _free_culprit_env()
        state = env.state
        obj = _first_portable_evidence_object(state)
        culprit = state.get_culprit()
        assert culprit is not None
        env._set_actor_location(culprit.id, obj.location_id)

        observation = env.observe_location(culprit.id)
        take_targets = observation.split("TAKE_OBJECT: ", 1)[1].split("\n", 1)[0]
        self.assertIn(obj.name, take_targets)

    def test_taking_evidence_stamps_anchored_tamper_trace(self) -> None:
        env = _free_culprit_env()
        state = env.state
        obj = _first_portable_evidence_object(state)
        culprit = state.get_culprit()
        assert culprit is not None
        env._set_actor_location(culprit.id, obj.location_id)
        source_loc = obj.location_id

        ev_before = set(state.evidence)
        env.step_for_actor(
            culprit.id, AgentAction.TAKE_OBJECT,
            object_name=obj.name, advance_world=False,
        )
        new_ev = [state.evidence[e] for e in state.evidence if e not in ev_before]

        self.assertTrue(new_ev, "taking evidence should stamp a tamper trace")
        trace = new_ev[0]
        self.assertTrue(trace.anchored)
        self.assertEqual(trace.linked_character_id, culprit.id)
        self.assertIsNotNone(trace.relevance)
        self.assertEqual(trace.relevance.edge_type, EdgeType.SUSPECT_ROOM)
        # Discoverable: hosted on a non-portable scene feature in the room.
        host = next(
            (o for o in state.objects.values() if o.evidence_id == trace.id), None
        )
        self.assertIsNotNone(host)
        self.assertFalse(host.portable)
        self.assertIn(host.id, state.locations[source_loc].objects_here)


class AnchoredFloorTests(unittest.TestCase):
    """The Locard triangle must stay closable from non-portable evidence."""

    def test_anchored_invariant_holds_across_seeds(self) -> None:
        for level in ComplexityLevel:
            cfg = COMPLEXITY_PRESETS[level]
            for seed in range(15):
                st = generate_mystery(config=cfg, seed=seed)
                cov = anchored_triangle_coverage(st)
                with self.subTest(level=level.name, seed=seed):
                    self.assertTrue(
                        all(c > 0 for c in cov.values()),
                        f"anchored triangle not closed: "
                        f"{ {e.name: c for e, c in cov.items()} }",
                    )
                    self.assertTrue(murder_weapon_class_unique(st))

    def test_wound_matches_weapon_class(self) -> None:
        for seed in range(15):
            st = generate_mystery(
                config=COMPLEXITY_PRESETS[ComplexityLevel.MEDIUM], seed=seed
            )
            weapon = st.objects.get(st.murder_weapon_id)
            body = next(
                (o for o in st.objects.values() if o.name.startswith("body of")), None
            )
            self.assertIsNotNone(body)
            with self.subTest(seed=seed):
                self.assertIn(wound_for_weapon(weapon.name), body.description)
                self.assertEqual(weapon.weapon_class, classify_weapon(weapon.name))

    def test_floor_survives_total_tampering(self) -> None:
        """Culprit hides EVERY portable evidence object — case stays solvable."""
        env = _free_culprit_env(seed=1, level=ComplexityLevel.HARD)
        state = env.state
        culprit = state.get_culprit()
        assert culprit is not None

        before = anchored_triangle_coverage(state)
        taken = 0
        for obj in _portable_evidence_objects(state):
            env._set_actor_location(culprit.id, obj.location_id)
            r = env.step_for_actor(
                culprit.id, AgentAction.TAKE_OBJECT,
                object_name=obj.name, advance_world=False,
            )
            if r.success:
                taken += 1

        self.assertGreaterEqual(taken, 1)
        after = anchored_triangle_coverage(state)
        # Every triangle edge still has anchored proof the culprit couldn't take.
        for edge, count in after.items():
            self.assertGreater(count, 0, f"{edge.name} floor erased by tampering")
            self.assertGreaterEqual(count, before[edge])

    def test_oracle_closes_triangle_after_tampering(self) -> None:
        """End-to-end: a perfect detective still proves all three edges after
        the culprit has hidden every portable clue."""
        env = _free_culprit_env(seed=2, level=ComplexityLevel.MEDIUM)
        state = env.state
        culprit = state.get_culprit()
        assert culprit is not None
        for obj in _portable_evidence_objects(state):
            env._set_actor_location(culprit.id, obj.location_id)
            env.step_for_actor(
                culprit.id, AgentAction.TAKE_OBJECT,
                object_name=obj.name, advance_world=False,
            )

        _run_oracle(env)
        summary = env.get_episode_summary()
        score = summary.get("score_result") or {}
        self.assertTrue(summary.get("accusation_correct"))
        for key in ("suspect_weapon_score", "weapon_victim_score", "suspect_room_score"):
            self.assertGreater(
                score.get(key, 0.0), 0.0,
                f"oracle failed to prove {key} from the anchored floor",
            )


if __name__ == "__main__":
    unittest.main()
