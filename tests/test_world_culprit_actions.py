from __future__ import annotations

import unittest
from dataclasses import replace

from mystery_world import COMPLEXITY_PRESETS, ComplexityLevel
from mystery_world.entities import EvidenceState, WorldObject
from mystery_world.generator import generate_mystery
from mystery_world.world import AgentAction, MysteryEnvironment


def _free_culprit_env() -> tuple[MysteryEnvironment, WorldObject]:
    config = replace(
        COMPLEXITY_PRESETS[ComplexityLevel.TRIVIAL],
        free_culprit_actions=True,
    )
    state = generate_mystery(config=config, seed=0)
    env = MysteryEnvironment(state)
    env.enable_free_culprit()

    for obj in state.objects.values():
        if (
            obj.evidence_id
            and obj.portable
            and obj.location_id in state.locations
            and state.evidence[obj.evidence_id].state != EvidenceState.HIDDEN
        ):
            return env, obj

    raise AssertionError("expected a visible portable evidence object")


class CulpritActionTests(unittest.TestCase):
    def test_culprit_cannot_take_evidence_bearing_object(self) -> None:
        env, obj = _free_culprit_env()
        state = env.state
        culprit = state.get_culprit()
        assert culprit is not None

        env._set_actor_location(culprit.id, obj.location_id)
        before_obj_location = obj.location_id
        before_evidence_location = state.evidence[obj.evidence_id].location_id
        before_objects_here = list(state.locations[obj.location_id].objects_here)

        result = env.step_for_actor(
            culprit.id,
            AgentAction.TAKE_OBJECT,
            object_name=obj.name,
            advance_world=False,
        )

        self.assertFalse(result.success)
        self.assertEqual(obj.location_id, before_obj_location)
        self.assertEqual(state.evidence[obj.evidence_id].location_id, before_evidence_location)
        self.assertEqual(state.locations[obj.location_id].objects_here, before_objects_here)
        self.assertNotIn(obj.id, culprit.inventory)

    def test_culprit_take_targets_exclude_evidence_bearing_objects(self) -> None:
        env, obj = _free_culprit_env()
        culprit = env.state.get_culprit()
        assert culprit is not None

        env._set_actor_location(culprit.id, obj.location_id)

        observation = env.observe_location(culprit.id)
        take_targets = observation.split("TAKE_OBJECT: ", 1)[1].split(".", 1)[0]

        self.assertNotIn(obj.name, take_targets)


if __name__ == "__main__":
    unittest.main()
