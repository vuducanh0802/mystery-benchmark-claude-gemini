from __future__ import annotations

import unittest

import numpy as np

from mystery_world import COMPLEXITY_PRESETS, DEFAULT_ASSET_POOL, ComplexityLevel
from mystery_world.entities import Character, CharacterRole
from mystery_world.generator import _generate_characters, generate_mystery


def _suspects(characters: dict[str, Character]) -> list[Character]:
    return [
        char for char in characters.values()
        if CharacterRole.SUSPECT in char.roles
    ]


def _culprit_suspect_index(characters: dict[str, Character]) -> int:
    suspects = _suspects(characters)
    culprit = next(char for char in suspects if char.is_culprit)
    return suspects.index(culprit)


class GeneratorCulpritTests(unittest.TestCase):
    def test_culprit_suspect_index_varies_across_generated_games(self) -> None:
        config = COMPLEXITY_PRESETS[ComplexityLevel.EASY]

        positions = {
            _culprit_suspect_index(generate_mystery(config=config, seed=seed).characters)
            for seed in range(12)
        }

        self.assertGreater(len(positions), 1)

    def test_ambiguity_traits_are_shared_for_randomized_culprit(self) -> None:
        config = COMPLEXITY_PRESETS[ComplexityLevel.MEDIUM]
        location_ids = ["loc_a", "loc_b", "loc_c", "loc_d", "loc_e"]

        for seed in range(12):
            characters = _generate_characters(
                config,
                DEFAULT_ASSET_POOL,
                np.random.default_rng(seed),
                location_ids,
            )
            suspects = _suspects(characters)
            culprit = next(char for char in suspects if char.is_culprit)
            others = [char for char in suspects if char.id != culprit.id]

            for trait_name in ("build", "hair", "hands"):
                culprit_value = getattr(culprit.physical_traits, trait_name)
                self.assertTrue(
                    any(
                        getattr(other.physical_traits, trait_name) == culprit_value
                        for other in others
                    ),
                    f"seed={seed} trait={trait_name}",
                )


if __name__ == "__main__":
    unittest.main()
