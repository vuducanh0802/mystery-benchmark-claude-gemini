from __future__ import annotations

import unittest

from arena.metrics import (
    PAYOFF_SCHEMA,
    culprit_payoff,
    detective_payoff,
    recompute_match_payoffs,
)
from arena.trueskill import compute_role_trueskill


class PayoffTests(unittest.TestCase):
    def test_culprit_payoff_is_not_composite_complement_when_caught(self) -> None:
        summary = {
            "accusation_correct": True,
            "score_result": {
                "correct_suspect": True,
                "correct_weapon": True,
                "correct_room": True,
                "accusation_score": 1.0,
                "composite_score": 0.35,
            },
        }

        self.assertEqual(detective_payoff(summary), 0.35)
        self.assertEqual(culprit_payoff(summary), 0.0)

    def test_culprit_payoff_prioritizes_escaping_identification(self) -> None:
        summary = {
            "score_result": {
                "correct_suspect": False,
                "correct_weapon": True,
                "correct_room": True,
                "accusation_score": 2 / 3,
                "composite_score": 0.7,
            },
        }

        self.assertAlmostEqual(culprit_payoff(summary), 0.7)

    def test_culprit_payoff_falls_back_to_legacy_solved_flag(self) -> None:
        self.assertEqual(culprit_payoff({"accusation_correct": True}), 0.0)
        self.assertEqual(culprit_payoff({"accusation_correct": False}), 1.0)

    def test_trueskill_compares_role_specific_payoffs(self) -> None:
        ratings = compute_role_trueskill([
            {
                "level": "TRIVIAL",
                "seed": 0,
                "match_id": "m0",
                "detective": {"name": "d"},
                "culprit": {"name": "c"},
                "detective_payoff": 0.35,
                "culprit_payoff": 0.0,
            }
        ])

        self.assertGreater(
            ratings["detective"]["d"]["mu"],
            ratings["culprit"]["c"]["mu"],
        )

    def test_recompute_match_payoffs_migrates_old_complement_records(self) -> None:
        match = {
            "detective_payoff": 0.35,
            "culprit_payoff": 0.65,
            "accusation_correct": True,
            "score_result": {
                "correct_suspect": True,
                "correct_weapon": True,
                "correct_room": True,
                "composite_score": 0.35,
            },
        }

        normalized = recompute_match_payoffs(match)

        self.assertEqual(normalized["payoff_schema"], PAYOFF_SCHEMA)
        self.assertEqual(normalized["detective_payoff"], 0.35)
        self.assertEqual(normalized["culprit_payoff"], 0.0)


if __name__ == "__main__":
    unittest.main()
