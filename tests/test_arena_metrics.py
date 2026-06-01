from __future__ import annotations

import unittest

from arena.aggregate import aggregate_matches
from arena.metrics import (
    CULPRIT_DEGRADATION_ALPHA,
    PAYOFF_SCHEMA,
    culprit_degradation_payoff,
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


class CulpritDegradationTests(unittest.TestCase):
    def test_none_without_baseline(self) -> None:
        self.assertIsNone(culprit_degradation_payoff(0.4, None))

    def test_scales_and_clamps(self) -> None:
        # raw drop 0.175 * alpha(2) = 0.35
        self.assertAlmostEqual(culprit_degradation_payoff(0.625, 0.8), 0.35)
        # negative drop (culprit "helped") clamps to 0
        self.assertEqual(culprit_degradation_payoff(0.9, 0.8), 0.0)
        # huge drop clamps to 1
        self.assertEqual(culprit_degradation_payoff(0.0, 0.9), 1.0)

    def test_alpha_is_two(self) -> None:
        self.assertEqual(CULPRIT_DEGRADATION_ALPHA, 2.0)


class CulpritLeaderboardTests(unittest.TestCase):
    """The leaderboard headline is baseline-relative degradation, not exposure."""

    def _matches(self) -> list[dict]:
        rows = []
        # Same reference detective on 3 cases, baseline payoff 0.8 vs passive.
        for seed in range(3):
            base = {"level": "MEDIUM", "seed": seed, "detective": {"name": "ref"}}
            rows.append({**base, "match_id": f"p{seed}", "culprit": {"name": "passive"},
                         "detective_payoff": 0.8, "culprit_payoff": 0.5})
            rows.append({**base, "match_id": f"a{seed}", "culprit": {"name": "amateur"},
                         "detective_payoff": 0.625, "culprit_payoff": 0.5})
            rows.append({**base, "match_id": f"s{seed}", "culprit": {"name": "skilled"},
                         "detective_payoff": 0.45, "culprit_payoff": 0.5})
        return rows

    def test_degradation_leaderboard(self) -> None:
        out = aggregate_matches(self._matches(), bootstrap_samples=50)
        by_name = {r["model"]: r for r in out["culprit_leaderboard"]}

        self.assertEqual(by_name["passive"]["mean_payoff"], 0.0)
        self.assertAlmostEqual(by_name["amateur"]["mean_payoff"], 0.35, places=4)
        self.assertAlmostEqual(by_name["skilled"]["mean_payoff"], 0.70, places=4)
        # Ranked skilled > amateur > passive.
        order = [r["model"] for r in out["culprit_leaderboard"]]
        self.assertLess(order.index("skilled"), order.index("amateur"))
        self.assertLess(order.index("amateur"), order.index("passive"))
        # Basis is recorded for transparency.
        self.assertEqual(by_name["amateur"]["payoff_basis"], "scaled_degradation_vs_passive")


if __name__ == "__main__":
    unittest.main()
