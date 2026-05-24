"""Aggregate Arena match JSONL into leaderboards and matrix files."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from arena.trueskill import compute_role_trueskill


def load_matches(arena_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(arena_dir) / "matches.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bootstrap_ci(values: list[float], *, samples: int = 1000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(mean(values[rng.randrange(n)] for _ in range(n)))
    means.sort()
    return (means[int(0.025 * samples)], means[int(0.975 * samples)])


def _score(match: dict[str, Any], key: str) -> float:
    score = match.get("score_result") or {}
    metrics = match.get("metrics") or {}
    return float(score.get(key, metrics.get(key, 0.0)) or 0.0)


def _passive_baseline(matches: list[dict[str, Any]]) -> dict[tuple[str, str, int], float]:
    baseline = {}
    for match in matches:
        if match.get("culprit", {}).get("name") != "passive":
            continue
        key = (
            match.get("detective", {}).get("name", ""),
            str(match.get("level", "")),
            int(match.get("seed", 0)),
        )
        baseline[key] = float(match.get("detective_payoff", 0.0))
    return baseline


def aggregate_matches(
    matches: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 1000,
    trueskill_mu: float = 25.0,
    trueskill_sigma: float = 25.0 / 3.0,
    trueskill_beta: float = 25.0 / 6.0,
    trueskill_tau: float = 25.0 / 300.0,
    trueskill_draw_threshold: float = 0.0,
) -> dict[str, Any]:
    ratings = compute_role_trueskill(
        matches,
        mu=trueskill_mu,
        sigma=trueskill_sigma,
        beta=trueskill_beta,
        tau=trueskill_tau,
        draw_threshold=trueskill_draw_threshold,
    )
    passive = _passive_baseline(matches)

    by_detective: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_culprit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        by_detective[match.get("detective", {}).get("name", "unknown")].append(match)
        by_culprit[match.get("culprit", {}).get("name", "unknown")].append(match)

    detective_rows = []
    for name, rows in by_detective.items():
        payoffs = [float(r.get("detective_payoff", 0.0)) for r in rows]
        ci_low, ci_high = _bootstrap_ci(payoffs, samples=bootstrap_samples, seed=17)
        detective_rows.append({
            "rank": 0,
            "model": name,
            "n": len(rows),
            "mean_payoff": round(_avg(payoffs), 4),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "trueskill": ratings["detective"].get(name, {
                "mu": round(trueskill_mu, 4),
                "sigma": round(trueskill_sigma, 4),
                "skill": round(trueskill_mu - 3.0 * trueskill_sigma, 4),
            }),
            "solve_rate": round(_avg([1.0 if r.get("solved") else 0.0 for r in rows]), 4),
            "accusation_accuracy": round(_avg([1.0 if r.get("accusation_correct") else 0.0 for r in rows]), 4),
            "triangle": round(_avg([_score(r, "triangle_score") / 3.0 for r in rows]), 4),
            "alibi": round(_avg([_score(r, "alibi_score") for r in rows]), 4),
            "elimination": round(_avg([_score(r, "elimination_score") for r in rows]), 4),
            "avg_actions": round(_avg([float(r.get("actions_taken", 0) or 0) for r in rows]), 2),
            "failed_action_rate": round(
                _avg([
                    float(r.get("detective_failed_actions", 0) or 0)
                    / max(1.0, float(r.get("actions_taken", 0) or 0))
                    for r in rows
                ]),
                4,
            ),
            "guard_blocked": round(_avg([float(r.get("guard_blocked_actions", 0) or 0) for r in rows]), 3),
        })
    detective_rows.sort(
        key=lambda r: (r["mean_payoff"], r["solve_rate"], r["trueskill"]["skill"]),
        reverse=True,
    )
    for i, row in enumerate(detective_rows, 1):
        row["rank"] = i

    culprit_rows = []
    for name, rows in by_culprit.items():
        payoffs = [float(r.get("culprit_payoff", 0.0)) for r in rows]
        ci_low, ci_high = _bootstrap_ci(payoffs, samples=bootstrap_samples, seed=23)
        drops = []
        for r in rows:
            key = (
                r.get("detective", {}).get("name", ""),
                str(r.get("level", "")),
                int(r.get("seed", 0)),
            )
            if key in passive and name != "passive":
                drops.append(passive[key] - float(r.get("detective_payoff", 0.0)))
        culprit_rows.append({
            "rank": 0,
            "model": name,
            "n": len(rows),
            "mean_payoff": round(_avg(payoffs), 4),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "trueskill": ratings["culprit"].get(name, {
                "mu": round(trueskill_mu, 4),
                "sigma": round(trueskill_sigma, 4),
                "skill": round(trueskill_mu - 3.0 * trueskill_sigma, 4),
            }),
            "detective_failure_rate": round(_avg([0.0 if r.get("solved") else 1.0 for r in rows]), 4),
            "score_drop_vs_passive": round(_avg(drops), 4) if drops else None,
            "avg_detective_actions": round(_avg([float(r.get("actions_taken", 0) or 0) for r in rows]), 2),
            "avg_culprit_actions": round(_avg([float(r.get("culprit_actions_taken", 0) or 0) for r in rows]), 2),
            "culprit_failed_action_rate": round(
                _avg([
                    float(r.get("culprit_failed_actions", 0) or 0)
                    / max(1.0, float(r.get("culprit_actions_taken", 0) or 0))
                    for r in rows
                ]),
                4,
            ),
            "guard_blocked": round(_avg([float(r.get("guard_blocked_actions", 0) or 0) for r in rows]), 3),
        })
    culprit_rows.sort(
        key=lambda r: (r["mean_payoff"], r["detective_failure_rate"], r["trueskill"]["skill"]),
        reverse=True,
    )
    for i, row in enumerate(culprit_rows, 1):
        row["rank"] = i

    matrix: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for match in matches:
        d = match.get("detective", {}).get("name", "unknown")
        c = match.get("culprit", {}).get("name", "unknown")
        grouped[(d, c)].append(float(match.get("detective_payoff", 0.0)))
    for (detective, culprit), values in grouped.items():
        matrix[detective][culprit] = {
            "detective_payoff": round(_avg(values), 4),
            "culprit_payoff": round(1.0 - _avg(values), 4),
            "n": len(values),
        }

    return {
        "detective_leaderboard": detective_rows,
        "culprit_leaderboard": culprit_rows,
        "ratings": ratings,
        "matrix": {k: dict(v) for k, v in sorted(matrix.items())},
        "summary": {
            "matches": len(matches),
            "detectives": len(by_detective),
            "culprits": len(by_culprit),
        },
    }


def write_outputs(
    arena_dir: str | Path,
    *,
    bootstrap_samples: int = 1000,
    trueskill_mu: float = 25.0,
    trueskill_sigma: float = 25.0 / 3.0,
    trueskill_beta: float = 25.0 / 6.0,
    trueskill_tau: float = 25.0 / 300.0,
    trueskill_draw_threshold: float = 0.0,
) -> dict[str, Any]:
    arena_dir = Path(arena_dir)
    matches = load_matches(arena_dir)
    outputs = aggregate_matches(
        matches,
        bootstrap_samples=bootstrap_samples,
        trueskill_mu=trueskill_mu,
        trueskill_sigma=trueskill_sigma,
        trueskill_beta=trueskill_beta,
        trueskill_tau=trueskill_tau,
        trueskill_draw_threshold=trueskill_draw_threshold,
    )
    (arena_dir / "detective_leaderboard.json").write_text(
        json.dumps(outputs["detective_leaderboard"], indent=2),
        encoding="utf-8",
    )
    (arena_dir / "culprit_leaderboard.json").write_text(
        json.dumps(outputs["culprit_leaderboard"], indent=2),
        encoding="utf-8",
    )
    (arena_dir / "role_ratings.json").write_text(
        json.dumps(outputs["ratings"], indent=2),
        encoding="utf-8",
    )
    (arena_dir / "duel_matrix.json").write_text(
        json.dumps(outputs["matrix"], indent=2),
        encoding="utf-8",
    )
    return outputs
