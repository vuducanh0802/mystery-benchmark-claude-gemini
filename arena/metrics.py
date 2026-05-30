"""Arena match records and payoff extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arena.roster import ModelSpec


PAYOFF_SCHEMA = "detective_composite_v1_culprit_exposure_v1"


def _clamp01(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = 0.0
    return max(0.0, min(1.0, x))


def _score_result(summary: dict[str, Any], metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    score = summary.get("score_result") or {}
    if score:
        return score
    metrics = metrics or {}
    return {
        "composite_score": metrics.get("composite_score", 0.0),
        "triangle_score": metrics.get("triangle_score", 0.0),
        "alibi_score": metrics.get("alibi_score", 0.0),
        "elimination_score": metrics.get("elimination_score", 0.0),
        "accusation_score": metrics.get("accusation_score", metrics.get("partial_score", 0.0)),
    }


def detective_payoff(summary: dict[str, Any], metrics: dict[str, Any] | None = None) -> float:
    """Primary Arena detective payoff in [0, 1]."""
    score = _score_result(summary, metrics)
    if "composite_score" in score:
        return _clamp01(score.get("composite_score"))
    if metrics and "composite_score" in metrics:
        return _clamp01(metrics.get("composite_score"))
    return 1.0 if summary.get("accusation_correct") else 0.0


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
    return None


def _bool_from(source: dict[str, Any], key: str) -> bool | None:
    if key not in source:
        return None
    return _as_bool(source.get(key))


def _first_bool(*values: bool | None) -> bool | None:
    for value in values:
        if value is not None:
            return value
    return None


def culprit_payoff(summary: dict[str, Any], metrics: dict[str, Any] | None = None) -> float:
    """Primary Arena culprit payoff in [0, 1].

    Detective payoff rewards full case quality, including evidence citation and
    elimination subscores. The culprit payoff is intentionally not its complement:
    the culprit's primary objective is escaping identification in the final
    accusation.
    """
    raw_score = summary.get("score_result") or {}
    score = _score_result(summary, metrics)
    metrics = metrics or {}

    suspect = _first_bool(
        _bool_from(score, "correct_suspect"),
        _bool_from(metrics, "suspect_correct"),
    )
    weapon = _first_bool(
        _bool_from(score, "correct_weapon"),
        _bool_from(metrics, "weapon_correct"),
    )
    room = _first_bool(
        _bool_from(score, "correct_room"),
        _bool_from(metrics, "location_correct"),
    )

    if suspect is None and weapon is None and room is None:
        has_accusation_score = (
            "accusation_score" in raw_score
            or "accusation_score" in metrics
            or "partial_score" in metrics
        )
        if has_accusation_score:
            return 1.0 - _clamp01(score.get("accusation_score"))
        accusation_correct = _as_bool(summary.get("accusation_correct"))
        if accusation_correct is not None:
            return 0.0 if accusation_correct else 1.0
        return 1.0

    # Identifying the culprit matters most; weapon and room correctness are
    # secondary case-closure signals.
    exposure = (
        0.70 * float(bool(suspect))
        + 0.15 * float(bool(weapon))
        + 0.15 * float(bool(room))
    )
    return 1.0 - _clamp01(exposure)


# Scaling for the baseline-relative culprit leaderboard score. Raw degradation
# (passive detective payoff minus this culprit's) is typically small; alpha maps
# a "tries but mediocre" culprit to ~0.3-0.4 with headroom above. Tune once real
# culprit runs land.
CULPRIT_DEGRADATION_ALPHA = 2.0


def culprit_degradation_payoff(
    detective_payoff_value: float,
    passive_detective_payoff: float | None,
    *,
    alpha: float = CULPRIT_DEGRADATION_ALPHA,
) -> float | None:
    """Baseline-relative culprit skill in [0, 1].

    Measures how far this culprit drove the detective's payoff *below* the
    passive-culprit baseline on the same (detective, level, seed) case, scaled
    by ``alpha`` and clamped. The passive culprit is the baseline, so it scores
    0 by construction — this rewards demonstrated interference, not merely a
    weak opposing detective (the flaw of the raw exposure payoff).

    Returns None when no passive baseline exists for the case (unmeasurable).
    """
    if passive_detective_payoff is None:
        return None
    drop = float(passive_detective_payoff) - float(detective_payoff_value)
    return max(0.0, min(1.0, alpha * drop))


def recompute_match_payoffs(match: dict[str, Any]) -> dict[str, Any]:
    """Return a match copy with current payoff semantics applied."""
    normalized = dict(match)
    if normalized.get("payoff_schema") == PAYOFF_SCHEMA:
        return normalized

    summary = {
        "score_result": normalized.get("score_result") or {},
        "accusation_correct": normalized.get("accusation_correct"),
    }
    metrics = normalized.get("metrics") or {}
    error = normalized.get("error")
    d_payoff = 0.0 if error else detective_payoff(summary, metrics)
    c_payoff = 0.0 if error else culprit_payoff(summary, metrics)
    normalized["detective_payoff"] = round(d_payoff, 6)
    normalized["culprit_payoff"] = round(c_payoff, 6)
    normalized["payoff_schema"] = PAYOFF_SCHEMA
    return normalized


def _failed_counts(action_trace: list[dict[str, Any]]) -> tuple[int, int]:
    detective_failed = 0
    culprit_failed = 0
    for rec in action_trace:
        if rec.get("success", True):
            continue
        role = rec.get("role", "detective")
        if role == "culprit":
            culprit_failed += 1
        else:
            detective_failed += 1
    return detective_failed, culprit_failed


def match_from_episode(
    *,
    run_id: str,
    match_id: str,
    level: str,
    seed: int,
    detective: ModelSpec,
    culprit: ModelSpec,
    npc: dict[str, Any],
    result,
    trajectory_path: str | Path,
) -> dict[str, Any]:
    summary = result.episode_summary or {}
    metrics = result.metrics.to_dict() if result.metrics else {}
    score = _score_result(summary, metrics)
    d_payoff = 0.0 if result.error else detective_payoff(summary, metrics)
    c_payoff = 0.0 if result.error else culprit_payoff(summary, metrics)
    detective_failed, culprit_failed = _failed_counts(result.action_trace)
    culprit_actions = sum(1 for rec in result.action_trace if rec.get("role") == "culprit")
    return {
        "run_id": run_id,
        "match_id": match_id,
        "level": level,
        "seed": seed,
        "detective": detective.to_dict(),
        "culprit": culprit.to_dict(),
        "npc": dict(npc),
        "payoff_schema": PAYOFF_SCHEMA,
        "detective_payoff": round(d_payoff, 6),
        "culprit_payoff": round(c_payoff, 6),
        "solved": bool(metrics.get("solved", summary.get("accusation_correct", False))),
        "accusation_correct": bool(summary.get("accusation_correct", metrics.get("solved", False))),
        "score_result": score,
        "metrics": metrics,
        "actions_taken": summary.get("actions_taken", metrics.get("actions_used", 0)),
        "culprit_actions_taken": culprit_actions,
        "detective_failed_actions": detective_failed,
        "culprit_failed_actions": culprit_failed,
        "guard_blocked_actions": summary.get("solvability_guard_blocked_actions", 0),
        "guard_suppressed_events": summary.get("solvability_guard_suppressed_events", 0),
        "elapsed_seconds": result.elapsed_seconds,
        "error": result.error,
        "trajectory_path": str(trajectory_path),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def match_from_trajectory(
    path: str | Path,
    *,
    run_id: str,
    match_id: str,
    detective: ModelSpec,
    culprit: ModelSpec,
    npc: dict[str, Any],
) -> dict[str, Any]:
    recs = read_jsonl(path)
    header = next((r for r in recs if r.get("kind") == "header"), {})
    footer = next((r for r in reversed(recs) if r.get("kind") == "footer"), {})
    steps = [r for r in recs if r.get("kind") == "step"]
    summary = footer.get("episode_summary") or {}
    metrics = footer.get("metrics") or {}
    score = _score_result(summary, metrics)
    error = footer.get("error")
    d_payoff = 0.0 if error else detective_payoff(summary, metrics)
    c_payoff = 0.0 if error else culprit_payoff(summary, metrics)
    detective_failed, culprit_failed = _failed_counts(steps)
    return {
        "run_id": header.get("arena_run_id", run_id),
        "match_id": header.get("arena_match_id", match_id),
        "level": header.get("level", ""),
        "seed": header.get("seed", 0),
        "detective": detective.to_dict(),
        "culprit": culprit.to_dict(),
        "npc": dict(npc),
        "payoff_schema": PAYOFF_SCHEMA,
        "detective_payoff": round(d_payoff, 6),
        "culprit_payoff": round(c_payoff, 6),
        "solved": bool(metrics.get("solved", summary.get("accusation_correct", False))),
        "accusation_correct": bool(summary.get("accusation_correct", metrics.get("solved", False))),
        "score_result": score,
        "metrics": metrics,
        "actions_taken": summary.get("actions_taken", metrics.get("actions_used", 0)),
        "culprit_actions_taken": sum(1 for r in steps if r.get("role") == "culprit"),
        "detective_failed_actions": detective_failed,
        "culprit_failed_actions": culprit_failed,
        "guard_blocked_actions": summary.get("solvability_guard_blocked_actions", 0),
        "guard_suppressed_events": summary.get("solvability_guard_suppressed_events", 0),
        "elapsed_seconds": footer.get("elapsed_seconds", 0.0),
        "error": error,
        "trajectory_path": str(path),
    }
