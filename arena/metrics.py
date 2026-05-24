"""Arena match records and payoff extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arena.roster import ModelSpec


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
    c_payoff = 1.0 - d_payoff
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
    detective_failed, culprit_failed = _failed_counts(steps)
    return {
        "run_id": header.get("arena_run_id", run_id),
        "match_id": header.get("arena_match_id", match_id),
        "level": header.get("level", ""),
        "seed": header.get("seed", 0),
        "detective": detective.to_dict(),
        "culprit": culprit.to_dict(),
        "npc": dict(npc),
        "detective_payoff": round(d_payoff, 6),
        "culprit_payoff": round(1.0 - d_payoff, 6),
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
