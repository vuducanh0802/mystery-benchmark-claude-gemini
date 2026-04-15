"""
Evaluation metrics for the mystery benchmark

Metrics:
    1. **Solve rate** - fraction of instances correctly solved (full + partial)
    2. **Belief accuracy** - KL-divergence / accuracy of beliefs vs ground truth at each step
    3. **Clue efficiency** - fraction of relevant (non-red-herring) evidence discovered
    4. **Token cost** - total LLM tokens consumed per instance
    5. **Action efficiency** - actions used / budget
    6. **Partial credit** - how many of (suspect, weapon, location) were correct

All metrics are computed per-instance and can be aggregated across complexity levels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpisodeMetrics:
    """Metrics for a single benchmark episode."""
    instance_id: str = ""
    seed: int = 0
    complexity_level: int = 1

    # --- Solve rate ---
    solved: bool = False
    suspect_correct: bool = False
    weapon_correct: bool = False
    location_correct: bool = False
    partial_score: float = 0.0      # fraction of (suspect, weapon, location) correct

    # Locard triangle (per-edge precision, recall, F1)
    suspect_weapon_precision: float = 0.0
    suspect_weapon_recall: float = 0.0
    suspect_weapon_score: float = 0.0
    weapon_victim_precision: float = 0.0
    weapon_victim_recall: float = 0.0
    weapon_victim_score: float = 0.0
    suspect_room_precision: float = 0.0
    suspect_room_recall: float = 0.0
    suspect_room_score: float = 0.0
    triangle_score: float = 0.0

    # Alibi
    alibi_score: float = 0.0

    # Elimination
    correct_eliminations: int = 0
    incorrect_eliminations: int = 0
    elimination_score: float = 0.0

    # Accusation + alibi (copied from score_result for reporting)
    accusation_score: float = 0.0
    alibi_cited: bool = False
    contradiction_found: bool = False
    contradiction_valid: bool = False

    # Composite
    composite_score: float = 0.0

    # Action efficiency
    examine_total: int = 0
    examine_hit: int = 0
    examine_efficiency: float = 0.0

    # --- Belief accuracy ---
    # Tracked at each step: was the top belief the ground truth?
    belief_accuracy_trace: list[float] = field(default_factory=list)
    final_belief_accuracy: float = 0.0

    # --- Clue efficiency ---
    total_relevant_evidence: int = 0
    evidence_discovered: int = 0
    clue_efficiency: float = 0.0     # discovered / total relevant

    # --- Token cost ---
    total_tokens: int = 0
    tokens_per_action: float = 0.0

    # --- Action efficiency ---
    actions_used: int = 0
    action_budget: int = 0
    action_efficiency: float = 0.0   # actions_used / budget

    # --- Timing ---
    total_steps: int = 0
    event_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "seed": self.seed,
            "complexity_level": self.complexity_level,
            "solved": self.solved,
            "suspect_correct": self.suspect_correct,
            "weapon_correct": self.weapon_correct,
            "location_correct": self.location_correct,
            "partial_score": self.partial_score,
            "belief_accuracy_trace": self.belief_accuracy_trace,
            "final_belief_accuracy": self.final_belief_accuracy,
            "total_relevant_evidence": self.total_relevant_evidence,
            "evidence_discovered": self.evidence_discovered,
            "clue_efficiency": self.clue_efficiency,
            "total_tokens": self.total_tokens,
            "tokens_per_action": self.tokens_per_action,
            "actions_used": self.actions_used,
            "action_budget": self.action_budget,
            "action_efficiency": self.action_efficiency,
            "total_steps": self.total_steps,
            "event_count": self.event_count,
            "accusation_score": self.accusation_score,
            "triangle_score": self.triangle_score,
            "suspect_weapon_score": self.suspect_weapon_score,
            "weapon_victim_score": self.weapon_victim_score,
            "suspect_room_score": self.suspect_room_score,
            "alibi_cited": self.alibi_cited,
            "contradiction_found": self.contradiction_found,
            "contradiction_valid": self.contradiction_valid,
            "alibi_score": self.alibi_score,
            "composite_score": self.composite_score,
        }


def compute_episode_metrics(
    episode_summary: dict[str, Any],
    belief_snapshots: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    total_tokens: int,
    complexity_level: int = 1,
) -> EpisodeMetrics:
    """
    Compute all metrics for a completed episode.

    Parameters
    ----------
    episode_summary: dict
        Output of ``MysteryEnvironment.get_episode_summary()``
    belief_snapshots: list[dict]
        Agent's belief state at each step.
    ground_truth: dict
        ``{"culprit_name": str, "weapon_name": str, "location_name": str}``.
    total_tokens: int
        Total LLM tokens consumed.
    complexity_level: int
        Complexity level (1-5).
    """
    m = EpisodeMetrics()
    m.instance_id = f"seed_{episode_summary.get('seed', 0)}"
    m.seed = episode_summary.get("seed", 0)
    m.complexity_level = complexity_level

    # Solve rate
    m.solved = bool(episode_summary.get("accusation_correct", False))
    if m.solved:
        m.suspect_correct = m.weapon_correct = m.location_correct = True
        m.partial_score = 1.0

    # Clue efficiency
    total_evidence = episode_summary.get("total_evidence", 1)
    discovered = len(episode_summary.get("evidence_discovered", []))
    m.total_relevant_evidence = total_evidence
    m.evidence_discovered = discovered
    m.clue_efficiency = discovered / max(1, total_evidence)

    # Belief accuracy trace
    gt_suspect = ground_truth.get("culprit_name", "")
    for snap in belief_snapshots:
        sprobs = snap.get("suspect_probs", {})
        if sprobs and gt_suspect:
            accuracy = sprobs.get(gt_suspect, 0)
            m.belief_accuracy_trace.append(accuracy)
        else:
            m.belief_accuracy_trace.append(0.0)
    if m.belief_accuracy_trace:
        m.final_belief_accuracy = m.belief_accuracy_trace[-1]

    # Token cost / action efficiency
    m.total_tokens = total_tokens
    m.actions_used = episode_summary.get("actions_taken", 0)
    m.action_budget = episode_summary.get("budget", 0)
    m.tokens_per_action = total_tokens / max(1, m.actions_used)
    m.action_efficiency = m.actions_used / max(1, m.action_budget)

    m.total_steps = episode_summary.get("steps_elapsed", 0)
    m.event_count = episode_summary.get("event_count", 0)

    # Examine efficiency (tracked by the environment)
    m.examine_total = episode_summary.get("examine_total", 0)
    m.examine_hit = episode_summary.get("examine_hit", 0)
    m.examine_efficiency = (
        m.examine_hit / m.examine_total if m.examine_total > 0 else 1.0
    )

    # Score breakdown — populated when the agent ACCUSEd with scoring kwargs
    score = episode_summary.get("score_result") or {}
    if score:
        m.partial_score = score.get("accusation_score", m.partial_score)
        m.accusation_score = score.get("accusation_score", 0.0)
        m.suspect_correct = bool(score.get("correct_suspect", m.suspect_correct))
        m.weapon_correct = bool(score.get("correct_weapon", m.weapon_correct))
        m.location_correct = bool(score.get("correct_room", m.location_correct))

        m.suspect_weapon_precision = score.get("suspect_weapon_precision", 0.0)
        m.suspect_weapon_recall = score.get("suspect_weapon_recall", 0.0)
        m.suspect_weapon_score = score.get("suspect_weapon_score", 0.0)
        m.weapon_victim_precision = score.get("weapon_victim_precision", 0.0)
        m.weapon_victim_recall = score.get("weapon_victim_recall", 0.0)
        m.weapon_victim_score = score.get("weapon_victim_score", 0.0)
        m.suspect_room_precision = score.get("suspect_room_precision", 0.0)
        m.suspect_room_recall = score.get("suspect_room_recall", 0.0)
        m.suspect_room_score = score.get("suspect_room_score", 0.0)
        m.triangle_score = score.get("triangle_score", 0.0)

        m.alibi_cited = bool(score.get("alibi_cited", False))
        m.contradiction_found = bool(score.get("contradiction_found", False))
        m.contradiction_valid = bool(score.get("contradiction_valid", False))
        m.alibi_score = score.get("alibi_score", 0.0)

        m.correct_eliminations = int(score.get("correct_eliminations", 0))
        m.incorrect_eliminations = int(score.get("incorrect_eliminations", 0))
        m.elimination_score = score.get("elimination_score", 0.0)

        m.composite_score = score.get("composite_score", 0.0)

    return m


# ---------------------------------------------------------------------------
# Aggregate metrics across instances
# ---------------------------------------------------------------------------

@dataclass
class AggregateMetrics:
    """Aggregated metrics across multiple episodes, grouped by complexity."""
    complexity_level: int = 0
    n_instances: int = 0
    solve_rate: float = 0.0
    mean_partial_score: float = 0.0
    mean_belief_accuracy: float = 0.0
    mean_clue_efficiency: float = 0.0
    mean_tokens: float = 0.0
    mean_action_efficiency: float = 0.0
    std_solve_rate: float = 0.0
    mean_triangle_score: float = 0.0
    mean_alibi_score: float = 0.0
    mean_composite_score: float = 0.0
    mean_triangle_precision: float = 0.0
    mean_triangle_recall: float = 0.0
    mean_elimination_score: float = 0.0
    mean_examine_efficiency: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "complexity_level": self.complexity_level,
            "n_instances": self.n_instances,
            "solve_rate": round(self.solve_rate, 4),
            "mean_partial_score": round(self.mean_partial_score, 4),
            "mean_belief_accuracy": round(self.mean_belief_accuracy, 4),
            "mean_clue_efficiency": round(self.mean_clue_efficiency, 4),
            "mean_tokens": round(self.mean_tokens, 1),
            "mean_action_efficiency": round(self.mean_action_efficiency, 4),
            "std_solve_rate": round(self.std_solve_rate, 4),
            "mean_triangle_score": round(self.mean_triangle_score, 4),
            "mean_alibi_score": round(self.mean_alibi_score, 4),
            "mean_composite_score": round(self.mean_composite_score, 4),
        }


def aggregate_metrics(episodes: list[EpisodeMetrics], level: int) -> AggregateMetrics:
    """Aggregate episode metrics for a given complexity level."""
    level_eps = [e for e in episodes if e.complexity_level == level]
    if not level_eps:
        return AggregateMetrics(complexity_level=level)

    n = len(level_eps)
    solve_rate = sum(e.solved for e in level_eps) / n
    agg = AggregateMetrics(
        complexity_level=level,
        n_instances=n,
        solve_rate=solve_rate,
        mean_partial_score=sum(e.partial_score for e in level_eps) / n,
        mean_belief_accuracy=sum(e.final_belief_accuracy for e in level_eps) / n,
        mean_clue_efficiency=sum(e.clue_efficiency for e in level_eps) / n,
        mean_tokens=sum(e.total_tokens for e in level_eps) / n,
        mean_action_efficiency=sum(e.action_efficiency for e in level_eps) / n,
        std_solve_rate=math.sqrt(solve_rate * (1 - solve_rate) / n) if n > 1 else 0.0,
        mean_triangle_score=sum(e.triangle_score for e in level_eps) / n,
        mean_alibi_score=sum(e.alibi_score for e in level_eps) / n,
        mean_composite_score=sum(e.composite_score for e in level_eps) / n,
        mean_triangle_precision=sum(
            (e.suspect_weapon_precision + e.weapon_victim_precision
             + e.suspect_room_precision) / 3.0
            for e in level_eps
        ) / n,
        mean_triangle_recall=sum(
            (e.suspect_weapon_recall + e.weapon_victim_recall
             + e.suspect_room_recall) / 3.0
            for e in level_eps
        ) / n,
        mean_elimination_score=sum(e.elimination_score for e in level_eps) / n,
        mean_examine_efficiency=sum(e.examine_efficiency for e in level_eps) / n,
    )
    return agg
