"""
Oracle calibration agent.

This agent knows the ground truth (culprit, weapon, location) but must still
interact with the game API to legally discover evidence before citing it.

Purpose: establish an upper-bound on benchmark scores by running a
deterministic minimum-action proof for every case. Any real agent's scores
should fall at or below the oracle's composite score.

Strategy
--------
1. Inspect ground truth to build a ``OraclePlan``:
   - One evidence item per Locard triangle edge (easiest fresh non-red-herring)
   - Alibi contradiction data from the culprit's alibi_claims
2. Plan an efficient visit order (nearest-neighbour TSP over evidence locations)
3. Execute: navigate → discover each evidence item
4. ACCUSE with the full triangle + alibi dict
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from agents.base_agent import BaseAgent, BeliefState
from mystery_world.entities import (
    AlibiClaim,
    CharacterRole,
    EdgeType,
    EvidenceState,
    EvidenceType,
)
from mystery_world.world import AgentAction, ActionResult, MysteryEnvironment


# ---------------------------------------------------------------------------
# Plan dataclasses
# ---------------------------------------------------------------------------

@dataclass
class _EvidenceTarget:
    """One evidence item the oracle needs to discover."""
    evidence_id: str
    location_id: str
    object_name: str | None  # non-None → use EXAMINE_OBJECT; None → SEARCH


@dataclass
class OraclePlan:
    """Minimum proof assembled from ground truth before any game actions."""
    sw_target: _EvidenceTarget | None = None   # SUSPECT_WEAPON edge
    wv_target: _EvidenceTarget | None = None   # WEAPON_VICTIM edge
    sr_target: _EvidenceTarget | None = None   # SUSPECT_ROOM edge
    visit_order: list[str] = field(default_factory=list)  # location IDs in visit order
    alibi_contradiction: dict[str, Any] | None = None     # passed to ACCUSE

    # Final accusation names (resolved from ground truth)
    culprit_name: str = ""
    weapon_name: str = ""
    location_name: str = ""


# ---------------------------------------------------------------------------
# Oracle agent
# ---------------------------------------------------------------------------

_MAX_SEARCHES_PER_TARGET = 10  # safety cap; avoids burning the whole budget on one stubborn piece


class OracleAgent(BaseAgent):
    """
    Deterministic calibration agent.

    Knows everything, proves with minimum legal actions.
    """

    def __init__(self, agent_id: str = "oracle"):
        super().__init__(agent_id)
        self._env: MysteryEnvironment | None = None
        self._plan: OraclePlan | None = None
        # Per-evidence search attempt counter — stops wasting actions once evidence is found
        self._search_attempts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def initialize(self, env: MysteryEnvironment, briefing: str) -> None:  # type: ignore[override]
        self._env = env
        self._plan = self._build_plan()
        self._search_attempts = {}
        # Set beliefs to ground truth immediately (oracle is omniscient)
        state = env.state
        culprit = state.get_culprit()
        if culprit:
            self.belief_state.suspect_probs = {culprit.full_name: 1.0}
        weapon = state.objects.get(state.murder_weapon_id)
        if weapon:
            self.belief_state.weapon_probs = {weapon.name: 1.0}
        murder_loc = state.locations.get(state.murder_location_id)
        if murder_loc:
            self.belief_state.location_probs = {murder_loc.name: 1.0}

    def decide_action(self, observation: str) -> tuple[AgentAction, dict[str, Any]]:
        """
        Dynamic action selection — checks live discovery state on every call.

        This avoids the static-queue problem where all search slots are emitted
        upfront: as soon as an evidence item is found (possibly alongside others
        in a single SEARCH), this method skips straight to the next target.
        """
        env = self._env
        assert env is not None
        plan = self._plan
        assert plan is not None

        discovered = env._discovered_evidence

        # All needed evidence collected → accuse
        needed = {
            t.evidence_id
            for t in (plan.sw_target, plan.wv_target, plan.sr_target)
            if t is not None
        }
        if needed.issubset(discovered):
            return self._make_accuse_action()

        # Follow the pre-planned visit order; skip locations whose targets are all done
        for loc_id in plan.visit_order:
            pending = [
                t for t in (plan.sw_target, plan.wv_target, plan.sr_target)
                if t is not None
                and t.location_id == loc_id
                and t.evidence_id not in discovered
                and self._search_attempts.get(t.evidence_id, 0) < _MAX_SEARCHES_PER_TARGET
            ]
            if not pending:
                continue

            target = pending[0]

            # Navigate one hop toward the target location if not already there
            if env.agent_location_id != target.location_id:
                path = self._bfs_path(env.agent_location_id, target.location_id)
                if path:
                    loc = env.state.locations.get(path[0])
                    if loc:
                        return AgentAction.MOVE, {"target_location": loc.name}

            # At the right location — attempt discovery
            if target.object_name:
                # Deterministic: linked object is always present and reveals evidence
                return AgentAction.EXAMINE_OBJECT, {"object_name": target.object_name}
            else:
                # Probabilistic search; increment counter so we don't loop forever
                self._search_attempts[target.evidence_id] = (
                    self._search_attempts.get(target.evidence_id, 0) + 1
                )
                return AgentAction.SEARCH_FOR_EVIDENCE, {}

        # All targets either found or exhausted their search budget → accuse
        return self._make_accuse_action()

    def update_beliefs(self, observation: str) -> None:
        pass  # Oracle beliefs are set at initialization

    def _make_accuse_action(self) -> tuple[AgentAction, dict[str, Any]]:
        """Assemble the ACCUSE kwargs from the plan."""
        plan = self._plan
        assert plan is not None
        kwargs: dict[str, Any] = {
            "suspect_name": plan.culprit_name,
            "weapon_name": plan.weapon_name,
            "location_name": plan.location_name,
        }
        if plan.sw_target:
            kwargs["suspect_weapon_evidence"] = [plan.sw_target.evidence_id]
        if plan.wv_target:
            kwargs["weapon_victim_evidence"] = [plan.wv_target.evidence_id]
        if plan.sr_target:
            kwargs["suspect_room_evidence"] = [plan.sr_target.evidence_id]
        if plan.alibi_contradiction:
            kwargs["alibi_contradiction"] = plan.alibi_contradiction
        return AgentAction.ACCUSE, kwargs

    # ------------------------------------------------------------------
    # Plan building (pure ground-truth reasoning, no game API calls)
    # ------------------------------------------------------------------

    def _build_plan(self) -> OraclePlan:
        env = self._env
        assert env is not None
        state = env.state

        plan = OraclePlan()

        # Resolve names
        culprit = state.get_culprit()
        weapon_obj = state.objects.get(state.murder_weapon_id)
        murder_loc = state.locations.get(state.murder_location_id)
        plan.culprit_name = culprit.full_name if culprit else ""
        plan.weapon_name = weapon_obj.name if weapon_obj else ""
        plan.location_name = murder_loc.name if murder_loc else ""

        # Pick best evidence per triangle edge
        plan.sw_target = self._best_evidence(EdgeType.SUSPECT_WEAPON)
        plan.wv_target = self._best_evidence(EdgeType.WEAPON_VICTIM)
        plan.sr_target = self._best_evidence(EdgeType.SUSPECT_ROOM)

        # Alibi contradiction
        if culprit and culprit.alibi_claims:
            plan.alibi_contradiction = self._build_alibi_contradiction(culprit.alibi_claims)

        # Visit order: start from agent's current location, greedy nearest-neighbour
        targets = [t for t in (plan.sw_target, plan.wv_target, plan.sr_target) if t is not None]
        unique_locs = list(dict.fromkeys(t.location_id for t in targets))  # preserve order, deduplicate
        plan.visit_order = self._greedy_visit_order(env.agent_location_id, unique_locs)

        return plan

    def _best_evidence(self, edge: EdgeType) -> _EvidenceTarget | None:
        """Return the easiest-to-discover, fresh, non-red-herring evidence for edge."""
        env = self._env
        assert env is not None
        state = env.state
        murder_ts = state.murder_timestamp
        threshold = state.freshness_threshold

        candidates = []
        for ev in state.evidence.values():
            if ev.is_red_herring:
                continue
            if ev.state in (EvidenceState.DESTROYED,):
                continue
            if ev.discovery_difficulty >= 1.0:
                continue
            if ev.relevance is None or ev.relevance.edge_type != edge:
                continue
            # Freshness check
            if abs(ev.relevance.contact_timestamp - murder_ts) >= threshold:
                continue
            # Must point to the right entities
            from mystery_world.world import _relevance_matches_truth
            if not _relevance_matches_truth(ev.relevance, edge, state):
                continue
            candidates.append(ev)

        if not candidates:
            return None

        # Sort: prefer non-HIDDEN first, then by difficulty ascending
        candidates.sort(key=lambda e: (e.state == EvidenceState.HIDDEN, e.discovery_difficulty))
        ev = candidates[0]

        # Find the linked WorldObject (if any) for deterministic discovery
        obj_name: str | None = None
        for obj in state.objects.values():
            if obj.evidence_id == ev.id and obj.location_id == ev.location_id:
                obj_name = obj.name
                break

        return _EvidenceTarget(
            evidence_id=ev.id,
            location_id=ev.location_id,
            object_name=obj_name,
        )

    def _build_alibi_contradiction(self, claims: list[AlibiClaim]) -> dict[str, Any]:
        """Build the alibi_contradiction dict expected by score_accusation."""
        env = self._env
        assert env is not None
        state = env.state
        murder_loc = state.locations.get(state.murder_location_id)
        murder_loc_name = murder_loc.name if murder_loc else ""

        if len(claims) == 1:
            # Type A: single claim at murder_step in wrong location
            claim = claims[0]
            return {
                "claimed_location": claim.location_name,
                "claimed_time": claim.clock_time_str,
                "contradiction_evidence": (
                    f"Physical evidence places suspect at the {murder_loc_name} "
                    f"at {claim.clock_time_str}, contradicting their stated alibi."
                ),
            }
        else:
            # Type B: two bracketing claims — route forced through crime scene
            before = min(claims, key=lambda a: a.step)
            after = max(claims, key=lambda a: a.step)
            return {
                "claimed_location": before.location_name,
                "claimed_time": before.clock_time_str,
                "contradiction_evidence": (
                    f"Route from {before.location_name} to {after.location_name} "
                    f"must pass through {murder_loc_name} — placing suspect at the "
                    f"crime scene at the murder time."
                ),
            }

    def _greedy_visit_order(self, start_id: str, target_ids: list[str]) -> list[str]:
        """Nearest-neighbour tour from start through all target location IDs."""
        env = self._env
        assert env is not None
        remaining = list(target_ids)
        order = []
        current = start_id
        while remaining:
            nearest = min(remaining, key=lambda t: self._bfs_dist(current, t))
            order.append(nearest)
            current = nearest
            remaining.remove(nearest)
        return order

    def _bfs_dist(self, from_id: str, to_id: str) -> int:
        """BFS hop count between two location IDs (0 = same location)."""
        if from_id == to_id:
            return 0
        env = self._env
        assert env is not None
        locations = env.state.locations
        visited = {from_id}
        queue: deque[tuple[str, int]] = deque([(from_id, 0)])
        while queue:
            current, dist = queue.popleft()
            loc = locations.get(current)
            if loc is None:
                continue
            for adj_id in loc.adjacent_ids:
                if adj_id == to_id:
                    return dist + 1
                if adj_id not in visited:
                    visited.add(adj_id)
                    queue.append((adj_id, dist + 1))
        return 999  # unreachable

    def _bfs_path(self, from_id: str, to_id: str) -> list[str]:
        """Return list of location IDs from from_id to to_id (excluding from_id)."""
        if from_id == to_id:
            return []
        env = self._env
        assert env is not None
        locations = env.state.locations
        visited = {from_id}
        # queue stores (current_id, path_so_far)
        queue: deque[tuple[str, list[str]]] = deque([(from_id, [])])
        while queue:
            current, path = queue.popleft()
            loc = locations.get(current)
            if loc is None:
                continue
            for adj_id in loc.adjacent_ids:
                new_path = path + [adj_id]
                if adj_id == to_id:
                    return new_path
                if adj_id not in visited:
                    visited.add(adj_id)
                    queue.append((adj_id, new_path))
        return []  # unreachable

    # ------------------------------------------------------------------
    # run() — convenience entry point
    # ------------------------------------------------------------------

    def run(self, env: MysteryEnvironment, briefing: str) -> dict[str, Any]:
        """
        Run the oracle to completion and return the episode summary.

        Parameters
        ----------
        env:      A freshly constructed MysteryEnvironment.
        briefing: The initial briefing string (from render_initial_briefing).

        Returns
        -------
        dict with keys: accusation_correct, triangle_score, alibi_score,
        composite_score, actions_taken, plan_summary.
        """
        self.initialize(env, briefing)

        while not env.is_solved and env.budget_remaining > 0:
            action, kwargs = self.decide_action("")
            result = env.step(action, **kwargs)
            self.record_action(action, kwargs, result.observation)
            if action == AgentAction.ACCUSE:
                break

        summary = env.get_episode_summary()
        details = env.action_history[-1].get("kwargs", {}) if env.action_history else {}

        # Extract scoring from the last action result (the ACCUSE)
        accuse_record = next(
            (r for r in reversed(env.action_history) if r["action"] == "ACCUSE"),
            {},
        )

        return {
            "accusation_correct": env.accusation_correct,
            "actions_taken": env.actions_taken,
            "plan_summary": {
                "culprit": self._plan.culprit_name if self._plan else "",
                "weapon": self._plan.weapon_name if self._plan else "",
                "location": self._plan.location_name if self._plan else "",
                "sw_evidence": self._plan.sw_target.evidence_id if self._plan and self._plan.sw_target else None,
                "wv_evidence": self._plan.wv_target.evidence_id if self._plan and self._plan.wv_target else None,
                "sr_evidence": self._plan.sr_target.evidence_id if self._plan and self._plan.sr_target else None,
                "alibi_type": (
                    "A" if self._plan and self._plan.alibi_contradiction
                    and len(env.state.get_culprit().alibi_claims) == 1
                    else ("B" if self._plan and self._plan.alibi_contradiction else "none")
                ) if self._plan else "none",
            },
            "episode_summary": summary,
        }
