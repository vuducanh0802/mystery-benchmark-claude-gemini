"""
World state and simulation engine

The ``WorldState`` holds the complete ground truth. The ``MysteryEnvironment``
wraps it with an agent-facing API that returns *observations* (partial information)
and accepts *actions*.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

from mystery_world import ComplexityConfig
from mystery_world.entities import (
    Character,
    CharacterRole,
    EdgeArgument,
    EdgeRelevance,
    EdgeType,
    Evidence,
    EvidenceState,
    Location,
    RouteConstraint,
    ScoreResult,
    TemporalLabel,
    TimelineEntry,
    WitnessStatement,
    WorldObject,
)
from mystery_world.events import WorldEvent, process_all_events
from mystery_world.npc_responder import NPCResponder

# ---------------------------------------------------------------------------
# Agent action space
# ---------------------------------------------------------------------------

class AgentAction(Enum):
    MOVE = auto()               # move to adjacent location
    EXAMINE_LOCATION = auto()               # look around current room
    EXAMINE_OBJECT = auto()               # inspect a specific object
    TALK_TO = auto()               # interrogate a character
    ACCUSE = auto()               # make final accusation
    WAIT = auto()               # pass one time step
    CHECK_INVENTORY = auto()               # review collected clues
    TAKE_OBJECT = auto()               # pick up portable object
    ANALYZE = auto()         # temporal assessment of one piece of evidence
    TRAVEL_TIME = auto()     # minimum steps between two rooms (constraint-aware)
    CHECK_ROUTE = auto()     # was a specific passage open at a given clock time?

def _perception_roll(seed: int, evidence_id: str, attempt_idx: int) -> float:
    """Deterministic uniform draw in [0, 1) keyed by (world seed, evidence, attempt).

    Pure function of its inputs — independent of global RNG ordering or agent
    determinism — so a logged trajectory replays bit-for-bit and
    ``verify_reproducibility`` stays valid even with non-deterministic agents.
    """
    digest = hashlib.blake2b(evidence_id.encode(), digest_size=8).digest()
    key = [int(seed), int.from_bytes(digest, "big"), int(attempt_idx)]
    return float(np.random.default_rng(key).random())


@dataclass
class ActionResult:
    success: bool = True
    observation: str = ""
    evidence_found: list[str] = field(default_factory=list)   # evidence IDs
    details: dict[str, Any] = field(default_factory=dict)


DETECTIVE_ACTOR_ID = "detective"


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _normalized_name(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


# ---------------------------------------------------------------------------
# WorldState (ground truth)
# ---------------------------------------------------------------------------

@dataclass
class WorldState:
    """Complete ground-truth state of the mystery world."""

    seed: int = 0
    config: ComplexityConfig = field(default_factory=ComplexityConfig)

    # --- Entity registries (id -> entity) ---
    locations: dict[str, Location] = field(default_factory=dict)
    characters: dict[str, Character] = field(default_factory=dict)
    objects: dict[str, WorldObject] = field(default_factory=dict)
    evidence: dict[str, Evidence] = field(default_factory=dict)

    # --- Temporal ---
    current_step: int = 0
    weather: str = "clear"
    ground_truth_timeline: list[TimelineEntry] = field(default_factory=list)
    event_log: list[WorldEvent] = field(default_factory=list)
    
    # --- Solution ---
    culprit_id: str = ""
    victim_id: str = ""
    murder_weapon_id: str = ""
    murder_location_id: str = ""   # where the murder was committed
    body_location_id: str = ""     # where the body was found (may differ)
    murder_step: int = 0
    motive: str = ""
    # --- Locard triangle ---
    murder_timestamp: float = 0.0
    freshness_threshold: float = 2.0

    # --- Temporal reasoning ---
    witness_statements: list[WitnessStatement] = field(default_factory=list)
    route_constraints: list[RouteConstraint] = field(default_factory=list)
    anchor_events: dict[str, int] = field(default_factory=dict)


    def get_culprit(self) -> Character | None:
        return self.characters.get(self.culprit_id)


    def get_victim(self) -> Character | None:
        return self.characters.get(self.victim_id)

    
    # --- Serialisation ---
    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "config": self.config.to_dict(),
            "locations": {k: v.to_dict() for k, v in self.locations.items()},
            "characters": {k: v.to_dict() for k, v in self.characters.items()},
            "objects": {k: v.to_dict() for k, v in self.objects.items()},
            "evidence": {k: v.to_dict() for k, v in self.evidence.items()},
            "current_step": self.current_step,
            "weather": self.weather,
            "ground_truth_timeline": [e.to_dict() for e in self.ground_truth_timeline],
            "event_log": [e.to_dict() for e in self.event_log],
            "culprit_id": self.culprit_id,
            "victim_id": self.victim_id,
            "murder_weapon_id": self.murder_weapon_id,
            "murder_location_id": self.murder_location_id,
            "body_location_id": self.body_location_id,
            "murder_step": self.murder_step,
            "motive": self.motive,
            "murder_timestamp": self.murder_timestamp,
            "freshness_threshold": self.freshness_threshold,
            "witness_statements": [w.to_dict() for w in self.witness_statements],
            "route_constraints": [r.to_dict() for r in self.route_constraints],
            "anchor_events": self.anchor_events,
        }


    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    
    @classmethod
    def load(cls, path: str | Path) -> "WorldState":
        d = json.loads(Path(path).read_text())
        ws = cls(
            seed=d["seed"],
            config=ComplexityConfig.from_dict(d["config"]),
            current_step=d["current_step"],
            weather=d["weather"],
            culprit_id=d["culprit_id"],
            victim_id=d["victim_id"],
            murder_weapon_id=d["murder_weapon_id"],
            murder_location_id=d["murder_location_id"],
            body_location_id=d.get("body_location_id", d["murder_location_id"]),
            murder_step=d["murder_step"],
            murder_timestamp=d.get("murder_timestamp", float(d["murder_step"])),
            freshness_threshold=d.get("freshness_threshold", 2.0),
            anchor_events=d.get("anchor_events", {}),
            motive=d["motive"],
        )
        ws.locations = {k: Location.from_dict(v) for k, v in d["locations"].items()}
        ws.characters = {k: Character.from_dict(v) for k, v in d["characters"].items()}
        ws.objects = {k: WorldObject.from_dict(v) for k, v in d["objects"].items()}
        ws.evidence = {k: Evidence.from_dict(v) for k, v in d["evidence"].items()}
        ws.ground_truth_timeline = [TimelineEntry.from_dict(e) for e in d["ground_truth_timeline"]]
        ws.witness_statements = [
            WitnessStatement.from_dict(w) for w in d.get("witness_statements", [])
        ]
        ws.route_constraints = [
            RouteConstraint.from_dict(r) for r in d.get("route_constraints", [])
        ]
        return ws


# ---------------------------------------------------------------------------
# MysteryEnvironment (agent-facing interface)
# ---------------------------------------------------------------------------

class MysteryEnvironment:
    """
    Wraps WorldState with an agent-facing partial-observability interface.

    The agent interacts through ``step(action, **kwargs) -> ActionResult``.
    Observations are rendered as natural-language strings by the narrator.
    """
    def __init__(self, world_state: WorldState):
        self._state = world_state
        self._rng = np.random.default_rng(world_state.seed + 1000)   # offset for event RNG
        self.agent_location_id: str = ""
        self.agent_inventory: list[str] = []  # evidence IDs collected
        self.actions_taken: int = 0
        self.action_history: list[dict[str, Any]] = []
        self.is_solved: bool = False
        self.accusation_correct: bool | None = None
        self._discovered_evidence: set[str] = set()
        self._interviewed_characters: set[str] = set()
        self._last_score_result: dict[str, Any] | None = None
        self._interview_histories: dict[str, list[dict[str, str]]] = {}
        self._examine_total: int = 0
        self._examine_hit: int = 0
        self._examine_present: int = 0   # EXAMINEs where the object DID hold usable evidence (roll-independent)
        self._perception_misses: list[dict[str, Any]] = []
        self._perception_disabled: bool = False   # oracles / solvability checks bypass the roll
        self._revealed_alibi_claims: list[dict[str, str]] = []
        self._npc_responder: NPCResponder | None = None
        self._active_actor_id: str = DETECTIVE_ACTOR_ID
        self._actor_actions_taken: dict[str, int] = {}
        self._actor_discovered_evidence: dict[str, set[str]] = {}
        self._solvability_guard_blocked_actions: int = 0
        self._solvability_guard_suppressed_events: int = 0

        # Place agent at a default starting location
        if world_state.locations:
            self.agent_location_id = next(iter(world_state.locations))


    @property
    def state(self) -> WorldState:
        return self._state


    @property
    def budget_remaining(self) -> int:
        return max(0, self._state.config.max_agent_actions - self.actions_taken)

    def budget_remaining_for(self, actor_id: str = DETECTIVE_ACTOR_ID) -> int:
        if actor_id == DETECTIVE_ACTOR_ID:
            return self.budget_remaining
        used = self._actor_actions_taken.get(actor_id, 0)
        return max(0, self._state.config.max_agent_actions - used)

    @property
    def culprit_budget_remaining(self) -> int:
        return self.budget_remaining_for(self._state.culprit_id)

    def enable_free_culprit(self) -> None:
        """Route culprit movement/tampering through explicit agent actions."""
        self._state.config = replace(self._state.config, free_culprit_actions=True)

    def set_npc_responder(self, responder: NPCResponder) -> None:
        """Attach an NPC responder for LLM-powered stateful interviews."""
        self._npc_responder = responder

    def _restore_state_in_place(self, snapshot: WorldState) -> None:
        """Restore WorldState without replacing the object shared by callers."""
        self._state.__dict__.clear()
        self._state.__dict__.update(copy.deepcopy(snapshot.__dict__))

    def _snapshot_runtime(self) -> dict[str, Any]:
        return {
            "state": copy.deepcopy(self._state),
            "agent_location_id": self.agent_location_id,
            "agent_inventory": list(self.agent_inventory),
            "actions_taken": self.actions_taken,
            "action_history": copy.deepcopy(self.action_history),
            "is_solved": self.is_solved,
            "accusation_correct": self.accusation_correct,
            "discovered_evidence": set(self._discovered_evidence),
            "interviewed_characters": set(self._interviewed_characters),
            "last_score_result": copy.deepcopy(self._last_score_result),
            "interview_histories": copy.deepcopy(self._interview_histories),
            "examine_total": self._examine_total,
            "examine_hit": self._examine_hit,
            "examine_present": self._examine_present,
            "perception_misses": copy.deepcopy(self._perception_misses),
            "revealed_alibi_claims": copy.deepcopy(self._revealed_alibi_claims),
            "active_actor_id": self._active_actor_id,
            "actor_actions_taken": dict(self._actor_actions_taken),
            "actor_discovered_evidence": {
                actor_id: set(eids)
                for actor_id, eids in self._actor_discovered_evidence.items()
            },
            "solvability_guard_blocked_actions": self._solvability_guard_blocked_actions,
            "solvability_guard_suppressed_events": self._solvability_guard_suppressed_events,
        }

    def _restore_runtime(self, snapshot: dict[str, Any]) -> None:
        self._restore_state_in_place(snapshot["state"])
        self.agent_location_id = snapshot["agent_location_id"]
        self.agent_inventory = snapshot["agent_inventory"]
        self.actions_taken = snapshot["actions_taken"]
        self.action_history = snapshot["action_history"]
        self.is_solved = snapshot["is_solved"]
        self.accusation_correct = snapshot["accusation_correct"]
        self._discovered_evidence = snapshot["discovered_evidence"]
        self._interviewed_characters = snapshot["interviewed_characters"]
        self._last_score_result = snapshot["last_score_result"]
        self._interview_histories = snapshot["interview_histories"]
        self._examine_total = snapshot["examine_total"]
        self._examine_hit = snapshot["examine_hit"]
        self._examine_present = snapshot["examine_present"]
        self._perception_misses = snapshot["perception_misses"]
        self._revealed_alibi_claims = snapshot["revealed_alibi_claims"]
        self._active_actor_id = snapshot["active_actor_id"]
        self._actor_actions_taken = snapshot["actor_actions_taken"]
        self._actor_discovered_evidence = snapshot["actor_discovered_evidence"]
        self._solvability_guard_blocked_actions = snapshot["solvability_guard_blocked_actions"]
        self._solvability_guard_suppressed_events = snapshot["solvability_guard_suppressed_events"]

    def _solvability_report(self) -> dict[str, Any]:
        if self.is_solved:
            return {"solvable": True}
        murder_ts = self._state.murder_timestamp
        threshold = self._state.freshness_threshold

        def _has_accessible_host(evidence_id: str) -> bool:
            for obj in self._state.objects.values():
                if obj.evidence_id != evidence_id:
                    continue
                return obj.location_id in self._state.locations
            ev = self._state.evidence.get(evidence_id)
            return bool(ev and ev.location_id in self._state.locations)

        def _edge_candidates(edge_type: EdgeType) -> list[str]:
            candidates = []
            for ev in self._state.evidence.values():
                if ev.is_red_herring or ev.relevance is None:
                    continue
                if ev.relevance.edge_type != edge_type:
                    continue
                if not _relevance_matches_truth(ev.relevance, edge_type, self._state):
                    continue
                if abs(ev.relevance.contact_timestamp - murder_ts) >= threshold:
                    continue
                if ev.id in self._discovered_evidence:
                    candidates.append(ev.id)
                    continue
                if ev.state in (EvidenceState.HIDDEN, EvidenceState.DESTROYED):
                    continue
                if ev.discovery_difficulty >= 1.0:
                    continue
                if not _has_accessible_host(ev.id):
                    continue
                candidates.append(ev.id)
            return candidates

        edges = {
            edge_type.name: _edge_candidates(edge_type)
            for edge_type in (
                EdgeType.SUSPECT_WEAPON,
                EdgeType.WEAPON_VICTIM,
                EdgeType.SUSPECT_ROOM,
            )
        }
        solvable = all(edges[name] for name in edges)
        return {
            "solvable": solvable,
            "available_triangle_evidence": edges,
        }
    
    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _actor_display_name(self, actor_id: str) -> str:
        if actor_id == DETECTIVE_ACTOR_ID:
            return "the detective"
        char = self._state.characters.get(actor_id)
        return char.full_name if char else actor_id

    def _actor_location_id(self, actor_id: str) -> str:
        if actor_id == DETECTIVE_ACTOR_ID:
            return self.agent_location_id
        char = self._state.characters.get(actor_id)
        return char.location_id if char else ""

    def _set_actor_location(self, actor_id: str, location_id: str) -> None:
        if actor_id == DETECTIVE_ACTOR_ID:
            self.agent_location_id = location_id
            return
        char = self._state.characters.get(actor_id)
        if char is None:
            return
        old_loc = self._state.locations.get(char.location_id)
        if old_loc and actor_id in old_loc.characters_here:
            old_loc.characters_here.remove(actor_id)
        char.location_id = location_id
        new_loc = self._state.locations.get(location_id)
        if new_loc and actor_id not in new_loc.characters_here:
            new_loc.characters_here.append(actor_id)

    def _actor_inventory(self, actor_id: str) -> list[str]:
        if actor_id == DETECTIVE_ACTOR_ID:
            return self.agent_inventory
        char = self._state.characters.get(actor_id)
        if char is None:
            return []
        return char.inventory

    def _discovered_for_actor(self, actor_id: str) -> set[str]:
        if actor_id == DETECTIVE_ACTOR_ID:
            return self._discovered_evidence
        return self._actor_discovered_evidence.setdefault(actor_id, set())

    def get_current_location(self, actor_id: str | None = None) -> Location | None:
        actor_id = actor_id or self._active_actor_id
        return self._state.locations.get(self._actor_location_id(actor_id))

    def _is_body_object(self, obj: WorldObject) -> bool:
        return _normalized_name(obj.name).startswith("body of ")

    def _actor_is_culprit(self, actor_id: str) -> bool:
        char = self._state.characters.get(actor_id)
        return bool(char and char.is_culprit)

    def _can_actor_take_object(self, actor_id: str, obj: WorldObject) -> bool:
        if not obj.portable:
            return False
        return not (self._actor_is_culprit(actor_id) and obj.evidence_id)

    def observe_location(self, actor_id: str | None = None) -> str:
        """Return a natural-language description of the current location."""
        actor_id = actor_id or self._active_actor_id
        loc = self.get_current_location(actor_id)
        if loc is None:
            return "You are nowhere."
        parts = [f"You are in the {loc.name}. {loc.description}"]
        # Characters present (alive) — include physical description
        chars_here = [
            self._state.characters[cid]
            for cid in loc.characters_here
            if (
                cid in self._state.characters
                and self._state.characters[cid].is_alive
                and cid != actor_id
            )
        ]
        # Dead bodies
        dead_here = [
            self._state.characters[cid]
            for cid in loc.characters_here
            if cid in self._state.characters and not self._state.characters[cid].is_alive
        ]
        for d in dead_here:
            parts.append(f"The body of {d.full_name} lies here.")
        if chars_here:
            names = ", ".join(c.full_name for c in chars_here)
            parts.append(f"Present here: {names}.")
        for c in chars_here:
            pt = c.physical_traits
            parts.append(f"{c.full_name} — {pt.build}, {pt.hair}, {pt.hands}.")
        # Visible objects (not hidden evidence)
        visible_objs = [
            self._state.objects[oid]
            for oid in loc.objects_here
            if oid in self._state.objects
        ]
        # Filter out hidden evidence
        visible = []
        for obj in visible_objs:
            if obj.evidence_id:
                ev = self._state.evidence.get(obj.evidence_id)
                if ev and ev.state == EvidenceState.HIDDEN and not self._is_body_object(obj):
                    continue
            visible.append(obj)
        
        if visible:
            obj_names = ", ".join(o.name for o in visible)
            parts.append(f"You notice: {obj_names}.")
        # Exits
        adj_names = [
            self._state.locations[aid].name
            for aid in loc.adjacent_ids
            if aid in self._state.locations
        ]
        if adj_names:
            parts.append(f"Exit leads to {', '.join(adj_names)}.")
        talk_targets = [c.full_name for c in chars_here]
        examine_targets = [o.name for o in visible]
        take_targets = [o.name for o in visible if self._can_actor_take_object(actor_id, o)]
        target_lines = [
            f"MOVE: {', '.join(adj_names) if adj_names else 'none'}",
            f"TALK_TO: {', '.join(talk_targets) if talk_targets else 'none'}",
            f"EXAMINE_OBJECT: {', '.join(examine_targets) if examine_targets else 'none'}",
            f"TAKE_OBJECT: {', '.join(take_targets) if take_targets else 'none'}",
        ]
        parts.append("Available targets: " + " | ".join(target_lines) + ".")
        # Weather (outdoor only)
        if loc.weather_exposed:
            parts.append(f"The weather is {self._state.weather.replace('_', ' ')}.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------
    def step(self, action: AgentAction, **kwargs: Any) -> ActionResult:
        """Execute a detective action and advance the world by one time step."""
        return self.step_for_actor(DETECTIVE_ACTOR_ID, action, **kwargs)

    def step_for_actor(
        self,
        actor_id: str,
        action: AgentAction,
        *,
        advance_world: bool = True,
        **kwargs: Any,
    ) -> ActionResult:
        """Execute an action for the detective or an in-world character."""
        is_detective = actor_id == DETECTIVE_ACTOR_ID
        if is_detective and self.is_solved:
            return ActionResult(success=False, observation="The case is already closed.")
        if not is_detective:
            actor = self._state.characters.get(actor_id)
            if actor is None or not actor.is_alive:
                return ActionResult(False, f"Actor '{actor_id}' cannot act.")
            if action == AgentAction.ACCUSE:
                return ActionResult(False, "Only the detective can make a final accusation.")
        if self.budget_remaining_for(actor_id) <= 0 and (is_detective and action != AgentAction.ACCUSE):
            return ActionResult(success=False, observation="You have exhausted your action budget. You must ACCUSE now.")
        if self.budget_remaining_for(actor_id) <= 0 and not is_detective:
            return ActionResult(False, f"{self._actor_display_name(actor_id)} has exhausted their action budget.")
        
        before_action = self._snapshot_runtime()
        prev_actor = self._active_actor_id
        self._active_actor_id = actor_id
        try:
            result = self._dispatch_action(action, **kwargs)
        finally:
            self._active_actor_id = prev_actor

        if result.success and action != AgentAction.ACCUSE:
            solvability = self._solvability_report()
            if not solvability.get("solvable", False):
                self._restore_runtime(before_action)
                self._solvability_guard_blocked_actions += 1
                return ActionResult(
                    False,
                    "That action is blocked because it would make the case unsolvable.",
                    details={"solvability": solvability},
                )

        # Record
        if is_detective:
            self.actions_taken += 1
        else:
            self._actor_actions_taken[actor_id] = self._actor_actions_taken.get(actor_id, 0) + 1
        self.action_history.append({
            "step": self._state.current_step,
            "actor_id": actor_id,
            "actor_name": self._actor_display_name(actor_id),
            "action": action.name,
            "kwargs": dict(kwargs),
            "success": result.success,
            "observation": result.observation[:500],
        })

        # Advance world simulation (dynamic events)
        if advance_world:
            before_events = copy.deepcopy(self._state)
            before_step = self._state.current_step
            self._state.current_step += 1
            new_events = process_all_events(self._state, self._rng)
            self._state.event_log.extend(new_events)
            solvability = self._solvability_report()
            if not solvability.get("solvable", False):
                self._restore_state_in_place(before_events)
                self._state.current_step = before_step + 1
                self._solvability_guard_suppressed_events += len(new_events)

        return result

    
    def _dispatch_action(self, action: AgentAction, **kwargs: Any) -> ActionResult:
        handlers = {
            AgentAction.MOVE: self._handle_move,
            AgentAction.EXAMINE_LOCATION: self._handle_examine_location,
            AgentAction.EXAMINE_OBJECT: self._handle_examine_object,
            AgentAction.TALK_TO: self._handle_talk,
            AgentAction.ACCUSE: self._handle_accuse,
            AgentAction.WAIT: self._handle_wait,
            AgentAction.CHECK_INVENTORY: self._handle_inventory,
            AgentAction.TAKE_OBJECT: self._handle_take,
            AgentAction.ANALYZE:      self._handle_analyze,
            AgentAction.TRAVEL_TIME:  self._handle_travel_time,
            AgentAction.CHECK_ROUTE:  self._handle_check_route,
        }
        handler = handlers.get(action, self._handle_wait)
        return handler(**kwargs)


    def _handle_analyze(self, evidence_id: str = "", **_: Any) -> ActionResult:
        """Temporal assessment of one piece of evidence. Costs 1 action."""
        if evidence_id not in self._discovered_evidence:
            return ActionResult(False, f"Evidence '{evidence_id}' not in your collection.")
        ev = self._state.evidence.get(evidence_id)
        if ev is None:
            return ActionResult(False, f"Evidence '{evidence_id}' does not exist.")
        if ev.relevance is None:
            return ActionResult(True, f"You analyze {ev.name}. No forensically relevant contact traces found.")

        rel = ev.relevance
        freshness = abs(rel.contact_timestamp - self._state.murder_timestamp)
        threshold = self._state.freshness_threshold

        if rel.surface_label == TemporalLabel.CLEARLY_FRESH:
            assessment = "This trace appears very recent — consistent with the time of the murder."
        elif rel.surface_label == TemporalLabel.CLEARLY_STALE:
            assessment = "This trace is old — clearly predates the murder by a significant margin."
        else:  # AMBIGUOUS
            if freshness < threshold:
                assessment = "Analysis suggests this trace is relatively recent, possibly within the relevant timeframe."
            else:
                assessment = "Analysis suggests this trace may be older than it first appears."

        return ActionResult(True, f"You analyze {ev.name}. {assessment}")


    def _handle_travel_time(
        self, from_room: str = "", to_room: str = "", at_time: str = "", **kwargs: Any
    ) -> ActionResult:
        """Minimum steps between two rooms, respecting route constraints active at_time.
        If at_time omitted, returns unconstrained minimum. Costs 1 action."""
        from_room = _first_nonempty(
            from_room,
            kwargs.get("from_location"),
            kwargs.get("from"),
            kwargs.get("source"),
        )
        to_room = _first_nonempty(
            to_room,
            kwargs.get("to_location"),
            kwargs.get("to"),
            kwargs.get("target"),
        )
        at_time = _first_nonempty(at_time, kwargs.get("time"))
        from_loc = next(
            (l for l in self._state.locations.values() if l.name.lower() == from_room.lower()),
            None,
        )
        to_loc = next(
            (l for l in self._state.locations.values() if l.name.lower() == to_room.lower()),
            None,
        )
        if from_loc is None:
            return ActionResult(False, f"Unknown room: '{from_room}'.")
        if to_loc is None:
            return ActionResult(False, f"Unknown room: '{to_room}'.")
        if from_loc.id == to_loc.id:
            return ActionResult(True, f"You are already in the {to_room}. No travel needed.")

        active: list[RouteConstraint] = []
        if at_time:
            at_step = _clock_str_to_step(at_time, self._state.config.world_start_hour)
            if at_step is not None:
                active = [
                    rc for rc in self._state.route_constraints
                    if rc.blocked_from_step <= at_step <= rc.blocked_until_step
                ]

        steps = _shortest_path_steps_constrained(
            from_loc.id, to_loc.id, self._state.locations, active
        )
        qualifier = f" at {at_time}" if at_time else ""
        if steps is None:
            return ActionResult(
                True, f"There is no open route from the {from_room} to the {to_room}{qualifier}."
            )
        minutes = steps * self._state.config.step_duration_minutes
        return ActionResult(
            True,
            f"The shortest open route from the {from_room} to the {to_room}{qualifier} takes "
            f"{steps} step{'s' if steps != 1 else ''} ({minutes} minutes).",
        )

    def _handle_check_route(
        self, from_room: str = "", to_room: str = "", at_time: str = "", **kwargs: Any
    ) -> ActionResult:
        """Was the direct passage between two rooms open at a given clock time?
        at_time format: '9:30 PM'. Costs 1 action."""
        from_room = _first_nonempty(
            from_room,
            kwargs.get("from_location"),
            kwargs.get("from"),
            kwargs.get("source"),
        )
        to_room = _first_nonempty(
            to_room,
            kwargs.get("to_location"),
            kwargs.get("to"),
            kwargs.get("target"),
        )
        at_time = _first_nonempty(at_time, kwargs.get("time"))
        from_loc = next(
            (l for l in self._state.locations.values() if l.name.lower() == from_room.lower()),
            None,
        )
        to_loc = next(
            (l for l in self._state.locations.values() if l.name.lower() == to_room.lower()),
            None,
        )
        if from_loc is None or to_loc is None:
            return ActionResult(False, f"Unknown room(s): '{from_room}', '{to_room}'.")

        step = _clock_str_to_step(at_time, self._state.config.world_start_hour)
        if step is None:
            return ActionResult(False, f"Could not parse time '{at_time}'. Use format like '9:30 PM'.")

        rc_match = next(
            (rc for rc in self._state.route_constraints
             if rc.blocked_from_step <= step <= rc.blocked_until_step
             and (
                 (rc.from_location_id == from_loc.id and rc.to_location_id == to_loc.id)
                 or (rc.from_location_id == to_loc.id and rc.to_location_id == from_loc.id)
             )),
            None,
        )
        if rc_match:
            return ActionResult(
                True,
                f"The direct passage between the {from_room} and the {to_room} "
                f"was closed at {at_time}: {rc_match.reason}.",
            )
        return ActionResult(
            True,
            f"The passage between the {from_room} and the {to_room} was open at {at_time}.",
        )
    
    def _handle_move(
        self,
        target_location: str = "",
        **kwargs: Any
    ) -> ActionResult:
        target_location = _first_nonempty(
            target_location,
            kwargs.get("name"),
            kwargs.get("target"),
            kwargs.get("to_location"),
            kwargs.get("location"),
        )
        actor_id = self._active_actor_id
        loc = self.get_current_location(actor_id)
        if loc is None:
            return ActionResult(False, "Cannot move: current location unknown.")
        # Allow moving by name or ID
        target_id = None
        for aid in loc.adjacent_ids:
            adj = self._state.locations.get(aid)
            if adj and (aid == target_location or adj.name.lower() == target_location.lower()):
                target_id = aid
                break
        if target_id is None:
            return ActionResult(False, f"Cannot move to '{target_location}'. Available {', '.join(self._state.locations[a].name for a in loc.adjacent_ids if a in self._state.locations)}.")
        self._set_actor_location(actor_id, target_id)
        obs = self.observe_location(actor_id)
        return ActionResult(True, f"You move to the {self._state.locations[target_id].name}.\n{obs}")


    def _handle_examine_location(self, **_: Any) -> ActionResult:
        obs = self.observe_location(self._active_actor_id)
        return ActionResult(True, obs)


    def _object_matches_query(self, obj: WorldObject, query: str) -> bool:
        query_norm = _normalized_name(query)
        if not query_norm:
            return False
        names = [obj.id, obj.name]
        if self._is_body_object(obj):
            victim = self._state.get_victim()
            names.extend(["body", "the body", "victim", "the victim"])
            if victim is not None:
                names.extend([
                    victim.full_name,
                    f"{victim.full_name}'s body",
                    f"body of {victim.full_name}",
                    f"the body of {victim.full_name}",
                ])
        return query_norm in {_normalized_name(name) for name in names}

    def _handle_examine_object(
        self, object_name: str = "", thorough: bool = False, **kwargs: Any
    ) -> ActionResult:
        object_name = _first_nonempty(
            object_name,
            kwargs.get("name"),
            kwargs.get("object"),
            kwargs.get("target_name"),
        )
        actor_id = self._active_actor_id
        loc = self.get_current_location(actor_id)
        if loc is None:
            return ActionResult(False, "No current location.")
        for oid in loc.objects_here:
            obj = self._state.objects.get(oid)
            if obj and self._object_matches_query(obj, object_name):
                if actor_id == DETECTIVE_ACTOR_ID:
                    self._examine_total += 1
                base_obs = f"You examine the {obj.name}. {obj.description}"
                if not obj.evidence_id:
                    return ActionResult(True, base_obs)
                ev = self._state.evidence.get(obj.evidence_id)
                if ev is None or ev.state in (EvidenceState.HIDDEN, EvidenceState.DESTROYED):
                    return ActionResult(True, base_obs)

                if actor_id != DETECTIVE_ACTOR_ID:
                    self._discovered_for_actor(actor_id).add(ev.id)
                    return ActionResult(
                        True, f"{base_obs} [Evidence {ev.id}] {ev.description}",
                        evidence_found=[ev.id],
                    )

                # Object holds usable evidence. Already found → re-reveal, no roll.
                if ev.id in self._discovered_evidence:
                    return ActionResult(
                        True, f"{base_obs} [Evidence {ev.id}] {ev.description}",
                        evidence_found=[ev.id],
                    )

                self._examine_present += 1

                # Decay-retry perception roll (keyed, reproducible). Disabled for
                # oracles / solvability checks so they remain exact upper bounds.
                cfg = self._state.config
                if self._perception_disabled or cfg.detective_miss_base <= 0.0:
                    miss_p = 0.0
                else:
                    miss_p = (
                        cfg.detective_miss_base
                        * ev.discovery_difficulty
                        * (cfg.examine_attempt_decay ** ev.examine_attempts)
                    )
                    if thorough:
                        miss_p *= cfg.search_miss_multiplier
                roll = _perception_roll(self._state.seed, ev.id, ev.examine_attempts)
                ev.examine_attempts += 1

                if roll < miss_p:
                    # Missed. Observation MUST be byte-identical to the
                    # "object holds no evidence" branch so the agent cannot
                    # distinguish a perceptual miss from an empty object.
                    self._perception_misses.append({
                        "step": self._state.current_step,
                        "evidence_id": ev.id,
                        "attempt": ev.examine_attempts - 1,
                        "roll": round(roll, 6),
                        "miss_p": round(miss_p, 6),
                    })
                    return ActionResult(True, base_obs)

                self._examine_hit += 1
                self._discovered_evidence.add(ev.id)
                return ActionResult(
                    True, f"{base_obs} [Evidence {ev.id}] {ev.description}",
                    evidence_found=[ev.id],
                )
        return ActionResult(False, f"No object called '{object_name}' here.")

    
    def _handle_talk(self, character_name: str = "", question: str = "", **kwargs: Any) -> ActionResult:
        character_name = _first_nonempty(
            character_name,
            kwargs.get("name"),
            kwargs.get("target_name"),
            kwargs.get("character"),
        )
        actor_id = self._active_actor_id
        loc = self.get_current_location(actor_id)
        if loc is None:
            return ActionResult(False, "No current location.")
        for cid in loc.characters_here:
            char = self._state.characters.get(cid)
            if (
                char
                and char.is_alive
                and cid != actor_id
                and char.full_name.lower() == character_name.lower()
            ):
                if actor_id == DETECTIVE_ACTOR_ID:
                    self._interviewed_characters.add(cid)
                return self._generate_interview(char, question, questioner_id=actor_id)

        for cid, char in self._state.characters.items():
            if char.full_name.lower() == character_name.lower() and not char.is_alive:
                return ActionResult(False, f"{char.full_name} is dead and cannot be spoken to.")
        return ActionResult(False, f"'{character_name}' is not here or cannot be spoken to.")


    def _generate_interview(
        self,
        char: Character,
        question: str = "",
        *,
        questioner_id: str = DETECTIVE_ACTOR_ID,
    ) -> ActionResult:
        """Stateful multi-turn interview.

        Uses NPCResponder (LLM) when attached; deterministic fallback otherwise.
        The prompt provides role facts but does not prescribe a lying strategy.
        """
        if not question:
            question = "Where were you at the time of the murder?"
        cid = char.id
        history_key = cid if questioner_id == DETECTIVE_ACTOR_ID else f"{questioner_id}->{cid}"
        if history_key not in self._interview_histories:
            self._interview_histories[history_key] = []
        history = self._interview_histories[history_key]

        # Alibi provenance: interviewing the culprit reveals their alibi claims
        if questioner_id == DETECTIVE_ACTOR_ID and char.is_culprit and char.alibi_claims:
            for claim in char.alibi_claims:
                self._revealed_alibi_claims.append({
                    "character": char.full_name,
                    "location": claim.location_name,
                    "time": claim.clock_time_str,
                })

        if self._npc_responder is not None:
            response = self._npc_responder.respond(
                char,
                self._state,
                question,
                history,
                questioner_name=self._actor_display_name(questioner_id),
            )
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": response})
            return ActionResult(True, f'{char.full_name}: "{response}"')

        return self._template_interview(char, question, history, questioner_id=questioner_id)

    def _template_interview(
        self,
        char: Character,
        question: str,
        history: list[dict],
        *,
        questioner_id: str = DETECTIVE_ACTOR_ID,
    ) -> ActionResult:
        """Deterministic fallback used when no LLM responder is attached."""
        if questioner_id == DETECTIVE_ACTOR_ID:
            parts = [f'You ask {char.full_name}: "{question}"']
        else:
            parts = [f'{self._actor_display_name(questioner_id)} asks {char.full_name}: "{question}"']
        if char.is_culprit:
            if char.has_alibi:
                parts.append(f'{char.full_name} says: "{char.alibi_details}"')
            else:
                parts.append(f"{char.full_name} says they cannot recall anything specific.")
        elif char.has_alibi:
            parts.append(f'{char.full_name} says: "{char.alibi_details}"')
        else:
            parts.append(f"{char.full_name} says they cannot recall anything specific.")
        for rel in char.relationships:
            target = self._state.characters.get(rel.target_id)
            if target:
                if rel.sentiment < -0.3:
                    parts.append(f"They choose their words carefully when {target.full_name} comes up.")
                elif rel.sentiment > 0.5:
                    parts.append(f"They speak warmly of {target.full_name}.")
        response_text = " ".join(parts[1:])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response_text})
        return ActionResult(True, " ".join(parts))


    def _handle_accuse(
        self,
        suspect_name: str = "",
        weapon_name: str = "",
        location_name: str = "",
        suspect_weapon_evidence: list[str] | None = None,
        weapon_victim_evidence: list[str] | None = None,
        suspect_room_evidence: list[str] | None = None,
        alibi_contradiction: dict[str, Any] | None = None,
        eliminations: dict[str, dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ActionResult:
        """Final accusation. Ends the episode."""
        suspect_name = _first_nonempty(
            suspect_name,
            kwargs.get("suspect"),
            kwargs.get("culprit"),
            kwargs.get("character_name"),
            kwargs.get("accused"),
        )
        weapon_name = _first_nonempty(
            weapon_name,
            kwargs.get("weapon"),
            kwargs.get("murder_weapon"),
            kwargs.get("object_name"),
        )
        location_name = _first_nonempty(
            location_name,
            kwargs.get("location"),
            kwargs.get("room"),
            kwargs.get("murder_location"),
            kwargs.get("location_id"),
        )
        suspect_weapon_evidence = (
            suspect_weapon_evidence if isinstance(suspect_weapon_evidence, list) else []
        )
        weapon_victim_evidence = (
            weapon_victim_evidence if isinstance(weapon_victim_evidence, list) else []
        )
        suspect_room_evidence = (
            suspect_room_evidence if isinstance(suspect_room_evidence, list) else []
        )
        alibi_contradiction = alibi_contradiction if isinstance(alibi_contradiction, dict) else {}
        eliminations = eliminations if isinstance(eliminations, dict) else {}

        self.is_solved = True
        culprit = self._state.get_culprit()
        weapon = self._state.objects.get(self._state.murder_weapon_id)
        murder_loc = self._state.locations.get(self._state.murder_location_id)

        correct_suspect = culprit and culprit.full_name.lower() == suspect_name.lower()
        correct_weapon = weapon and weapon.name.lower() == weapon_name.lower()
        correct_location = murder_loc and murder_loc.name.lower() == location_name.lower()
        self.accusation_correct = correct_suspect and correct_weapon and correct_location

        details = {
            "suspect_correct": correct_suspect,
            "weapon_correct": correct_weapon,
            "location_correct": correct_location,
            "partial_score": sum([correct_suspect, correct_weapon, correct_location]) / 3.0,
        }

        has_scoring = any([
            suspect_weapon_evidence, weapon_victim_evidence,
            suspect_room_evidence, alibi_contradiction,
        ])
        if has_scoring:
            accused_ids = self._resolve_names_to_ids(suspect_name, weapon_name, location_name)
            triangle = {}
            if suspect_weapon_evidence:
                triangle["SUSPECT_WEAPON"] = EdgeArgument(evidence_ids=suspect_weapon_evidence)
            if weapon_victim_evidence:
                triangle["WEAPON_VICTIM"] = EdgeArgument(evidence_ids=weapon_victim_evidence)
            if suspect_room_evidence:
                triangle["SUSPECT_ROOM"] = EdgeArgument(evidence_ids=suspect_room_evidence)

            score = score_accusation(
                accused_ids=accused_ids,
                triangle=triangle,
                state=self._state,
                alibi_contradiction=alibi_contradiction,
                revealed_alibi_claims=self._revealed_alibi_claims,
                eliminations=eliminations,
                discovered_evidence=self._discovered_evidence,
                interviewed_characters=self._interviewed_characters,
                examine_total=self._examine_total,
                examine_hit=self._examine_hit,
            )
            score_dict = score.to_dict()
            self._last_score_result = score_dict
            details["score_result"] = score_dict
            details["triangle_score"] = score.triangle_score
            details["alibi_score"] = score.alibi_score
            details["composite_score"] = score.composite_score

        if self.accusation_correct:
            obs = (f"CORRECT! {culprit.full_name} committed the crime "
                   f"with the {weapon.name} in the {murder_loc.name}.")
        else:
            obs = (f"INCORRECT. The true answer: "
                   f"{culprit.full_name if culprit else '?'} with the "
                   f"{weapon.name if weapon else '?'} in the "
                   f"{murder_loc.name if murder_loc else '?'}.")
        if has_scoring:
            obs += (f" Triangle: {score.triangle_score:.1f}/3."
                    f" Alibi: {score.alibi_score:.2f}."
                    f" Composite: {score.composite_score:.2f}.")
        return ActionResult(True, obs, details=details)

    def _resolve_names_to_ids(
        self, suspect_name: str, weapon_name: str, location_name: str
    ) -> dict[str, str]:
        suspect_name = _first_nonempty(suspect_name)
        weapon_name = _first_nonempty(weapon_name)
        location_name = _first_nonempty(location_name)
        result = {"suspect": "", "weapon": "", "room": ""}
        for cid, c in self._state.characters.items():
            if c.full_name.lower() == suspect_name.lower():
                result["suspect"] = cid
                break
        for oid, o in self._state.objects.items():
            if o.name.lower() == weapon_name.lower():
                result["weapon"] = oid
                break
        for lid, l in self._state.locations.items():
            if l.name.lower() == location_name.lower():
                result["room"] = lid
                break
        return result


    def _handle_wait(self, **_: Any) -> ActionResult:
        return ActionResult(True, "You wait and observe. Time passes.") # Thong: is this a wasteful move? The agent is supposed to do something to not waste a move?


    def _handle_inventory(self, **_: Any) -> ActionResult:
        actor_id = self._active_actor_id
        if actor_id != DETECTIVE_ACTOR_ID:
            inventory = self._actor_inventory(actor_id)
            if not inventory:
                return ActionResult(True, "You are carrying nothing.")
            parts = ["You are carrying:"]
            for oid in inventory:
                obj = self._state.objects.get(oid)
                if obj:
                    parts.append(f"- {obj.name}")
            return ActionResult(True, "\n".join(parts))

        if not self._discovered_evidence:
            return ActionResult(True, "Your evidence collection is empty.")
        parts = ["Evidence collected:"]
        for eid in self._discovered_evidence:
            ev = self._state.evidence.get(eid)
            if ev:
                parts.append(f"- [{eid}] {ev.name} [{ev.evidence_type.name}] ({ev.state.name}): {ev.description}")
        return ActionResult(True, "\n".join(parts))


    def _handle_take(self, object_name: str = "", **kwargs: Any) -> ActionResult:
        object_name = _first_nonempty(
            object_name,
            kwargs.get("name"),
            kwargs.get("object"),
            kwargs.get("target_name"),
        )
        actor_id = self._active_actor_id
        loc = self.get_current_location(actor_id)
        if loc is None:
            return ActionResult(False, "No current location.")
        for oid in loc.objects_here:
            obj = self._state.objects.get(oid)
            if obj and self._object_matches_query(obj, object_name):
                if not obj.portable:
                    return ActionResult(False, f"The {obj.name} cannot be taken.")
                if self._actor_is_culprit(actor_id) and obj.evidence_id:
                    return ActionResult(False, f"The {obj.name} cannot be taken.")
                loc.objects_here.remove(oid)
                self._actor_inventory(actor_id).append(oid)
                obj.location_id = f"inventory:{actor_id}"
                if obj.evidence_id and obj.evidence_id in self._state.evidence:
                    self._state.evidence[obj.evidence_id].location_id = obj.location_id
                return ActionResult(True, f"You take the {obj.name}.")
        return ActionResult(False, f"No object called '{object_name}' here.")


    # ------------------------------------------------------------------
    # Summary for evaluation
    # ------------------------------------------------------------------
    def get_episode_summary(self) -> dict[str, Any]:
        return {
            "seed": self._state.seed,
            "complexity": self._state.config.to_dict(),
            "actions_taken": self.actions_taken,
            "budget": self._state.config.max_agent_actions,
            "is_solved": self.is_solved,
            "accusation_correct": self.accusation_correct,
            "evidence_discovered": list(self._discovered_evidence),
            "total_evidence": len(self._state.evidence),
            "examine_total": self._examine_total,
            "examine_hit": self._examine_hit,
            "examine_present": self._examine_present,
            "perception_misses": self._perception_misses,
            "perception_disabled": self._perception_disabled,
            "characters_interviewed": list(self._interviewed_characters),
            "alibi_claims_revealed": len(self._revealed_alibi_claims),
            "total_characters": len(self._state.characters),
            "steps_elapsed": self._state.current_step,
            "event_count": len(self._state.event_log),
            "action_history": self.action_history,
            "actor_actions_taken": dict(self._actor_actions_taken),
            "free_culprit_actions": self._state.config.free_culprit_actions,
            "solvability_guard_blocked_actions": self._solvability_guard_blocked_actions,
            "solvability_guard_suppressed_events": self._solvability_guard_suppressed_events,
            "score_result": self._last_score_result,
        }

    # ------------------------------------------------------------------
    # Session save / load (world + agent state + interview transcripts)
    # ------------------------------------------------------------------
    def save_session(self, directory: str | Path) -> Path:
        """
        Save the full session to *directory*:
          world.json          — full WorldState (can be reloaded with --load)
          session.json        — agent state + interview transcripts + action log
        Returns the directory path.
        """
        import datetime
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # World state
        self._state.save(directory / "world.json")

        # Agent state
        session = {
            "saved_at": datetime.datetime.now().isoformat(),
            "seed": self._state.seed,
            "agent_location_id": self.agent_location_id,
            "agent_inventory": self.agent_inventory,
            "actions_taken": self.actions_taken,
            "is_solved": self.is_solved,
            "accusation_correct": self.accusation_correct,
            "discovered_evidence": list(self._discovered_evidence),
            "examine_total": self._examine_total,
            "examine_hit": self._examine_hit,
            "examine_present": self._examine_present,
            "perception_misses": self._perception_misses,
            "interviewed_characters": list(self._interviewed_characters),
            "interview_histories": self._interview_histories,
            "revealed_alibi_claims": self._revealed_alibi_claims,
            "action_history": self.action_history,
            "active_actor_id": self._active_actor_id,
            "actor_actions_taken": self._actor_actions_taken,
            "actor_discovered_evidence": {
                actor_id: sorted(eids)
                for actor_id, eids in self._actor_discovered_evidence.items()
            },
            "solvability_guard_blocked_actions": self._solvability_guard_blocked_actions,
            "solvability_guard_suppressed_events": self._solvability_guard_suppressed_events,
        }
        (directory / "session.json").write_text(json.dumps(session, indent=2))
        return directory

    def load_session(self, directory: str | Path) -> None:
        """
        Restore agent state from a previously saved session directory.
        Call this after constructing MysteryEnvironment with the saved world.json.
        """
        directory = Path(directory)
        session = json.loads((directory / "session.json").read_text())
        self.agent_location_id = session["agent_location_id"]
        self.agent_inventory = session["agent_inventory"]
        self.actions_taken = session["actions_taken"]
        self.is_solved = session["is_solved"]
        self.accusation_correct = session["accusation_correct"]
        self._discovered_evidence = set(session["discovered_evidence"])
        self._examine_total = session.get("examine_total", 0)
        self._examine_hit = session.get("examine_hit", 0)
        self._examine_present = session.get("examine_present", 0)
        self._perception_misses = session.get("perception_misses", [])
        self._interviewed_characters = set(session["interviewed_characters"])
        self._interview_histories = session["interview_histories"]
        self._revealed_alibi_claims = session.get("revealed_alibi_claims", [])
        self.action_history = session["action_history"]
        self._active_actor_id = session.get("active_actor_id", DETECTIVE_ACTOR_ID)
        self._actor_actions_taken = session.get("actor_actions_taken", {})
        self._actor_discovered_evidence = {
            actor_id: set(eids)
            for actor_id, eids in session.get("actor_discovered_evidence", {}).items()
        }
        self._solvability_guard_blocked_actions = session.get("solvability_guard_blocked_actions", 0)
        self._solvability_guard_suppressed_events = session.get("solvability_guard_suppressed_events", 0)


# ---------------------------------------------------------------------------
# Locard triangle + alibi scoring
# ---------------------------------------------------------------------------

def score_accusation(
    accused_ids: dict[str, str],
    triangle: dict[str, EdgeArgument],
    state: WorldState,
    alibi_contradiction: dict[str, Any] | None = None,
    revealed_alibi_claims: list[dict[str, str]] | None = None,
    eliminations: dict[str, dict[str, str]] | None = None,
    discovered_evidence: set[str] | None = None,
    interviewed_characters: set[str] | None = None,
    examine_total: int = 0,
    examine_hit: int = 0,
) -> ScoreResult:
    """Score an accusation: accusation + triangle F1 + alibi + elimination."""
    result = ScoreResult()
    murder_ts = state.murder_timestamp
    threshold = state.freshness_threshold

    # --- Score 1: Accusation correctness ---
    result.correct_suspect = accused_ids.get("suspect") == state.culprit_id
    result.correct_weapon = accused_ids.get("weapon") == state.murder_weapon_id
    result.correct_room = accused_ids.get("room") == state.murder_location_id
    result.accusation_score = sum([
        result.correct_suspect, result.correct_weapon, result.correct_room
    ]) / 3.0

    # --- Score 2: Locard triangle (precision + recall → F1 per edge) ---
    def _count_available(edge_type: EdgeType) -> int:
        return sum(
            1 for ev in state.evidence.values()
            if not ev.is_red_herring
            and ev.relevance is not None
            and ev.relevance.edge_type == edge_type
            and _relevance_matches_truth(ev.relevance, edge_type, state)
            and abs(ev.relevance.contact_timestamp - murder_ts) < threshold
        )

    def _score_edge(edge_type: EdgeType) -> tuple[float, float, float]:
        arg = triangle.get(edge_type.name)
        total_available = _count_available(edge_type)
        if arg is None or not arg.evidence_ids:
            return (0.0, 0.0, 0.0)
        total_cited = len(arg.evidence_ids)
        correct_fresh = 0
        correct_stale = 0
        for eid in arg.evidence_ids:
            ev = state.evidence.get(eid)
            if ev is None or ev.is_red_herring or ev.relevance is None:
                continue
            rel = ev.relevance
            if rel.edge_type != edge_type:
                continue
            if not _relevance_matches_truth(rel, edge_type, state):
                continue
            if abs(rel.contact_timestamp - murder_ts) < threshold:
                correct_fresh += 1
            else:
                correct_stale += 1
        effective_correct = correct_fresh + 0.5 * correct_stale
        precision = effective_correct / total_cited if total_cited > 0 else 0.0
        recall = correct_fresh / total_available if total_available > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )
        return (precision, recall, f1)

    sw_p, sw_r, sw_f1 = _score_edge(EdgeType.SUSPECT_WEAPON)
    wv_p, wv_r, wv_f1 = _score_edge(EdgeType.WEAPON_VICTIM)
    sr_p, sr_r, sr_f1 = _score_edge(EdgeType.SUSPECT_ROOM)

    result.suspect_weapon_precision = sw_p
    result.suspect_weapon_recall = sw_r
    result.suspect_weapon_score = sw_f1
    result.weapon_victim_precision = wv_p
    result.weapon_victim_recall = wv_r
    result.weapon_victim_score = wv_f1
    result.suspect_room_precision = sr_p
    result.suspect_room_recall = sr_r
    result.suspect_room_score = sr_f1
    result.triangle_score = sw_f1 + wv_f1 + sr_f1

    # --- Score 3: Alibi verification (provenance + evidence_id contradiction) ---
    if alibi_contradiction and revealed_alibi_claims:
        cited_loc = _first_nonempty(alibi_contradiction.get("claimed_location")).lower()
        cited_time = _first_nonempty(alibi_contradiction.get("claimed_time")).lower()
        cited_ev_ids = alibi_contradiction.get("contradiction_evidence", []) or []
        if not isinstance(cited_ev_ids, list):
            cited_ev_ids = []

        claim_matches = any(
            r["location"].lower() == cited_loc and r["time"].lower() == cited_time
            for r in revealed_alibi_claims
        )

        discovered = discovered_evidence or set()

        def _alibi_ev_valid(eid: str) -> bool:
            if eid not in discovered or eid not in state.evidence:
                return False
            ev = state.evidence[eid]
            if ev.is_red_herring or ev.relevance is None:
                return False
            rel = ev.relevance
            return (
                rel.edge_type == EdgeType.SUSPECT_ROOM
                and state.culprit_id in rel.subject_ids
                and state.murder_location_id in rel.subject_ids
                and abs(rel.contact_timestamp - murder_ts) < threshold
            )

        if claim_matches:
            result.alibi_cited = True
            result.contradiction_found = any(_alibi_ev_valid(eid) for eid in cited_ev_ids)
            result.contradiction_valid = _validate_alibi_contradiction(
                accused_ids.get("suspect", ""), alibi_contradiction, state
            )
            result.alibi_score = sum([
                result.alibi_cited,
                result.contradiction_found,
                result.contradiction_valid,
            ]) / 3.0

    # --- Score 4: Elimination (SUSPECT_ELSEWHERE + corroborator interview) ---
    if eliminations:
        # Characters who appear only as corroborators (witnesses for someone else's
        # alibi) but have no SUSPECT_ELSEWHERE evidence for themselves are not
        # independent elimination targets — exclude them from the denominator.
        # Mutual corroborators (A witnesses B and B witnesses A) both have SE
        # evidence targeting themselves, so both stay in the denominator.
        corroborator_ids = {
            ev.corroborator_id
            for ev in state.evidence.values()
            if ev.relevance is not None
            and ev.relevance.edge_type == EdgeType.SUSPECT_ELSEWHERE
            and ev.corroborator_id
        }
        se_target_ids = {
            cid
            for ev in state.evidence.values()
            if ev.relevance is not None
            and ev.relevance.edge_type == EdgeType.SUSPECT_ELSEWHERE
            for cid in ev.relevance.subject_ids
        }
        corroborator_only_ids = corroborator_ids - se_target_ids
        total_innocents = sum(
            1 for c in state.characters.values()
            if c.is_alive and not c.is_culprit
            and CharacterRole.SUSPECT in c.roles
            and c.id not in corroborator_only_ids
        )
        correct = 0
        incorrect = 0
        discovered = discovered_evidence or set()
        interviewed = interviewed_characters or set()

        for suspect_name, claim in eliminations.items():
            suspect_name = _first_nonempty(suspect_name)
            char = next(
                (c for c in state.characters.values()
                 if c.full_name.lower() == suspect_name.lower()),
                None,
            )
            if char is None or not isinstance(claim, dict):
                continue

            evidence_id = claim.get("evidence_id", "")
            corroborator_name = claim.get("corroborator", "")

            ev_valid = False
            if (
                evidence_id
                and evidence_id in discovered
                and evidence_id in state.evidence
            ):
                ev = state.evidence[evidence_id]
                if (
                    not ev.is_red_herring
                    and ev.relevance is not None
                    and ev.relevance.edge_type == EdgeType.SUSPECT_ELSEWHERE
                    and char.id in ev.relevance.subject_ids
                    and abs(ev.relevance.contact_timestamp - murder_ts) < threshold
                ):
                    ev_valid = True

            corr_valid = False
            if ev_valid and corroborator_name:
                ev = state.evidence[evidence_id]
                if ev.corroborator_id:
                    corr_char = state.characters.get(ev.corroborator_id)
                    if (
                        corr_char is not None
                        and corr_char.full_name.lower() == corroborator_name.lower()
                        and ev.corroborator_id in interviewed
                    ):
                        corr_valid = True

            if ev_valid and corr_valid:
                if char.is_culprit:
                    incorrect += 1
                else:
                    correct += 1
            else:
                if char.is_culprit:
                    incorrect += 1

        result.total_innocents = total_innocents
        result.correct_eliminations = correct
        result.incorrect_eliminations = incorrect
        if total_innocents > 0:
            result.elimination_score = max(
                0.0, (correct - 2 * incorrect) / total_innocents
            )

    # --- Composite ---
    examine_efficiency = examine_hit / max(1, examine_total) if examine_total > 0 else 1.0
    base = (
        0.35 * result.accusation_score
        + 0.35 * (result.triangle_score / 3.0)
        + 0.15 * result.alibi_score
        + 0.15 * result.elimination_score
    )
    result.composite_score = base * (0.8 + 0.2 * examine_efficiency)
    return result


def _relevance_matches_truth(
    rel: EdgeRelevance, edge_type: EdgeType, state: WorldState
) -> bool:
    if edge_type == EdgeType.SUSPECT_WEAPON:
        return state.culprit_id in rel.subject_ids and state.murder_weapon_id in rel.subject_ids
    elif edge_type == EdgeType.WEAPON_VICTIM:
        return state.murder_weapon_id in rel.subject_ids and state.victim_id in rel.subject_ids
    elif edge_type == EdgeType.SUSPECT_ROOM:
        return state.culprit_id in rel.subject_ids and state.murder_location_id in rel.subject_ids
    return False


def _is_blocked(
    from_id: str,
    to_id: str,
    active_constraints: list[RouteConstraint],
) -> bool:
    return any(
        (rc.from_location_id == from_id and rc.to_location_id == to_id)
        or (rc.from_location_id == to_id and rc.to_location_id == from_id)
        for rc in active_constraints
    )


def _shortest_path_steps_constrained(
    from_id: str,
    to_id: str,
    locations: dict[str, Location],
    active_constraints: list[RouteConstraint] | None = None,
) -> int | None:
    """BFS respecting blocked passages. Returns steps or None if unreachable."""
    from collections import deque
    constraints = active_constraints or []
    visited = {from_id}
    queue = deque([(from_id, 0)])
    while queue:
        current_id, steps = queue.popleft()
        loc = locations.get(current_id)
        if loc is None:
            continue
        for adj_id in loc.adjacent_ids:
            if _is_blocked(current_id, adj_id, constraints):
                continue
            if adj_id == to_id:
                return steps + 1
            if adj_id not in visited:
                visited.add(adj_id)
                queue.append((adj_id, steps + 1))
    return None


def _has_path_avoiding(
    from_id: str,
    to_id: str,
    avoid_id: str,
    locations: dict[str, Location],
    active_constraints: list[RouteConstraint] | None = None,
) -> bool:
    """True if a path exists from→to that never enters avoid_id."""
    from collections import deque
    constraints = active_constraints or []
    visited = {from_id, avoid_id}
    queue = deque([from_id])
    while queue:
        current_id = queue.popleft()
        loc = locations.get(current_id)
        if loc is None:
            continue
        for adj_id in loc.adjacent_ids:
            if _is_blocked(current_id, adj_id, constraints):
                continue
            if adj_id == to_id:
                return True
            if adj_id not in visited:
                visited.add(adj_id)
                queue.append(adj_id)
    return False


def _clock_str_to_step(clock_str: str, world_start_hour: int) -> int | None:
    """Parse '9:30 PM' -> step index. Returns None if unparseable."""
    import re
    m = re.match(r"(\d{1,2}):(\d{2})\s*(AM|PM)?", clock_str.strip(), re.IGNORECASE)
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    meridiem = (m.group(3) or "").upper()
    if meridiem == "PM" and hour != 12:
        hour += 12
    elif meridiem == "AM" and hour == 12:
        hour = 0
    total_minutes = hour * 60 + minute
    start_minutes = world_start_hour * 60
    delta = total_minutes - start_minutes
    if delta < 0:
        delta += 24 * 60
    return delta // 30


def _step_to_clock_str(step: int, world_start_hour: int) -> str:
    """Convert step index to '9:30 PM' string."""
    total_minutes = world_start_hour * 60 + step * 30
    total_minutes %= 24 * 60
    hour, minute = divmod(total_minutes, 60)
    meridiem = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour}:{minute:02d} {meridiem}"


def _validate_alibi_contradiction(
    suspect_id: str,
    contradiction: dict[str, Any],
    state: WorldState,
) -> bool:
    """Check that the agent's cited contradiction is logically valid.

    Type A (one alibi claim): suspect claims to be at location X at murder_step,
    but X is not the murder location. Valid when the claim is at murder_step and
    the location differs — a witness or physical evidence at the crime scene
    disproves the claim.

    Type B (two alibi claims): murder_step falls between the two claims, and every
    open route from the before-location to the after-location passes through the
    murder location at exactly murder_step.
    """
    suspect = state.characters.get(suspect_id)
    if not suspect or not suspect.alibi_claims:
        return False

    murder_step = state.murder_step
    murder_loc_id = state.murder_location_id
    active_constraints = [
        rc for rc in state.route_constraints
        if rc.blocked_from_step <= murder_step <= rc.blocked_until_step
    ]

    def _find_loc(name: str) -> Location | None:
        return next(
            (l for l in state.locations.values() if l.name.lower() == name.lower()),
            None,
        )

    # --- Type A ---
    if len(suspect.alibi_claims) == 1:
        claim = suspect.alibi_claims[0]
        if claim.step != murder_step:
            return False
        claimed_loc = _find_loc(claim.location_name)
        if claimed_loc is None:
            return False
        return claimed_loc.id != murder_loc_id

    # --- Type B ---
    if len(suspect.alibi_claims) >= 2:
        before = min(suspect.alibi_claims, key=lambda a: a.step)
        after = max(suspect.alibi_claims, key=lambda a: a.step)
        if not (before.step < murder_step < after.step):
            return False
        before_loc = _find_loc(before.location_name)
        after_loc = _find_loc(after.location_name)
        if before_loc is None or after_loc is None:
            return False
        # No open path from before→after that avoids the murder location
        can_avoid = _has_path_avoiding(
            before_loc.id, after_loc.id, murder_loc_id,
            state.locations, active_constraints,
        )
        if can_avoid:
            return False
        # Timing: before_step + travel_to_murder == murder_step
        steps_to_murder = _shortest_path_steps_constrained(
            before_loc.id, murder_loc_id, state.locations, active_constraints
        )
        if steps_to_murder is None:
            return False
        return before.step + steps_to_murder == murder_step

    return False
