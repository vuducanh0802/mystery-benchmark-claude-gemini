"""
World state and simulation engine

The ``WorldState`` holds the complete ground truth. The ``MysteryEnvironment``
wraps it with an agent-facing API that returns *observations* (partial information)
and accepts *actions*.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

from mystery_world import ComplexityConfig
from mystery_world.entities import (
    Character,
    CharacterRole,
    Evidence,
    EvidenceState,
    Location,
    TimelineEntry,
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
    SEARCH_FOR_EVIDENCE = auto()               # active search (higher chance of finding hidden clues)
    ACCUSE = auto()               # make final accusation
    WAIT = auto()               # pass one time step
    CHECK_INVENTORY = auto()               # review collected clues
    TAKE_OBJECT = auto()               # pick up portable object


@dataclass
class ActionResult:
    success: bool = True
    observation: str = ""
    evidence_found: list[str] = field(default_factory=list)   # evidence IDs
    details: dict[str, Any] = field(default_factory=dict)


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
            motive=d["motive"],
        )
        ws.locations = {k: Location.from_dict(v) for k, v in d["locations"].items()}
        ws.characters = {k: Character.from_dict(v) for k, v in d["characters"].items()}
        ws.objects = {k: WorldObject.from_dict(v) for k, v in d["objects"].items()}
        ws.evidence = {k: Evidence.from_dict(v) for k, v in d["evidence"].items()}
        ws.ground_truth_timeline = [TimelineEntry.from_dict(e) for e in d["ground_truth_timeline"]]
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
        self._interview_histories: dict[str, list[dict[str, str]]] = {}
        self._npc_responder: NPCResponder | None = None

        # Place agent at a default starting location
        if world_state.locations:
            self.agent_location_id = next(iter(world_state.locations))


    @property
    def state(self) -> WorldState:
        return self._state


    @property
    def budget_remaining(self) -> int:
        return max(0, self._state.config.max_agent_actions - self.actions_taken)

    def set_npc_responder(self, responder: NPCResponder) -> None:
        """Attach an NPC responder for LLM-powered stateful interviews."""
        self._npc_responder = responder
    
    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def get_current_location(self) -> Location | None:
        return self._state.locations.get(self.agent_location_id)


    def observe_location(self) -> str:
        """Return a natural-language description of the current location."""
        loc = self.get_current_location()
        if loc is None:
            return "You are nowhere."
        parts = [f"You are in the {loc.name}. {loc.description}"]
        # Characters present
        chars_here = [
            self._state.characters[cid]
            for cid in loc.characters_here
            if cid in self._state.characters and self._state.characters[cid].is_alive
        ]
        if chars_here:
            names = ", ".join(c.full_name for c in chars_here)
            parts.append(f"Present here: {names}.")
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
                if ev and ev.state == EvidenceState.HIDDEN:
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
        # Weather (outdoor only)
        if loc.weather_exposed:
            parts.append(f"The weather is {self._state.weather.replace('_', ' ')}.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------
    def step(self, action: AgentAction, **kwargs: Any) -> ActionResult:
        """Execute an agent action and advance the world by one time step."""
        if self.is_solved:
            return ActionResult(success=False, observation="The case is already closed.")
        if self.budget_remaining <= 0 and action != AgentAction.ACCUSE:
            return ActionResult(success=False, observation="You have exhausted your action budget. You must ACCUSE now.")
        
        result = self._dispatch_action(action, **kwargs)

        # Record
        self.actions_taken += 1
        self.action_history.append({
            "step": self._state.current_step,
            "action": action.name,
            "kwargs": {k: str(v) for k, v in kwargs.items()},
            "success": result.success,
            "observation": result.observation[:500],
        })

        # Advance world simulation (dynamic events)
        self._state.current_step += 1
        new_events = process_all_events(self._state, self._rng)
        self._state.event_log.extend(new_events)

        return result

    
    def _dispatch_action(self, action: AgentAction, **kwargs: Any) -> ActionResult:
        handlers = {
            AgentAction.MOVE: self._handle_move,
            AgentAction.EXAMINE_LOCATION: self._handle_examine_location,
            AgentAction.EXAMINE_OBJECT: self._handle_examine_object,
            AgentAction.TALK_TO: self._handle_talk,
            AgentAction.SEARCH_FOR_EVIDENCE: self._handle_search,
            AgentAction.ACCUSE: self._handle_accuse,
            AgentAction.WAIT: self._handle_wait,
            AgentAction.CHECK_INVENTORY: self._handle_inventory,
            AgentAction.TAKE_OBJECT: self._handle_take,
        }
        handler = handlers.get(action, self._handle_wait)
        return handler(**kwargs)

    
    def _handle_move(
        self,
        target_location: str = "",
        **_: Any
    ) -> ActionResult:
        loc = self.get_current_location()
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
        self.agent_location_id = target_id
        obs = self.observe_location()
        return ActionResult(True, f"You move to the {self._state.locations[target_id].name}.\n{obs}")


    def _handle_examine_location(self, **_: Any) -> ActionResult:
        obs = self.observe_location()
        return ActionResult(True, obs)


    def _handle_examine_object(self, object_name: str = "", **_: Any) -> ActionResult:
        loc = self.get_current_location()
        if loc is None:
            return ActionResult(False, "No current location.")
        for oid in loc.objects_here:
            obj = self._state.objects.get(oid)
            if obj and obj.name.lower() == object_name.lower():
                parts = [f"You examine the {obj.name}. {obj.description}"]
                if obj.evidence_id: # Thong: how do we know this is the evidence?
                    ev = self._state.evidence.get(obj.evidence_id)
                    if ev and ev.state != EvidenceState.HIDDEN and ev.state != EvidenceState.DESTROYED:
                        parts.append(f"This is evidence: {ev.description}")
                        self._discovered_evidence.add(ev.id)
                        return ActionResult(True, " ".join(parts), evidence_found=[ev.id])
                return ActionResult(True, " ".join(parts))
        return ActionResult(False, f"No object called '{object_name}' here.")

    
    def _handle_talk(self, character_name: str = "", question: str = "", **_: Any) -> ActionResult:
        loc = self.get_current_location()
        if loc is None:
            return ActionResult(False, "No current location.")
        for cid in loc.characters_here:
            char = self._state.characters.get(cid)
            if char and char.is_alive and char.full_name.lower() == character_name.lower():
                self._interviewed_characters.add(cid)
                return self._generate_interview(char, question)

        for cid, char in self._state.characters.items():
            if char.full_name.lower() == character_name.lower() and not char.is_alive:
                return ActionResult(False, f"{char.full_name} is dead and cannot be spoken to.")
        return ActionResult(False, f"'{character_name}' is not here or cannot be spoken to.")


    def _generate_interview(self, char: Character, question: str = "") -> ActionResult:
        """
        Stateful multi-turn interview.
        Uses NPCResponder (LLM) when attached; deterministic fallback otherwise.
        Lying is injected from ground-truth flags — the LLM does not decide it.
        """
        if not question:
            question = "Where were you at the time of the murder?"
        cid = char.id
        if cid not in self._interview_histories:
            self._interview_histories[cid] = []
        history = self._interview_histories[cid]

        if self._npc_responder is not None:
            response = self._npc_responder.respond(char, self._state, question, history)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": response})
            return ActionResult(True, f'{char.full_name}: "{response}"')

        return self._template_interview(char, question, history)

    def _template_interview(self, char: Character, question: str, history: list[dict]) -> ActionResult:
        """Deterministic fallback used when no LLM responder is attached."""
        parts = [f'You ask {char.full_name}: "{question}"']
        if char.is_culprit:
            parts.append(f"{char.full_name} seems evasive and avoids answering directly.")
            if char.has_alibi:
                parts.append(f'After a pause they say: "{char.alibi_details}"')
        elif char.has_alibi:
            parts.append(f'{char.full_name} says: "{char.alibi_details}"')
        else:
            parts.append(f"{char.full_name} says they cannot recall anything specific.")
        for rel in char.relationships:
            target = self._state.characters.get(rel.target_id)
            if target:
                if rel.sentiment < -0.3:
                    parts.append(f"They tense up when {target.full_name} is mentioned.")
                elif rel.sentiment > 0.5:
                    parts.append(f"They speak warmly of {target.full_name}.")
        response_text = " ".join(parts[1:])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response_text})
        return ActionResult(True, " ".join(parts))


    def _handle_search(self, **_: Any) -> ActionResult:
        """Active search: can find hidden evidence with some probability."""
        loc = self.get_current_location()
        if loc is None:
            return ActionResult(False, "No current location.")
        found: list[str] = []
        parts = [f"You conduct a thorough search of the {loc.name}.", "You discover:"]
        for eid, ev in self._state.evidence.items():
            if ev.location_id != loc.id:
                continue
            if ev.state == EvidenceState.DESTROYED:
                continue
            if eid in self._discovered_evidence:
                continue
            # Discovery probability = 1 - discovery_difficulty (+ bonus for active search)
            prob = 1.0 - ev.discovery_difficulty + 0.3   # active search bonus
            prob = max(0.1, min(1.0, prob))
            if ev.state == EvidenceState.HIDDEN:
                prob *= 0.5   # hidden evidence harder to find
            if self._rng.random() < prob:
                found.append(eid)
                self._discovered_evidence.add(eid)
                parts.append(f"  • {ev.name}: {ev.description}")
        if not found:
            parts.append("You find nothing new of interest.")
        return ActionResult(True, "\n".join(parts), evidence_found=found)


    def _handle_accuse(self, suspect_name: str = "", weapon_name: str = "", location_name: str = "", **_: Any) -> ActionResult:
        """Final accusation. Ends the episode."""
        self.is_solved = True
        # Match suspect
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
        if self.accusation_correct:
            obs = f"CORRECT! {culprit.full_name} committed the crime with the {weapon.name} in the {murder_loc.name}."
        else:
            obs = f"INCORRECT. The true answer: {culprit.full_name if culprit else '?'} with the {weapon.name if weapon else '?'} in the {murder_loc.name if murder_loc else '?'}."
        return ActionResult(True, obs, details=details)


    def _handle_wait(self, **_: Any) -> ActionResult:
        return ActionResult(True, "You wait and observe. Time passes.") # Thong: is this a wasteful move? The agent is supposed to do something to not waste a move?


    def _handle_inventory(self, **_: Any) -> ActionResult:
        if not self._discovered_evidence:
            return ActionResult(True, "Your evidence collection is empty.")
        parts = ["Evidence collected:"]
        for eid in self._discovered_evidence:
            ev = self._state.evidence.get(eid)
            if ev:
                parts.append(f"- {ev.name} [{ev.evidence_type.name}] ({ev.state.name}): {ev.description}")
        return ActionResult(True, "\n".join(parts))


    def _handle_take(self, object_name: str = "", **_: Any) -> ActionResult:
        loc = self.get_current_location()
        if loc is None:
            return ActionResult(False, "No current location.")
        for oid in loc.objects_here:
            obj = self._state.objects.get(oid)
            if obj and obj.name.lower() == object_name.lower():
                if not obj.portable:
                    return ActionResult(False, f"The {obj.name} cannot be taken.")
                loc.objects_here.remove(oid)
                self.agent_inventory.append(oid)
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
            "characters_interviewed": list(self._interviewed_characters),
            "total_characters": len(self._state.characters),
            "steps_elapsed": self._state.current_step,
            "event_count": len(self._state.event_log),
            "action_history": self.action_history,
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
            "interviewed_characters": list(self._interviewed_characters),
            "interview_histories": self._interview_histories,
            "action_history": self.action_history,
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
        self._interviewed_characters = set(session["interviewed_characters"])
        self._interview_histories = session["interview_histories"]
        self.action_history = session["action_history"]
