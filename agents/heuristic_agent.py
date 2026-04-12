"""
Heuristic baseline agent.

A rule-based agent that follows a fixed investigation strategy:
    1. Visit all locations and search for evidence
    2. Interview all available characters
    3. Eliminate suspects with verified alibis
    4. Accuse the remaining suspect

Useful as a non-LLM baseline (lower bound) and for testing the pipeline.
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent, BeliefState
from mystery_world.entities import CharacterRole
from mystery_world.world import AgentAction, MysteryEnvironment


class HeuristicAgent(BaseAgent):
    """Rule-based detective that follows a systematic investigation plan."""
    
    def __init__(self, agent_id: str = "heuristic_agent"):
        super().__init__(agent_id)
        self._env: MysteryEnvironment | None = None
        self._plan: list[tuple[AgentAction, dict[str, str]]] = []
        self._plan_idx: int = 0
        self._all_suspects: list[str] = []
        self._all_weapons: list[str] = []
        self._all_locations: list[str] = []
        self._location_names_to_ids: dict[str, str] = {}

    
    def initialize(self, env: MysteryEnvironment, briefing: str) -> None:
        self._env = env
        state = env.state

        # Build entity name maps
        for cid, char in state.characters.items():
            if CharacterRole.SUSPECT in char.roles:
                self._all_suspects.append(char.full_name)
                self.belief_state.suspect_probs[char.full_name] = 1.0
            if char.is_alive:
                pass  # tracked for interviewing
        
        for oid, obj in state.objects.items():
            if obj.is_weapon:
                self._all_weapons.append(obj.name)
                self.belief_state.weapon_probs[obj.name] = 1.0
        
        for lid, loc in state.locations.items():
            self._all_locations.append(loc.name)
            self.belief_state.location_probs[loc.name] = 1.0
            self._location_names_to_ids[loc.name] = lid
        
        self.belief_state.normalize()

        # Build investigation plan
        self._build_plan(env)

    
    def _build_plan(self, env: MysteryEnvironment) -> None:
        """Build a systematic investigation plan."""
        state = env.state
        plan: list[tuple[AgentAction, dict[str, str]]] = []

        # Phase 1: Examine starting location
        plan.append((AgentAction.EXAMINE_LOCATION, {}))
        plan.append((AgentAction.SEARCH_FOR_EVIDENCE, {}))

        # Phase 2: Visit every other location, search, and interview anyone there
        start_loc = env.get_current_location()
        visited = {start_loc.id} if start_loc else set()

        for lid, loc in state.locations.items():
            if lid in visited:
                continue

            plan.append((AgentAction.MOVE, {"target_location": loc.name}))
            plan.append((AgentAction.EXAMINE_LOCATION, {}))
            plan.append((AgentAction.SEARCH_FOR_EVIDENCE, {}))
            # Interview characters at this location
            for cid in loc.characters_here:
                char = state.characters.get(cid)
                if char and char.is_alive:
                    plan.append((AgentAction.TALK_TO, {
                        "character_name": char.full_name,
                        "question": "Where were you at the time of the murder? Do you have an alibi?",
                    }))
                    plan.append((AgentAction.TALK_TO, {
                        "character_name": char.full_name,
                        "question": "Did you see or hear anything suspicious that evening?",
                    }))
            visited.add(lid)
        
        # Phase 3: Check inventory
        plan.append((AgentAction.CHECK_INVENTORY, {}))

        # Phase 4: Accuse (will be filled in at runtime based on beliefs)
        plan.append((AgentAction.ACCUSE, {}))

        self._plan = plan
    

    def decide_action(self, observation: str) -> tuple[AgentAction, dict[str, str]]:
        # Update beliefs from observation
        self._update_from_observation(observation)

        if self._plan_idx >= len(self._plan):
            # Out of planned actions, make accusation
            return self._make_accusation()
        
        action, kwargs = self._plan[self._plan_idx]
        self._plan_idx += 1

        # If this is the accusation placeholder, fill in from beliefs
        if action == AgentAction.ACCUSE:
            return self._make_accusation()
        
        return action, kwargs

    
    def _make_accusation(self) -> tuple[AgentAction, dict[str, str]]:
        bs = self.belief_state
        suspect = bs.top_suspect() or (self._all_suspects[0] if self._all_suspects else "")
        weapon = bs.top_weapon() or (self._all_weapons[0] if self._all_weapons else "")
        location = bs.top_location() or (self._all_locations[0] if self._all_locations else "")
        return AgentAction.ACCUSE, {
            "suspect_name": suspect,
            "weapon_name": weapon,
            "location_name": location,
        }


    def _update_from_observation(self, observation: str) -> None:
        """Simple keyword-based belief update."""
        obs_lower = observation.lower()
        bs = self.belief_state

        # If someone has a verified alibi, reduce their probability
        for suspect in list(bs.suspect_probs.keys()):
            suspect_lower = suspect.lower()
            if suspect_lower in obs_lower:
                if "can confirm" in obs_lower or "corroborated" in obs_lower:
                    bs.suspect_probs[suspect] *= 0.2
                    bs.eliminated_suspects.add(suspect)
                if "evasive" in obs_lower or "no alibi" in obs_lower:
                    bs.suspect_probs[suspect] *= 2.0
        
        # Evidence keywords boost relevant beliefs
        if "culprit" in obs_lower or "traces" in obs_lower or "pointing to" in obs_lower:
            # Try to identify which suspect the evidence points to
            for suspect in bs.suspect_probs:
                if suspect.lower() in obs_lower:
                    bs.suspect_probs[suspect] *= 1.5
        
        # Weapon mentions
        if "used recently" in obs_lower or "murder weapon" in obs_lower:
            for weapon in bs.weapon_probs:
                if weapon.lower() in obs_lower:
                    bs.weapon_probs[weapon] *= 3.0
        
        bs.normalize()

    
    def update_beliefs(self, observation: str) -> None:
        self._update_from_observation(observation)
