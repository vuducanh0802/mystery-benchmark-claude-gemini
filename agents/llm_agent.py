"""
Pure LLM agent: uses only natural-language prompting for reasoning.

This agent sends the full observation history to an LLM and asks it to
(a) reason about the case and (b) choose the next action. No external
symbolic tools are used - this is the baseline for RQ1.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agents.base_agent import BaseAgent, BeliefState, LLMClient, LLMConfig
from mystery_world.world import AgentAction, MysteryEnvironment

# Re-exported for backward compatibility: `from agents.llm_agent import LLMClient`
# still works (used by agents.symbolic_agent). The implementation now lives in
# agents.base_agent alongside the shared LLM transport.
__all__ = ["LLMDetectiveAgent", "LLMAgent", "LLMClient", "SYSTEM_PROMPT"]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a detective AI solving a procedurally generated murder mystery.
You must determine WHO committed the murder, WHAT weapon was used, and WHERE it occurred.

At each step you receive an observation and must output EXACTLY this JSON:
{
  "reasoning": "<your step-by-step reasoning>",
  "beliefs": {
    "top_suspect": "<name or null>",
    "suspect_confidence": <0.0-1.0>,
    "top_weapon": "<name or null>",
    "weapon_confidence": <0.0-1.0>,
    "top_location": "<name or null>",
    "location_confidence": <0.0-1.0>,
    "eliminated_suspects": ["<name>", ...],
    "new_facts": ["<fact>", ...]
  },
  "action": "<ACTION_NAME>",
  "action_args": {"<key>": "<value>", ...}
}

Actions: MOVE (target_location), EXAMINE_LOCATION, EXAMINE_OBJECT (object_name),
TALK_TO (character_name, question), TAKE_OBJECT (object_name),
CHECK_INVENTORY, WAIT, ACCUSE (see schema below).
Note: TALK_TO is a stateful interview — each call asks ONE focused question. You can ask the same character multiple times.
When the observation includes Available targets, use exact names from those lists
for MOVE, TALK_TO, EXAMINE_OBJECT, and TAKE_OBJECT. Do not name a character or
object that is not listed in your current location.

ACCUSE action_args schema (all keys required for full credit):
{
  "suspect_name": "<name>",
  "weapon_name": "<name>",
  "location_name": "<room>",
  "suspect_weapon_evidence": ["<evidence_id>", ...],
  "weapon_victim_evidence": ["<evidence_id>", ...],
  "suspect_room_evidence": ["<evidence_id>", ...],
  "alibi_contradiction": {
    "claimed_location": "<where the culprit said they were>",
    "claimed_time": "<time they claimed>",
    "contradiction_evidence": ["<evidence_id>", ...]
  },
  "eliminations": {
    "<innocent_name>": {"evidence_id": "<id>", "corroborator": "<witness_name>"}
  }
}

Be methodical. Gather evidence before accusing. Track alibis and eliminate suspects.
"""

def _build_user_message(briefing: str, history: list[str], current_obs: str, budget: int) -> str:
    parts = [briefing, ""]
    if history:
        parts.append("=== PREVIOUS OBSERVATIONS ===")
        # Keep last N observations to avoid context overflow
        for obs in history[-10:]:
            parts.append(obs)
            parts.append("---")
    parts.append(f"=== CURRENT OBSERVATION (budget remaining: {budget}) ===")
    parts.append(current_obs)
    parts.append("")
    parts.append("Respond with the JSON object described in the system prompt.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

class LLMDetectiveAgent(BaseAgent):
    """
    Pure LLM prompting agent (the Detective role).

    Strategy: send all observations to the LLM, parse structured JSON output
    containing reasoning, belief updates, and the chosen action.

    Its model transport is :class:`BaseAgent`'s shared one — the same transport
    every NPC uses — so a single ``BaseAgent.configure_litellm(...)`` repoints
    the whole benchmark at one LiteLLM gateway.
    """

    role_name = "detective"

    def __init__(
        self,
        agent_id: str = "llm_agent",
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
    ):
        BaseAgent.__init__(self, agent_id, config=LLMConfig(
            provider=provider or "anthropic", model=model,
            base_url=base_url, api_key=api_key, api_key_env=api_key_env,
        ))
        self.briefing: str = ""
        self._env: MysteryEnvironment | None = None
        self._suspect_names: dict[str, str] = {}   # id -> name
        self._weapon_names: dict[str, str] = {}   # id -> name
        self._location_names: dict[str, str] = {}   # id -> name


    
    def initialize(self, env: MysteryEnvironment, briefing: str) -> None:
        self.briefing = briefing
        self._env = env
        state = env.state

        # Build name lookups for belief tracking
        from mystery_world.entities import CharacterRole
        for cid, char in state.characters.items():
            if CharacterRole.SUSPECT in char.roles:
                self._suspect_names[cid] = char.full_name
                self.belief_state.suspect_probs[char.full_name] = 1.0 / max(1, len([
                    c for c in state.characters.values() if CharacterRole.SUSPECT in c.roles
                ]))
        
        for oid, obj in state.objects.items():
            if obj.is_weapon:
                self._weapon_names[oid] = obj.name
                self.belief_state.weapon_probs[obj.name] = 1.0 / max(1, state.config.num_weapons)
        for lid, loc in state.locations.items():
            self._location_names[lid] = loc.name
            self.belief_state.location_probs[loc.name] = 1.0 / max(1, len(state.locations))

    
    def _complete(self, system: str, user: str) -> tuple[str, int]:
        """Unified-transport completion with the legacy graceful fallback:
        no client / missing key → heuristic JSON; API error → error envelope.
        Behaviour is byte-identical to the retired LLMClient.complete()."""
        from agents.base_agent import LLMUnavailable
        try:
            return self.chat(system, user)
        except LLMUnavailable:
            return LLMClient._heuristic_response(), 0
        except Exception as e:  # noqa: BLE001 — preserve legacy error envelope
            return json.dumps({
                "reasoning": f"API error: {e}",
                "beliefs": {},
                "action": "EXAMINE_LOCATION",
                "action_args": {},
            }), 0

    def decide_action(self, observation: str) -> tuple[AgentAction, dict[str, str]]:
        budget = self._env.budget_remaining if self._env else 0
        user_msg = _build_user_message(
            self.briefing, self.observation_history, observation, budget,
        )
        response_text, tokens = self._complete(SYSTEM_PROMPT, user_msg)
        self.total_tokens_used += tokens
        self.last_raw_response = response_text

        # Parse JSON from response
        parsed = self._parse_response(response_text)

        # Update beliefs from parsed output
        beliefs = parsed.get("beliefs", {})
        self._apply_belief_update(beliefs)

        # Extract action
        action_str = parsed.get("action", "EXAMINE_LOCATION").upper()
        action_args = parsed.get("action_args", {})

        try:
            action = AgentAction[action_str]
        except KeyError:
            action = AgentAction.EXAMINE_LOCATION
            action_args = {}
        
        return action, action_args

    
    def update_beliefs(self, observation: str) -> None:
        # Beliefs are updated inside decide_action from LLM output
        pass

    
    def _parse_response(self, text: str) -> dict[str, Any]:
        """Extract JSON from LLM response, handling markdown fences."""
        # Try to find JSON block
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {"action": "EXAMINE_LOCATION", "action_args": {}, "beliefs": {}, "reasoning": text}

    
    def _apply_belief_update(self, beliefs: dict[str, Any]) -> None:
        bs = self.belief_state
        if beliefs.get("top_suspect"):
            name = beliefs["top_suspect"]
            conf = beliefs.get("suspect_confidence", 0.5)
            if name in bs.suspect_probs:
                # Shift probability mass toward the top suspect
                for k in bs.suspect_probs:
                    bs.suspect_probs[k] *= (1 - conf)
                bs.suspect_probs[name] = conf
                bs.normalize()

        if beliefs.get("top_weapon"):
            name = beliefs["top_weapon"]
            conf = beliefs.get("weapon_confidence", 0.5)
            if name in bs.weapon_probs:
                for k in bs.weapon_probs:
                    bs.weapon_probs[k] *= (1 - conf)
                bs.weapon_probs[name] = conf
                bs.normalize()

        if beliefs.get("top_location"):
            name = beliefs["top_location"]
            conf = beliefs.get("location_confidence", 0.5)
            if name in bs.location_probs:
                for k in bs.location_probs:
                    bs.location_probs[k] *= (1 - conf)
                bs.location_probs[name] = conf
                bs.normalize()
        
        for s in beliefs.get("eliminated_suspects", []):
            bs.eliminated_suspects.add(s)
        
        for f in beliefs.get("new_facts", []):
            bs.known_facts.append(f)
        
        if beliefs.get("reasoning"):
            bs.reasoning_trace.append(str(beliefs["reasoning"])[:500])


LLMAgent = LLMDetectiveAgent
