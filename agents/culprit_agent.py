"""LLM-driven free-action culprit agent."""

from __future__ import annotations

import json
import re
from typing import Any

from agents.base_agent import BaseAgent, LLMConfig, LLMUnavailable
from mystery_world.narrator import render_culprit_briefing
from mystery_world.world import AgentAction, MysteryEnvironment


SYSTEM_PROMPT = """
You are the culprit in a procedurally generated murder mystery.
You can act freely in the same world as the detective.

At each step you receive your current observation and must output EXACTLY this JSON:
{
  "reasoning": "<brief reason for this action>",
  "action": "<ACTION_NAME>",
  "action_args": {"<key>": "<value>", ...}
}

Actions: MOVE (target_location), EXAMINE_LOCATION, EXAMINE_OBJECT (object_name),
TALK_TO (character_name, question), TAKE_OBJECT (object_name),
CHECK_INVENTORY, WAIT.
When the observation includes Available targets, use exact names from those lists
for MOVE, TALK_TO, EXAMINE_OBJECT, and TAKE_OBJECT. Do not name a character or
object that is not listed in your current location.

You cannot use ACCUSE.
"""


def _build_user_message(briefing: str, history: list[str], current_obs: str, budget: int) -> str:
    parts = [briefing, ""]
    if history:
        parts.append("=== PREVIOUS OBSERVATIONS ===")
        for obs in history[-10:]:
            parts.append(obs)
            parts.append("---")
    parts.append(f"=== CURRENT OBSERVATION (budget remaining: {budget}) ===")
    parts.append(current_obs)
    parts.append("")
    parts.append("Respond with the JSON object described in the system prompt.")
    return "\n".join(parts)


class LLMCulpritAgent(BaseAgent):
    """Culprit role that chooses environment actions from local observations."""

    role_name = "culprit"

    def __init__(
        self,
        agent_id: str = "culprit_agent",
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        super().__init__(agent_id, config=LLMConfig(
            provider=provider or "anthropic",
            model=model,
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
        ))
        self.briefing = ""
        self._env: MysteryEnvironment | None = None
        self._culprit_id = ""

    def initialize(self, env: MysteryEnvironment, briefing: str = "") -> None:
        self._env = env
        culprit = env.state.get_culprit()
        self._culprit_id = culprit.id if culprit else ""
        self.briefing = briefing or render_culprit_briefing(env)

    def decide_action(self, observation: str) -> tuple[AgentAction, dict[str, str]]:
        budget = (
            self._env.budget_remaining_for(self._culprit_id)
            if self._env and self._culprit_id
            else 0
        )
        user_msg = _build_user_message(
            self.briefing, self.observation_history, observation, budget,
        )
        response_text, tokens = self._complete(SYSTEM_PROMPT, user_msg)
        self.total_tokens_used += tokens
        self.last_raw_response = response_text

        parsed = self._parse_response(response_text)
        action_str = str(parsed.get("action", "WAIT")).upper()
        action_args = parsed.get("action_args", {})
        if not isinstance(action_args, dict):
            action_args = {}

        try:
            action = AgentAction[action_str]
        except KeyError:
            action = AgentAction.WAIT
            action_args = {}
        if action == AgentAction.ACCUSE:
            action = AgentAction.WAIT
            action_args = {}
        return action, action_args

    def update_beliefs(self, observation: str) -> None:
        return None

    def _complete(self, system: str, user: str) -> tuple[str, int]:
        try:
            return self.chat(system, user)
        except LLMUnavailable:
            return json.dumps({
                "reasoning": "No LLM available; waiting.",
                "action": "WAIT",
                "action_args": {},
            }), 0
        except Exception as exc:  # noqa: BLE001
            return json.dumps({
                "reasoning": f"API error: {exc}",
                "action": "WAIT",
                "action_args": {},
            }), 0

    def _parse_response(self, text: str) -> dict[str, Any]:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"reasoning": text, "action": "WAIT", "action_args": {}}


__all__ = ["LLMCulpritAgent", "SYSTEM_PROMPT"]
