"""
Pure LLM agent: uses only natural-language prompting for reasoning.

This agent sends the full observation history to an LLM and asks it to
(a) reason about the case and (b) choose the next action. No external
symbolic tools are used - this is the baseline for RQ1.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from agents.base_agent import BaseAgent, BeliefState, LLMClient, LLMConfig
from mystery_world.world import AgentAction, MysteryEnvironment

# Re-exported for backward compatibility: `from agents.llm_agent import LLMClient`
# still works (used by agents.symbolic_agent). The implementation now lives in
# agents.base_agent alongside the shared LLM transport.
__all__ = [
    "LLMDetectiveAgent",
    "BiasGuardedLLMDetectiveAgent",
    "LLMAgent",
    "LLMClient",
    "SYSTEM_PROMPT",
]

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

BIAS_GUARDED_SYSTEM_PROMPT = SYSTEM_PROMPT + """

Additional exposure-bias control protocol:

1. Evasion is unresolved evidence, not innocence. If TALK_TO fails because a
   suspect is absent, keep that suspect active and follow up when available.
2. Normalize suspicion for exposure. Do not accuse someone because they were
   interviewed first, interviewed often, or appeared in more observations.
   Compare independent evidence per suspect.
3. Repeated examination is not weapon evidence. Re-examining an object must
   not raise weapon confidence unless it reveals a new ev_* ID or factual link.
4. Before ACCUSE, verify all three evidence edges: suspect-to-weapon,
   weapon-to-victim, and suspect-to-room. Cite observed ev_* IDs where possible;
   demeanor, talk order, and exposure counts are not evidence.
5. If an accusation is supported mainly by exposure, continue investigating.
"""

def _build_user_message(briefing: str, history: list[str], current_obs: str, budget: int) -> str:
    parts = [briefing, ""]
    if history:
        parts.append("=== PREVIOUS OBSERVATIONS ===")
        # Keep the baseline default at 10; experiments may lower it explicitly.
        window = max(0, int(os.environ.get("MYSTERY_LLM_HISTORY_WINDOW", "10")))
        for obs in (history[-window:] if window else []):
            parts.append(obs)
            parts.append("---")
    parts.append(f"=== CURRENT OBSERVATION (budget remaining: {budget}) ===")
    parts.append(current_obs)
    parts.append("")
    parts.append("Respond with the JSON object described in the system prompt.")
    return "\n".join(parts)


def _safe_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, confidence))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _stringify_belief_item(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


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
        model: str = "claude-sonnet-4-6",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        max_tokens: int | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        timeout_seconds: float = 180.0,
    ):
        BaseAgent.__init__(self, agent_id, config=LLMConfig(
            provider=provider or "anthropic", model=model,
            base_url=base_url, api_key=api_key, api_key_env=api_key_env,
            max_tokens=max_tokens or LLMConfig.max_tokens,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            timeout_seconds=timeout_seconds,
        ))
        self.briefing: str = ""
        self._env: MysteryEnvironment | None = None
        self._suspect_names: dict[str, str] = {}   # id -> name
        self._weapon_names: dict[str, str] = {}   # id -> name
        self._location_names: dict[str, str] = {}   # id -> name
        self.last_raw_response: str | None = None
        self.last_proposed_action: str | None = None
        self.last_proposed_action_args: dict[str, Any] | None = None
        self.last_guard_intervention: dict[str, Any] | None = None


    
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
        """Complete one turn without inventing a fallback action.

        Missing credentials, transport failures, empty responses, and malformed
        JSON must invalidate the episode rather than masquerade as model output.
        """
        return self.chat(system, user)

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
        action_str = str(parsed["action"]).upper()
        action_args = parsed.get("action_args", {})
        if not isinstance(action_args, dict):
            raise ValueError("model action_args must be a JSON object")

        try:
            action = AgentAction[action_str]
        except KeyError as exc:
            raise ValueError(f"unknown model action {action_str!r}") from exc

        self.last_proposed_action = action.name
        self.last_proposed_action_args = dict(action_args)
        self.last_guard_intervention = None
        
        return action, action_args

    
    def update_beliefs(self, observation: str) -> None:
        # Beliefs are updated inside decide_action from LLM output
        pass

    
    def _parse_response(self, text: str) -> dict[str, Any]:
        """Extract JSON from LLM response, handling markdown fences."""
        # Try to find JSON block
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            raise ValueError("model response did not contain a JSON object")
        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed model JSON: {exc}") from exc
        if not isinstance(parsed, dict) or not parsed.get("action"):
            raise ValueError("model JSON must contain a non-empty action")
        if not isinstance(parsed.get("beliefs", {}), dict):
            raise ValueError("model beliefs must be a JSON object")
        return parsed

    
    def _apply_belief_update(self, beliefs: dict[str, Any]) -> None:
        bs = self.belief_state
        if beliefs.get("top_suspect"):
            name = str(beliefs["top_suspect"])
            conf = _safe_confidence(beliefs.get("suspect_confidence", 0.5))
            if name in bs.suspect_probs:
                # Shift probability mass toward the top suspect
                for k in bs.suspect_probs:
                    bs.suspect_probs[k] *= (1 - conf)
                bs.suspect_probs[name] = conf
                bs.normalize()

        if beliefs.get("top_weapon"):
            name = str(beliefs["top_weapon"])
            conf = _safe_confidence(beliefs.get("weapon_confidence", 0.5))
            if name in bs.weapon_probs:
                for k in bs.weapon_probs:
                    bs.weapon_probs[k] *= (1 - conf)
                bs.weapon_probs[name] = conf
                bs.normalize()

        if beliefs.get("top_location"):
            name = str(beliefs["top_location"])
            conf = _safe_confidence(beliefs.get("location_confidence", 0.5))
            if name in bs.location_probs:
                for k in bs.location_probs:
                    bs.location_probs[k] *= (1 - conf)
                bs.location_probs[name] = conf
                bs.normalize()
        
        for s in _as_list(beliefs.get("eliminated_suspects", [])):
            bs.eliminated_suspects.add(_stringify_belief_item(s))
        
        for f in _as_list(beliefs.get("new_facts", [])):
            bs.known_facts.append(_stringify_belief_item(f))
        
        if beliefs.get("reasoning"):
            bs.reasoning_trace.append(str(beliefs["reasoning"])[:500])


class BiasGuardedLLMDetectiveAgent(LLMDetectiveAgent):
    """Detective policy that controls three exposure shortcuts.

    The guard only uses observations available to the detective. It never reads
    the hidden culprit, weapon, room, or evidence graph. It balances interview
    exposure, treats failed interviews as unresolved, discounts repeated object
    examinations, and blocks weak early accusations while budget remains.
    """

    _EVID_RE = re.compile(r"\bev_\d+\b")

    def __init__(
        self,
        agent_id: str = "llm_bias_guarded",
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-6",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        max_tokens: int | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        timeout_seconds: float = 180.0,
        max_accuse_blocks: int = 2,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            timeout_seconds=timeout_seconds,
        )
        self.max_accuse_blocks = max_accuse_blocks
        self._talk_success: dict[str, int] = {}
        self._talk_failed: dict[str, int] = {}
        self._first_talk_order: list[str] = []
        self._object_examines: dict[str, int] = {}
        self._object_no_new_evidence: dict[str, int] = {}
        self._object_evidence_ids: dict[str, set[str]] = {}
        self._location_visits: dict[str, int] = {}
        self._seen_evidence_ids: set[str] = set()
        self._blocked_accusations = 0
        self._last_guard_feedback = ""

    def initialize(self, env: MysteryEnvironment, briefing: str) -> None:
        super().initialize(env, briefing)
        for name in self._suspect_names.values():
            self._talk_success.setdefault(name, 0)
            self._talk_failed.setdefault(name, 0)

    def decide_action(self, observation: str) -> tuple[AgentAction, dict[str, Any]]:
        budget = self._env.budget_remaining if self._env else 0
        user_msg = self._build_guarded_user_message(observation, budget)
        response_text, tokens = self._complete(BIAS_GUARDED_SYSTEM_PROMPT, user_msg)
        self.total_tokens_used += tokens
        self.last_raw_response = response_text

        parsed = self._parse_response(response_text)
        self._apply_belief_update(parsed.get("beliefs", {}))

        action_str = str(parsed["action"]).upper()
        action_args = parsed.get("action_args", {})
        if not isinstance(action_args, dict):
            raise ValueError("model action_args must be a JSON object")
        try:
            proposed_action = AgentAction[action_str]
        except KeyError as exc:
            raise ValueError(f"unknown model action {action_str!r}") from exc

        self.last_proposed_action = proposed_action.name
        self.last_proposed_action_args = dict(action_args)
        self.last_guard_intervention = None
        executed_action, executed_args = self._guard_action(
            proposed_action, action_args, observation, budget,
        )
        if executed_action != proposed_action or executed_args != action_args:
            self.last_guard_intervention = {
                "reason": self._last_guard_feedback or "guard policy redirect",
                "proposed_action": proposed_action.name,
                "proposed_action_args": action_args,
                "executed_action": executed_action.name,
                "executed_action_args": executed_args,
            }
        return executed_action, executed_args

    def record_action(self, action: AgentAction, kwargs: dict, observation: str) -> None:
        super().record_action(action, kwargs, observation)
        self._update_bias_ledger(action, kwargs, observation)

    def _build_guarded_user_message(self, current_obs: str, budget: int) -> str:
        parts = [self.briefing, ""]
        if self.observation_history:
            parts.append("=== PREVIOUS OBSERVATIONS ===")
            window = max(0, int(os.environ.get("MYSTERY_LLM_HISTORY_WINDOW", "10")))
            for obs in (
                self.observation_history[-window:] if window else []
            ):
                parts.append(obs)
                parts.append("---")
        parts.extend([
            "=== BIAS CONTROL LEDGER ===",
            self._bias_ledger_text(),
            "",
            f"=== CURRENT OBSERVATION (budget remaining: {budget}) ===",
            current_obs,
            "",
            "Respond with the JSON object described in the system prompt.",
        ])
        return "\n".join(parts)

    def _bias_ledger_text(self) -> str:
        lines = ["Suspect exposure table (talk count is NOT evidence):"]
        if self._suspect_names:
            for name in sorted(self._suspect_names.values()):
                successful = self._talk_success.get(name, 0)
                failed = self._talk_failed.get(name, 0)
                if successful:
                    status = "interviewed"
                elif failed:
                    status = "unresolved_after_failed_talk"
                else:
                    status = "unseen_or_uninterviewed"
                rank = (
                    self._first_talk_order.index(name) + 1
                    if name in self._first_talk_order
                    else "-"
                )
                lines.append(
                    f"- {name}: success={successful}, failed={failed}, "
                    f"first_talk_rank={rank}, status={status}"
                )
        else:
            lines.append("- no suspect roster available")

        lines.append("Object exposure table (repetition is NOT weapon evidence):")
        if self._object_examines:
            for obj, count in sorted(
                self._object_examines.items(), key=lambda item: (-item[1], item[0]),
            )[:12]:
                evidence_ids = sorted(self._object_evidence_ids.get(obj, set()))
                ids_text = ", ".join(evidence_ids) if evidence_ids else "none"
                lines.append(
                    f"- {obj}: examines={count}, "
                    f"no_new_evidence={self._object_no_new_evidence.get(obj, 0)}, "
                    f"evidence_ids={ids_text}"
                )
        else:
            lines.append("- no object has been examined yet")

        evidence_text = (
            ", ".join(sorted(self._seen_evidence_ids))
            if self._seen_evidence_ids else "none"
        )
        lines.append(f"Evidence IDs observed so far: {evidence_text}")
        if self._last_guard_feedback:
            lines.append(f"Previous guard feedback: {self._last_guard_feedback}")
        lines.append(
            "Decision rule: accuse only from evidence, not exposure. Failed "
            "TALK_TO is unresolved; first-talk order and repeated EXAMINE_OBJECT "
            "are confounders."
        )
        return "\n".join(lines)

    def _update_bias_ledger(
        self,
        action: AgentAction,
        kwargs: dict[str, Any],
        observation: str,
    ) -> None:
        evidence_ids = set(self._EVID_RE.findall(observation or ""))
        new_ids = evidence_ids - self._seen_evidence_ids
        self._seen_evidence_ids.update(evidence_ids)

        location_match = re.search(
            r"\bYou (?:move to|are in) the ([^.]+)\.", observation or "",
        )
        if location_match:
            location = location_match.group(1).strip()
            self._location_visits[location] = self._location_visits.get(location, 0) + 1

        if action == AgentAction.TALK_TO:
            name = str(kwargs.get("character_name") or "").strip()
            if not name:
                return
            failed = "not here or cannot be spoken to" in (observation or "")
            if failed:
                self._talk_failed[name] = self._talk_failed.get(name, 0) + 1
            else:
                self._talk_success[name] = self._talk_success.get(name, 0) + 1
                if name not in self._first_talk_order:
                    self._first_talk_order.append(name)
            return

        if action == AgentAction.EXAMINE_OBJECT:
            obj = str(
                kwargs.get("object_name") or kwargs.get("name") or "",
            ).strip()
            if not obj:
                return
            self._object_examines[obj] = self._object_examines.get(obj, 0) + 1
            self._object_evidence_ids.setdefault(obj, set()).update(evidence_ids)
            if not new_ids:
                self._object_no_new_evidence[obj] = (
                    self._object_no_new_evidence.get(obj, 0) + 1
                )

    def _guard_action(
        self,
        action: AgentAction,
        action_args: dict[str, Any],
        observation: str,
        budget: int,
    ) -> tuple[AgentAction, dict[str, Any]]:
        if action == AgentAction.TALK_TO:
            redirected = self._redirect_repeated_talk(action_args, observation)
            if redirected is not None:
                return redirected

        if action == AgentAction.EXAMINE_OBJECT:
            redirected = self._redirect_repeated_object(action_args, observation)
            if redirected is not None:
                return redirected

        if action != AgentAction.ACCUSE:
            return action, action_args
        if budget <= 5 or self._blocked_accusations >= self.max_accuse_blocks:
            return action, action_args

        reasons = self._weak_accusation_reasons(action_args)
        if not reasons:
            return action, action_args
        self._blocked_accusations += 1
        self._last_guard_feedback = "; ".join(reasons)
        return self._fallback_investigation_action(observation)

    def _redirect_repeated_talk(
        self,
        action_args: dict[str, Any],
        observation: str,
    ) -> tuple[AgentAction, dict[str, Any]] | None:
        name = str(action_args.get("character_name") or "").strip()
        if not name or self._talk_success.get(name, 0) == 0:
            return None
        unseen_here = [
            candidate
            for candidate in self._available_targets(observation).get("TALK_TO", [])
            if self._talk_success.get(candidate, 0) == 0
        ]
        if not unseen_here:
            return None
        unseen_here.sort(
            key=lambda candidate: (self._talk_failed.get(candidate, 0), candidate),
        )
        self._last_guard_feedback = (
            f"redirected repeated interview of {name}; unseen visible suspect first"
        )
        return AgentAction.TALK_TO, {
            "character_name": unseen_here[0],
            "question": (
                "Where were you at the time of the murder, and can anyone "
                "corroborate that?"
            ),
        }

    def _redirect_repeated_object(
        self,
        action_args: dict[str, Any],
        observation: str,
    ) -> tuple[AgentAction, dict[str, Any]] | None:
        obj = str(
            action_args.get("object_name") or action_args.get("name") or "",
        ).strip()
        if not obj or self._object_examines.get(obj, 0) == 0:
            return None
        self._last_guard_feedback = (
            f"redirected repeated examination of {obj}; repetition is not evidence"
        )
        return self._fallback_investigation_action(observation, avoid_object=obj)

    def _weak_accusation_reasons(self, action_args: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        suspect_weapon = self._ids_from_value(
            action_args.get("suspect_weapon_evidence"),
        ) & self._seen_evidence_ids
        weapon_victim = self._ids_from_value(
            action_args.get("weapon_victim_evidence"),
        ) & self._seen_evidence_ids
        suspect_room = self._ids_from_value(
            action_args.get("suspect_room_evidence"),
        ) & self._seen_evidence_ids
        all_ids = suspect_weapon | weapon_victim | suspect_room

        if not suspect_weapon:
            reasons.append("suspect-to-weapon edge lacks observed ev_* support")
        if not weapon_victim:
            reasons.append("weapon-to-victim edge lacks observed ev_* support")
        if not suspect_room:
            reasons.append("suspect-to-room edge lacks observed ev_* support")
        if len(all_ids) < 2:
            reasons.append("fewer than two cited evidence IDs")

        accused = str(action_args.get("suspect_name") or "").strip()
        if accused and self._first_talk_order and accused == self._first_talk_order[0]:
            talk_count = (
                self._talk_success.get(accused, 0)
                + self._talk_failed.get(accused, 0)
            )
            unseen = sum(
                self._talk_success.get(name, 0) == 0
                for name in self._suspect_names.values()
            )
            if talk_count > 1 and unseen > 0 and len(all_ids) < 3:
                reasons.append(
                    "first-talk suspect has an exposure advantage but weak evidence",
                )

        weapon = str(action_args.get("weapon_name") or "").strip()
        if weapon:
            repeats = self._object_examines.get(weapon, 0)
            no_new = self._object_no_new_evidence.get(weapon, 0)
            if (
                repeats > 1
                and no_new >= repeats - 1
                and not (suspect_weapon or weapon_victim)
            ):
                reasons.append("weapon is supported only by repeated examination")
        return reasons

    def _fallback_investigation_action(
        self,
        observation: str,
        avoid_object: str | None = None,
    ) -> tuple[AgentAction, dict[str, Any]]:
        targets = self._available_targets(observation)
        unresolved = [
            name
            for name in targets.get("TALK_TO", [])
            if self._talk_success.get(name, 0) == 0
        ]
        if unresolved:
            unresolved.sort(
                key=lambda name: (
                    -self._talk_failed.get(name, 0),
                    self._talk_success.get(name, 0) + self._talk_failed.get(name, 0),
                    name,
                ),
            )
            return AgentAction.TALK_TO, {
                "character_name": unresolved[0],
                "question": (
                    "Where were you at the time of the murder, and can anyone "
                    "corroborate that?"
                ),
            }

        objects = targets.get("EXAMINE_OBJECT", [])
        if avoid_object:
            objects = [obj for obj in objects if obj != avoid_object]
        if objects:
            objects.sort(key=lambda obj: (self._object_examines.get(obj, 0), obj))
            return AgentAction.EXAMINE_OBJECT, {"object_name": objects[0]}

        moves = targets.get("MOVE", [])
        if moves:
            moves.sort(key=lambda loc: (self._location_visits.get(loc, 0), loc))
            return AgentAction.MOVE, {"target_location": moves[0]}
        return AgentAction.EXAMINE_LOCATION, {}

    @staticmethod
    def _available_targets(observation: str) -> dict[str, list[str]]:
        match = re.search(r"Available targets:\s*(.+?)(?:\n|$)", observation or "")
        if not match:
            return {}
        targets: dict[str, list[str]] = {}
        for part in match.group(1).split("|"):
            if ":" not in part:
                continue
            key, raw = part.split(":", 1)
            raw = raw.strip().rstrip(".")
            targets[key.strip().upper()] = (
                []
                if not raw or raw.lower() == "none"
                else [item.strip() for item in raw.split(",") if item.strip()]
            )
        return targets

    def _ids_from_value(self, value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            return set(self._EVID_RE.findall(value))
        if isinstance(value, list):
            return set().union(*(self._ids_from_value(item) for item in value))
        if isinstance(value, dict):
            return set().union(*(self._ids_from_value(item) for item in value.values()))
        return set(self._EVID_RE.findall(str(value)))


LLMAgent = LLMDetectiveAgent
