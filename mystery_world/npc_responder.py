"""
NPC response engine for stateful interview interactions.

The prompt provides role facts and local knowledge without prescribing a
truthfulness or deception strategy. NPCs can respond in character from that
context.
Uses any OpenAI-compatible endpoint (vLLM, Together AI, etc.).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from agents.base_agent import BaseAgent, LLMConfig, LLMUnavailable

if TYPE_CHECKING:
    from mystery_world.entities import Character
    from mystery_world.world import WorldState

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. Qwen3)."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    # Also strip any leftover "Thinking Process:" preamble style output
    text = re.sub(r"(?i)^thinking process:.*?(?=\n\S)", "", text, flags=re.DOTALL)
    return text.strip()


_DEFAULT_NPC_URL = "http://localhost:8000/v1"
_DEFAULT_NPC_MODEL = "Qwen/Qwen2.5-27B-Instruct"
_FIXED_SEED = 42


def _witnessed_summary(char: "Character", state: "WorldState") -> str:
    lines = []
    for eid in char.witnessed_events:
        for te in state.ground_truth_timeline:
            key = f"{te.step}_{te.actor_id}_{te.action}"
            if (key == eid or te.action == eid) and te.is_public:
                lines.append(f"- {te.details}")
                break
    return "\n".join(lines) if lines else "- Nothing notable that you can clearly recall."


def _relationship_summary(char: "Character", state: "WorldState") -> str:
    lines = []
    for rel in char.relationships:
        target = state.characters.get(rel.target_id)
        if not target:
            continue
        if rel.sentiment > 0.3:
            feeling = f"friendly with"
        elif rel.sentiment < -0.3:
            feeling = f"hostile toward"
        else:
            feeling = f"neutral toward"
        lines.append(f"- You are {feeling} {target.full_name} ({rel.kind}).")
    return "\n".join(lines) if lines else "- No strong connections to speak of."


def _corroboration_summary(char: "Character", state: "WorldState") -> str:                                                                                                             
    """                                                                                                                                                                                
    If this character is a genuine alibi corroborator for someone,
    tell them that fact so they can confirm it truthfully.                                                                                                                             
    False-alibi context is handled separately by _private_context.
    """                                                                                                                                                                                
    lines = []                          
    for other in state.characters.values():                                                                                                                                            
        if (                                                                                                                                                                           
            other.alibi_corroborator_id == char.id                                                                                                                                     
            and other.alibi_corroboration_is_genuine                                                                                                                                   
            and other.is_alive                            
        ):                                                                                                                                                                             
            lines.append(
                f"- You were with {other.full_name} during the relevant time window "                                                                                                  
                f"and can confirm their whereabouts."     
            )
    return "\n".join(lines) if lines else ""                                                                                                                                           


def _private_context(char: "Character", state: "WorldState") -> str:
    """Return private role facts without prescribing a response strategy."""
    culprit = state.get_culprit()
    parts: list[str] = []

    if char.is_culprit:
        victim = state.get_victim()
        weapon = state.objects.get(state.murder_weapon_id)
        murder_loc = state.locations.get(state.murder_location_id)
        facts = [
            f"- You committed the murder of {victim.full_name if victim else 'the victim'}.",
            f"- The murder weapon was {weapon.name if weapon else 'unknown'}.",
            f"- The murder happened in the {murder_loc.name if murder_loc else 'unknown location'}.",
        ]
        if state.motive:
            facts.append(f"- Your motive: {state.motive}.")
        if char.alibi_details:
            facts.append(f"- Your existing alibi claim: {char.alibi_details}")
        facts.append("- Your objective: avoid being identified as the culprit.")
        parts.extend(facts)
    elif (
        culprit is not None
        and culprit.alibi_corroborator_id == char.id
        and not culprit.alibi_corroboration_is_genuine
    ):
        parts.append(
            f"- You previously agreed to provide a false alibi for {culprit.full_name}."
        )

    return "\n".join(parts) if parts else "- No private facts beyond your own memories and relationships."


def build_npc_system_prompt(
    char: "Character",
    state: "WorldState",
    questioner_name: str = "the detective",
) -> str:
    from mystery_world.entities import CharacterRole
    role_label = "suspect" if CharacterRole.SUSPECT in char.roles else "witness"

    current_loc = state.locations.get(char.location_id)
    current_loc_name = current_loc.name if current_loc else "unknown"

    # Check if this character is a genuine alibi corroborator for another suspect.
    # If so, their whereabouts line must reflect being with that suspect — not "alone".
    corroborated_suspect = None
    for other in state.characters.values():
        if (
            other.alibi_corroborator_id == char.id
            and other.alibi_corroboration_is_genuine
            and other.is_alive
        ):
            corroborated_suspect = other
            break

    if char.has_alibi:
        alibi_line = f"Your whereabouts: {char.alibi_details}"
    elif corroborated_suspect is not None:
        suspect_loc = state.locations.get(corroborated_suspect.location_id)
        suspect_loc_name = suspect_loc.name if suspect_loc else current_loc_name
        alibi_line = (
            f"At the time of the murder you were with {corroborated_suspect.full_name} "
            f"in the {suspect_loc_name}. You can confirm they were there with you."
        )
    else:
        alibi_line = (
            f"At the time of the murder you were in the {current_loc_name}, alone. "
            f"You have no alibi and no one can confirm your whereabouts."
        )

    private_block = _private_context(char, state)

    known_locations = ", ".join(loc.name for loc in state.locations.values())
    known_people = ", ".join(
        c.full_name for c in state.characters.values() if c.id != char.id and c.is_alive
    )

    return f"""You are {char.full_name}, a {char.personality} {role_label} speaking with {questioner_name} about a recent murder.
You are {char.full_name}. You are currently in the {current_loc_name}.

WHAT YOU KNOW (these are the ONLY facts you may draw on):
Your whereabouts: {alibi_line}
Things you personally witnessed:
{_witnessed_summary(char, state)}
Your relationships:
{_relationship_summary(char, state)}

The only locations that exist: {known_locations}.
The only other people: {known_people}.

PRIVATE CONTEXT:
{private_block}

RESPONSE BOUNDARIES:
- You are {char.full_name}. Never refer to yourself in the third person.
- Use the facts, memories, relationships, and private context above as your world model.
- Do NOT invent any other names, locations, times, objects, or events.
- If you do not know or do not recall something, you can say so.
- Do not mention any room, person, or object not in the lists above.
- Be consistent with everything you have already said in this conversation.
- Keep responses to 1-3 sentences. Stay in character at all times.
"""


class NPCResponder(BaseAgent):
    """
    Generates NPC interview responses using a local or remote LLM.

    An NPC is a player too, so it is a :class:`BaseAgent` — it just answers
    questions instead of solving the case (it doesn't override the
    ``decide_action`` / ``update_beliefs`` hooks). It shares the exact same
    model transport as the detective, so one ``BaseAgent.configure_litellm(...)``
    call repoints the whole benchmark at a LiteLLM gateway.

    The public surface (``base_url``, ``model``, ``seed``, ``api_key``,
    ``api_key_env`` and ``respond(...)``) is unchanged from the pre-refactor
    version, including the loud failure on a missing explicit ``api_key_env``
    and the silent in-dialog error on transient API failures.

    Parameters
    ----------
    base_url : str
        OpenAI-compatible API base URL (vLLM, Together, LiteLLM, ...).
    model : str
        Model name as served by the endpoint.
    seed : int
        Fixed seed for reproducibility (passed via extra_body).
    """

    role_name = "npc"

    def __init__(
        self,
        base_url: str | None = _DEFAULT_NPC_URL,
        model: str = _DEFAULT_NPC_MODEL,
        seed: int = _FIXED_SEED,
        api_key: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        # Preserve the legacy "no key + no env → EMPTY" vLLM convention exactly
        # (without it, a stray OPENAI_API_KEY in the env could leak to a local
        # endpoint). An explicit api_key_env still fails loud if unset.
        if api_key is None and api_key_env is None:
            api_key = "EMPTY"
        super().__init__(agent_id="npc", config=LLMConfig(
            provider="openai", model=model, base_url=base_url,
            api_key=api_key, api_key_env=api_key_env,
            seed=seed, max_tokens=512, temperature=0.7,
        ))

    # Back-compat read-only accessors (some callers/tests inspect these).
    @property
    def base_url(self) -> str | None:
        return self.config.base_url

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def seed(self) -> int | None:
        return self.config.seed

    def respond(
        self,
        char: "Character",
        state: "WorldState",
        question: str,
        history: list[dict[str, str]],
        *,
        questioner_name: str = "the detective",
    ) -> str:
        """
        Generate the NPC's response to the detective's question.

        Parameters
        ----------
        char : Character
            The NPC being questioned.
        state : WorldState
            Full ground-truth state (used for role facts and witnessed events).
        question : str
            The detective's question.
        history : list[dict]
            Prior turns in this interview: [{"role": "user"|"assistant", "content": "..."}]
        """
        system = build_npc_system_prompt(char, state, questioner_name=questioner_name)
        messages = list(history) + [{"role": "user", "content": question}]
        extra_body: dict[str, Any] = {"seed": self.config.seed}
        # chat_template_kwargs is a vLLM-only knob (Qwen3 "thinking" toggle).
        # OpenAI / OpenRouter reject unknown args, so only send it for vLLM-style
        # endpoints (custom base_url that isn't OpenRouter).
        bu = self.config.base_url
        is_vllm = bu is not None and "openrouter.ai" not in bu
        if is_vllm:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            raw, _ = self.chat(
                system, messages, max_tokens=512, temperature=0.7,
                extra_body=extra_body,
            )
            return _strip_thinking(raw).strip()
        except LLMUnavailable:
            raise  # misconfigured key/env must fail loud, as before
        except Exception as exc:  # noqa: BLE001 — transient API error stays silent
            return f"{char.full_name} stares at you silently and says nothing. (Error: {exc})"
