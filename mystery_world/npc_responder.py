"""
NPC response engine for stateful interview interactions.

Lying is injected into the system prompt from ground-truth flags.
The LLM has no agency over whether to lie — that decision comes from WorldState.
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
    Lying corroborators are handled separately by _lying_instruction.
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


def _lying_instruction(char: "Character", state: "WorldState") -> str:
    """
    Returns ground-truth-derived instruction text telling the character HOW to lie.
    Empty string if the character should be honest.
    """
    culprit = state.get_culprit()
    parts: list[str] = []

    if char.is_culprit:
        alibi_claim = char.alibi_details or "you were alone and did not see anyone"
        parts.append(
            "IMPORTANT: You committed the murder. Never admit this under any circumstances. "
            f"If asked about your whereabouts at the time of the crime, claim: \"{alibi_claim}\". "
            "If pressed hard, deflect by expressing concern or casting vague suspicion on others. "
            "Do not contradict yourself."
        )
    elif (
        culprit is not None
        and culprit.alibi_corroborator_id == char.id
        and not culprit.alibi_corroboration_is_genuine
    ):
        parts.append(
            f"IMPORTANT: You have agreed to provide a false alibi for {culprit.full_name}. "
            f"If asked where {culprit.full_name} was, confirm they were with you. "
            "Do not admit this arrangement — act as if it is the simple truth."
        )

    return "\n".join(parts)


def build_npc_system_prompt(char: "Character", state: "WorldState") -> str:
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

    lying_block = _lying_instruction(char, state)

    known_locations = ", ".join(loc.name for loc in state.locations.values())
    known_people = ", ".join(
        c.full_name for c in state.characters.values() if c.id != char.id and c.is_alive
    )

    return f"""You are {char.full_name}, a {char.personality} {role_label} being questioned by a detective about a recent murder.
You are {char.full_name}. You are currently in the {current_loc_name}.

WHAT YOU KNOW (these are the ONLY facts you may draw on):
Your whereabouts: {alibi_line}
Things you personally witnessed:
{_witnessed_summary(char, state)}
Your relationships:
{_relationship_summary(char, state)}

The only locations that exist: {known_locations}.
The only other people: {known_people}.

STRICT RULES — follow these exactly:
- You are {char.full_name}. Never refer to yourself in the third person.
- You may ONLY state facts listed above. Do NOT invent any other names, locations, times, or events.
- If asked about something not in your knowledge above, say "I don't know" or "I don't recall" — never fabricate.
- Do not mention any room, person, or object not in the lists above.
- Be consistent with everything you have already said in this conversation.
- Keep responses to 1-3 sentences. Stay in character at all times.
- Do not volunteer information the detective has not asked about. Answer ONLY the question that was asked — nothing extra.
- LAYERED ALIBI REVEAL — this is a hard constraint:
  • When asked simply where you were (e.g. "where were you", "what were you doing at the time"), state ONLY the location and a brief activity. Do NOT mention being alone, by yourself, that no one was there, that you have no alibi, that you cannot be confirmed, or that you spoke to no one. Strip those phrases from any whereabouts text given to you above.
  • Only reveal whether you were alone or with someone if the detective explicitly asks about company, witnesses, or corroboration — e.g. "was anyone with you", "who else was there", "can anyone confirm", "do you have a witness". Until that follow-up is asked, keep the alone/no-witness detail to yourself.
  • If your whereabouts text mentions a corroborator (someone who was with you), the same rule applies: don't volunteer their name on the first "where were you" question — just give the location. Mention them only when asked about company or witnesses.
- If the detective greets you or asks an open-ended question (e.g. "tell me about yourself", "anything to share"), respond briefly without disclosing where you were at the time of the murder. Wait to be asked specifically.
- Never preemptively defend yourself by stating you were alone or had no witnesses. That is volunteering information.
{lying_block}"""


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
    ) -> str:
        """
        Generate the NPC's response to the detective's question.

        Parameters
        ----------
        char : Character
            The NPC being questioned.
        state : WorldState
            Full ground-truth state (used only for lying injection and witnessed events).
        question : str
            The detective's question.
        history : list[dict]
            Prior turns in this interview: [{"role": "user"|"assistant", "content": "..."}]
        """
        system = build_npc_system_prompt(char, state)
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