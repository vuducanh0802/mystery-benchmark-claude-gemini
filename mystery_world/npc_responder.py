"""
NPC response engine for stateful interview interactions.

Lying is injected into the system prompt from ground-truth flags.
The LLM has no agency over whether to lie — that decision comes from WorldState.
Uses any OpenAI-compatible endpoint (vLLM, Together AI, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mystery_world.entities import Character
    from mystery_world.world import WorldState

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
    alibi_line = (
        f"Your alibi: {char.alibi_details}"
        if char.has_alibi
        else "You have no specific alibi for the critical time window."
    )
    lying_block = _lying_instruction(char, state)

    return f"""You are {char.full_name}, a {char.personality} {role_label} being questioned by a detective about a recent murder.

WHAT YOU KNOW:
{alibi_line}
Things you witnessed:
{_witnessed_summary(char, state)}
Your relationships:
{_relationship_summary(char, state)}

BEHAVIOURAL RULES:
- Respond only from your own perspective. Never invent facts you could not know.
- Be consistent with everything you have already said in this conversation.
- Keep responses to 1-3 sentences. Stay in character at all times.
- Do not volunteer information the detective has not asked about.
{lying_block}"""


class NPCResponder:
    """
    Generates NPC interview responses using a local or remote LLM.

    Parameters
    ----------
    base_url : str
        OpenAI-compatible API base URL.
        For vLLM: "http://localhost:8000/v1"
        For Together AI: "https://api.together.xyz/v1"
    model : str
        Model name as served by the endpoint.
    seed : int
        Fixed seed for reproducibility (passed via extra_body).
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_NPC_URL,
        model: str = _DEFAULT_NPC_MODEL,
        seed: int = _FIXED_SEED,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.seed = seed
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import openai
            self._client = openai.OpenAI(base_url=self.base_url, api_key="EMPTY")
        except ImportError as exc:
            raise RuntimeError("openai package required: pip install openai") from exc

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
        self._ensure_client()
        system = build_npc_system_prompt(char, state)
        messages = list(history) + [{"role": "user", "content": question}]
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}] + messages,
                max_tokens=256,
                temperature=0.7,
                extra_body={"seed": self.seed},
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            return f"{char.full_name} stares at you silently and says nothing. (Error: {exc})"