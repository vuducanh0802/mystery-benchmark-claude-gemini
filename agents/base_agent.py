"""
Base agent interface, belief-state tracking, and the shared LLM transport.

Every player in the benchmark is a :class:`BaseAgent` — detective, culprit,
witness, plain NPC alike. ``BaseAgent`` provides three things to all of them:

1. The observe → think → act hooks (``initialize`` / ``decide_action`` /
   ``update_beliefs``). These are *optional*: a solver overrides them; a role
   that only answers questions (e.g. an NPC) simply doesn't.
2. ``BeliefState`` — probabilistic beliefs over suspects/weapons/locations,
   used for evaluation.
3. A single LLM transport (:meth:`chat`) plus a one-call gateway switch
   (:meth:`configure_litellm`). Pointing the whole benchmark at one LiteLLM
   proxy is therefore a single call::

       BaseAgent.configure_litellm(base_url="https://my-litellm/v1",
                                   api_key_env="LITELLM_KEY")

   After that, any agent constructed without an explicit config uses that
   gateway. Agents that never call an LLM (heuristic, oracle) carry the
   transport but never touch it — harmless.

The transport is OpenAI-compatible (LiteLLM, vLLM, OpenRouter, OpenAI all speak
it). The legacy Anthropic-native path is preserved for back-compat when an
agent is on ``provider="anthropic"`` with no ``base_url`` (i.e. plain
``ANTHROPIC_API_KEY`` usage), so existing ``--agent claude`` runs are unchanged.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-only import: a runtime import would create a cycle
    # (base_agent → mystery_world.world → npc_responder → base_agent).
    # `from __future__ import annotations` keeps the hints below as strings.
    from mystery_world.world import AgentAction


# ---------------------------------------------------------------------------
# LLM transport configuration
# ---------------------------------------------------------------------------

# Provider -> default OpenAI-compatible base URL (None = use SDK default host).
_PROVIDER_BASE_URL: dict[str, str | None] = {
    "openai": None,
    "anthropic": None,  # anthropic-native unless a base_url is supplied
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
    "litellm": None,    # base_url must be supplied explicitly
}

# Provider -> env var consulted when no explicit key is configured.
_PROVIDER_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "litellm": "OPENAI_API_KEY",
}


def _content_to_text(content: Any) -> str:
    """Normalize SDK/gateway message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part
            for part in (_content_to_text(item) for item in content)
            if part
        )
    if isinstance(content, dict):
        for key in ("text", "content", "output_text", "input_text"):
            value = content.get(key)
            if value is not None:
                return _content_to_text(value)
        import json
        return json.dumps(content, default=str)
    text = getattr(content, "text", None)
    if text is not None:
        return _content_to_text(text)
    value = getattr(content, "content", None)
    if value is not None:
        return _content_to_text(value)
    return str(content)


class LLMUnavailable(RuntimeError):
    """Raised when no usable LLM client can be constructed (missing SDK / key).

    Callers decide the fallback (e.g. the detective drops to a heuristic
    response, an NPC stays silent) — the transport never invents content.
    """


@dataclass
class LLMConfig:
    """Everything needed to reach one model behind one gateway."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    # Defaults an agent may override per call.
    max_tokens: int = 16384
    temperature: float | None = None
    seed: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    timeout_seconds: float = 180.0

    def resolved_base_url(self) -> str | None:
        if self.base_url:
            return self.base_url
        return _PROVIDER_BASE_URL.get(self.provider)

    def resolved_api_key(self) -> str:
        if self.api_key is not None:
            return self.api_key or "EMPTY"
        if self.api_key_env:
            env_names = (self.api_key_env,)
        elif self.provider == "google":
            # Both names are common. Google currently documents GEMINI_API_KEY,
            # while older benchmark scripts used GOOGLE_API_KEY.
            env_names = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
        else:
            env_name = _PROVIDER_KEY_ENV.get(self.provider, "")
            env_names = (env_name,) if env_name else ()

        for env_name in env_names:
            key = os.environ.get(env_name, "")
            if key:
                return key

        if env_names and self.provider in {
            "openai", "anthropic", "google", "openrouter",
        }:
            joined = " or ".join(env_names)
            raise LLMUnavailable(
                f"required API key environment variable is unset: {joined}"
            )
        return "EMPTY"  # vLLM / keyless-proxy convention


# ---------------------------------------------------------------------------
# Belief state
# ---------------------------------------------------------------------------

@dataclass
class BeliefState:
    """
    Probabilistic belief state maintained by the agent.

    For evaluation we compare these distributions against the ground truth
    at every step to measure *belief accuracy*.
    """

    # Probability distributions over IDs
    suspect_probs: dict[str, float] = field(default_factory=dict)
    weapon_probs: dict[str, float] = field(default_factory=dict)
    location_probs: dict[str, float] = field(default_factory=dict)

    # Structured knowledge (for symbolic agents)
    known_facts: list[str] = field(default_factory=list)
    eliminated_suspects: set[str] = field(default_factory=set)
    eliminated_weapons: set[str] = field(default_factory=set)
    eliminated_locations: set[str] = field(default_factory=set)

    # Free-form reasoning trace
    reasoning_trace: list[str] = field(default_factory=list)

    def normalize(self) -> None:
        """Normalize probability distributions to sum to 1."""
        for d in (self.suspect_probs, self.weapon_probs, self.location_probs):
            total = sum(d.values())
            if total > 0:
                for k in d:
                    d[k] /= total

    def top_suspect(self) -> str | None:
        if not self.suspect_probs:
            return None
        return max(self.suspect_probs, key=self.suspect_probs.get)

    def top_weapon(self) -> str | None:
        if not self.weapon_probs:
            return None
        return max(self.weapon_probs, key=self.weapon_probs.get)

    def top_location(self) -> str | None:
        if not self.location_probs:
            return None
        return max(self.location_probs, key=self.location_probs.get)

    def snapshot(self) -> dict[str, Any]:
        return {
            "suspect_probs": dict(self.suspect_probs),
            "weapon_probs": dict(self.weapon_probs),
            "location_probs": dict(self.location_probs),
            "eliminated_suspects": list(self.eliminated_suspects),
            "known_facts": list(self.known_facts),
            "reasoning_trace": list(self.reasoning_trace[-5:])  # last 5
        }


# ---------------------------------------------------------------------------
# Base agent (game interface + shared LLM transport)
# ---------------------------------------------------------------------------

class BaseAgent:
    """Base class for every player in the mystery (detective, NPC, …).

    Subclasses get: belief/history bookkeeping, a lazily-built shared-shape
    LLM client with a single :meth:`chat` entry point, and the class-level
    :meth:`configure_litellm` switch that repoints *all* agents at one gateway.

    ``initialize`` / ``decide_action`` / ``update_beliefs`` are optional hooks:
    a mystery-solving agent overrides all three; a role that only answers
    questions (e.g. an NPC) overrides none and exposes its own surface.
    """

    # Set by configure_litellm(); takes precedence over per-instance config.
    _GLOBAL_OVERRIDE: LLMConfig | None = None

    # Short identifier for logging/telemetry; subclasses override.
    role_name: str = "agent"

    def __init__(self, agent_id: str = "agent", config: LLMConfig | None = None):
        self.agent_id = agent_id
        self.belief_state = BeliefState()
        self.observation_history: list[str] = []
        self.action_history: list[dict[str, Any]] = []
        self.total_tokens_used: int = 0  # for token cost metric
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0

        # LLM transport. The global gateway (if configured) wins for transport
        # fields; the agent keeps its own model unless the override pins one.
        base = config or LLMConfig()
        ov = BaseAgent._GLOBAL_OVERRIDE
        if ov is not None:
            base = replace(
                base,
                provider=ov.provider,
                base_url=ov.base_url,
                api_key=ov.api_key,
                api_key_env=ov.api_key_env,
                model=ov.model or base.model,
            )
        self.config = base
        self._client: Any = None
        self._client_kind: str | None = None  # "anthropic" | "openai"

    # ------------------------------------------------------------------
    # Single-point gateway configuration
    # ------------------------------------------------------------------
    @classmethod
    def configure_litellm(
        cls,
        base_url: str,
        *,
        api_key: str | None = None,
        api_key_env: str | None = None,
        model: str | None = None,
    ) -> None:
        """Repoint every agent at one LiteLLM (OpenAI-compatible) gateway.
        Call once at process start; affects agents built afterwards."""
        BaseAgent._GLOBAL_OVERRIDE = LLMConfig(
            provider="litellm",
            model=model or "",
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
        )

    @classmethod
    def clear_litellm(cls) -> None:
        BaseAgent._GLOBAL_OVERRIDE = None

    # ------------------------------------------------------------------
    # Game interface (optional hooks)
    # ------------------------------------------------------------------
    def initialize(self, observation: str) -> None:
        """Receive the initial briefing and set up internal state."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement initialize()"
        )

    def decide_action(self, observation: str) -> tuple[AgentAction, dict[str, str]]:
        """Given the latest observation, decide the next action.

        Returns ``(action, kwargs)`` where ``kwargs`` are the action parameters
        (e.g. target_location, character_name).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement decide_action()"
        )

    def update_beliefs(self, observation: str) -> None:
        """Update internal belief state based on new observation."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement update_beliefs()"
        )

    def record_action(self, action: AgentAction, kwargs: dict, observation: str) -> None:
        self.observation_history.append(observation)
        self.action_history.append({
            "action": action.name,
            "kwargs": kwargs,
        })

    def get_belief_snapshot(self) -> dict[str, Any]:
        return self.belief_state.snapshot()

    # ------------------------------------------------------------------
    # LLM transport
    # ------------------------------------------------------------------
    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        cfg = self.config
        base_url = cfg.resolved_base_url()
        # Anthropic-native only when explicitly anthropic AND no OpenAI-compatible
        # base_url (LiteLLM/OpenRouter/etc. all go through the openai client).
        if cfg.provider == "anthropic" and not base_url:
            try:
                import anthropic
            except ImportError as exc:
                raise LLMUnavailable("anthropic SDK not installed") from exc
            key = cfg.resolved_api_key()
            kwargs: dict[str, Any] = {
                "max_retries": 0,
                "timeout": cfg.timeout_seconds,
            }
            if key and key != "EMPTY":
                kwargs["api_key"] = key
            self._client = anthropic.Anthropic(**kwargs)
            self._client_kind = "anthropic"
            return
        try:
            import openai
        except ImportError as exc:
            raise LLMUnavailable("openai SDK not installed") from exc
        kwargs = {
            "api_key": cfg.resolved_api_key(),
            "max_retries": 0,
            "timeout": cfg.timeout_seconds,
        }
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        self._client_kind = "openai"

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status in {408, 409, 425, 429}:
            return True
        if isinstance(status, int) and status >= 500:
            return True
        name = type(exc).__name__.lower()
        return any(
            marker in name
            for marker in ("timeout", "connection", "ratelimit", "overloaded")
        )

    def chat(
        self,
        system: str,
        messages: str | list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        """Run one completion. Returns (text, total_tokens).

        ``messages`` may be a bare user string or an OpenAI-style message list.
        Raises :class:`LLMUnavailable` if no client can be built; network/API
        errors propagate so the caller can choose its own fallback.
        """
        self._ensure_client()
        cfg = self.config
        mt = max_tokens if max_tokens is not None else cfg.max_tokens
        if isinstance(messages, str):
            msg_list = [{"role": "user", "content": messages}]
        else:
            msg_list = list(messages)

        for attempt in range(cfg.max_retries + 1):
            try:
                if self._client_kind == "anthropic":
                    resp = self._client.messages.create(
                        model=cfg.model,
                        max_tokens=mt,
                        system=system,
                        messages=msg_list,
                    )
                    text = _content_to_text(resp.content)
                    input_tokens = int(resp.usage.input_tokens)
                    output_tokens = int(resp.usage.output_tokens)
                else:
                    body = dict(cfg.extra_body)
                    if extra_body:
                        body.update(extra_body)
                    temp = temperature if temperature is not None else cfg.temperature
                    call_kwargs: dict[str, Any] = {
                        "model": cfg.model,
                        "max_tokens": mt,
                        "messages": [{"role": "system", "content": system}] + msg_list,
                    }
                    if temp is not None:
                        call_kwargs["temperature"] = temp
                    if body:
                        call_kwargs["extra_body"] = body
                    resp = self._client.chat.completions.create(**call_kwargs)
                    text = _content_to_text(resp.choices[0].message.content)
                    usage = resp.usage
                    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

                if not text.strip():
                    raise ValueError(f"empty completion from model {cfg.model!r}")
                if input_tokens + output_tokens <= 0:
                    raise ValueError(
                        f"model {cfg.model!r} returned no token usage; "
                        "the episode cannot be included in API-backed results"
                    )
                self.last_input_tokens = input_tokens
                self.last_output_tokens = output_tokens
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                return text, input_tokens + output_tokens
            except Exception as exc:
                if attempt >= cfg.max_retries or not self._is_transient_error(exc):
                    raise
                time.sleep(cfg.retry_backoff_seconds * (2 ** attempt))

        raise AssertionError("unreachable completion retry state")


def configure_litellm(
    base_url: str,
    *,
    api_key: str | None = None,
    api_key_env: str | None = None,
    model: str | None = None,
) -> None:
    """Module-level convenience for :meth:`BaseAgent.configure_litellm`."""
    BaseAgent.configure_litellm(
        base_url, api_key=api_key, api_key_env=api_key_env, model=model
    )


class LLMClient(BaseAgent):
    """Backward-compatible transport adapter.

    Pre-refactor code (``SymbolicAgent``) constructs
    ``LLMClient(provider=..., model=...)`` and calls ``.complete(system,
    user) -> (text, tokens)`` with a graceful no-API fallback. That contract is
    preserved verbatim so those call sites need no changes.
    """

    role_name = "llm_client"

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-6",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        super().__init__(agent_id="llm_client", config=LLMConfig(
            provider=provider, model=model, base_url=base_url,
            api_key=api_key, api_key_env=api_key_env,
        ))

    def complete(self, system: str, user: str) -> tuple[str, int]:
        try:
            return self.chat(system, user)
        except LLMUnavailable:
            return self._heuristic_response(), 0
        except Exception as e:  # noqa: BLE001 — preserve legacy error envelope
            import json
            return json.dumps({
                "reasoning": f"API error: {e}",
                "beliefs": {},
                "action": "EXAMINE_LOCATION",
                "action_args": {},
            }), 0

    @staticmethod
    def _heuristic_response() -> str:
        import json
        return json.dumps({
            "reasoning": "No LLM available; using heuristic fallback.",
            "beliefs": {
                "top_suspect": None, "suspect_confidence": 0.0,
                "top_weapon": None, "weapon_confidence": 0.0,
                "top_location": None, "location_confidence": 0.0,
                "eliminated_suspects": [], "new_facts": [],
            },
            "action": "EXAMINE_LOCATION",
            "action_args": {},
        })
