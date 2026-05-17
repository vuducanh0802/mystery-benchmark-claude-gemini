"""
Unified LLM transport for every model-backed role in the benchmark.

Historically the detective (``LLMAgent`` / ``SymbolicAgent``) and the NPCs
(``NPCResponder``) each carried their own, slightly different LLM client. This
module collapses both onto a single abstract base, :class:`LLMRole`, whose
``chat()`` method is *the* place an LLM call happens. Every role — Detective,
Corporate, NPC — inherits :class:`LLMRole`, so pointing the whole benchmark at
one gateway (e.g. a LiteLLM proxy) is a single call:

    LLMRole.configure_litellm(base_url="https://my-litellm/v1",
                              api_key_env="LITELLM_KEY")

After that, any role constructed without an explicit config uses that gateway.

The transport is OpenAI-compatible (LiteLLM, vLLM, OpenRouter, OpenAI all speak
it). The legacy Anthropic-native path is preserved for back-compat when a role
is on ``provider="anthropic"`` with no ``base_url`` (i.e. plain
``ANTHROPIC_API_KEY`` usage), so existing ``--agent claude`` runs are unchanged.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any

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


class LLMUnavailable(RuntimeError):
    """Raised when no usable LLM client can be constructed (missing SDK / key).

    Callers decide the fallback (e.g. the detective drops to a heuristic
    response, an NPC stays silent) — the transport never invents content.
    """


@dataclass
class LLMConfig:
    """Everything needed to reach one model behind one gateway."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    # Defaults a role may override per call.
    max_tokens: int = 16384
    temperature: float | None = None
    seed: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    def resolved_base_url(self) -> str | None:
        if self.base_url:
            return self.base_url
        return _PROVIDER_BASE_URL.get(self.provider)

    def resolved_api_key(self) -> str:
        if self.api_key is not None:
            return self.api_key or "EMPTY"
        env_name = self.api_key_env or _PROVIDER_KEY_ENV.get(self.provider, "")
        if env_name:
            key = os.environ.get(env_name, "")
            if key:
                return key
            if self.api_key_env:  # explicitly requested env must exist — fail loud
                raise LLMUnavailable(
                    f"env var {self.api_key_env} is unset or empty. Export it in "
                    f"the shell that launches the run, e.g. `export "
                    f"{self.api_key_env}=...` (otherwise calls 401 silently)."
                )
        return "EMPTY"  # vLLM / keyless-proxy convention


class LLMRole(ABC):
    """Abstract base for every LLM-backed role (Detective, Corporate, NPC).

    Subclasses get a lazily-built, shared-shape client and a single
    :meth:`chat` entry point. The class-level override set by
    :meth:`configure_litellm` lets one call repoint *all* roles at one gateway.
    """

    # Set by configure_litellm(); takes precedence over per-instance config.
    _GLOBAL_OVERRIDE: LLMConfig | None = None

    def __init__(self, config: LLMConfig | None = None) -> None:
        base = config or LLMConfig()
        ov = LLMRole._GLOBAL_OVERRIDE
        if ov is not None:
            # Global gateway wins for transport fields; keep the role's model
            # unless the override pins one.
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
        """Repoint every LLMRole subclass at one LiteLLM (OpenAI-compatible)
        gateway. Call once at process start; affects roles built afterwards."""
        cls._GLOBAL_OVERRIDE = LLMConfig(
            provider="litellm",
            model=model or "",
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
        )

    @classmethod
    def clear_litellm(cls) -> None:
        cls._GLOBAL_OVERRIDE = None

    @property
    @abstractmethod
    def role_name(self) -> str:
        """Short identifier for logging/telemetry (e.g. 'detective', 'npc')."""
        ...

    # ------------------------------------------------------------------
    # Transport
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
            kwargs: dict[str, Any] = {}
            if key and key != "EMPTY":
                kwargs["api_key"] = key
            self._client = anthropic.Anthropic(**kwargs)
            self._client_kind = "anthropic"
            return
        try:
            import openai
        except ImportError as exc:
            raise LLMUnavailable("openai SDK not installed") from exc
        kwargs = {"api_key": cfg.resolved_api_key()}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        self._client_kind = "openai"

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

        if self._client_kind == "anthropic":
            resp = self._client.messages.create(
                model=cfg.model,
                max_tokens=mt,
                system=system,
                messages=msg_list,
            )
            text = resp.content[0].text
            tokens = resp.usage.input_tokens + resp.usage.output_tokens
            return text, tokens

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
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        return text, tokens


def configure_litellm(
    base_url: str,
    *,
    api_key: str | None = None,
    api_key_env: str | None = None,
    model: str | None = None,
) -> None:
    """Module-level convenience for :meth:`LLMRole.configure_litellm`."""
    LLMRole.configure_litellm(
        base_url, api_key=api_key, api_key_env=api_key_env, model=model
    )


class LLMClient(LLMRole):
    """Backward-compatible adapter.

    Pre-refactor code (``LLMAgent``, ``SymbolicAgent``) constructed
    ``LLMClient(provider=..., model=...)`` and called ``.complete(system,
    user) -> (text, tokens)`` with a graceful no-API fallback. That contract is
    preserved verbatim so those call sites need no changes.
    """

    role_name = "llm_client"  # type: ignore[assignment]

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        super().__init__(LLMConfig(
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
