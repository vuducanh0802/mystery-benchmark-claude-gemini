"""Model roster and agent factories for Arena runs."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

from agents.culprit_agent import LLMCulpritAgent
from agents.detective_agent import HeuristicAgent, LLMDetectiveAgent, OracleAgent

Role = Literal["detective", "culprit"]


@dataclass(frozen=True)
class ModelSpec:
    """A model entry that can play one or both Arena roles."""

    name: str
    provider: str | None = None
    model: str | None = None
    roles: tuple[Role, ...] = ("detective", "culprit")
    kind: str = "llm"  # llm | heuristic | oracle_min | oracle_max | passive

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def can_detective(self) -> bool:
        return "detective" in self.roles

    @property
    def can_culprit(self) -> bool:
        return "culprit" in self.roles


REGISTRY: dict[str, ModelSpec] = {
    "passive": ModelSpec("passive", roles=("culprit",), kind="passive"),
    "heuristic": ModelSpec("heuristic", roles=("detective",), kind="heuristic"),
    "oracle_min": ModelSpec("oracle_min", roles=("detective",), kind="oracle_min"),
    "oracle_max": ModelSpec("oracle_max", roles=("detective",), kind="oracle_max"),
    "claude": ModelSpec("claude", provider="anthropic", model="claude-sonnet-4-6"),
    "claude-opus": ModelSpec("claude-opus", provider="anthropic", model="claude-opus-4-7"),
    "chatgpt": ModelSpec("chatgpt", provider="openai", model="gpt-4o"),
    "chatgpt-mini": ModelSpec("chatgpt-mini", provider="openai", model="gpt-4o-mini"),
    "deepseek-v4-pro": ModelSpec("deepseek-v4-pro", provider="openai", model="deepseek-v4-pro"),
    "glm-4.7": ModelSpec("glm-4.7", provider="openai", model="glm-4.7"),
    "glm-5": ModelSpec("glm-5", provider="openai", model="glm-5"),
    "glm-5.1": ModelSpec("glm-5.1", provider="openai", model="glm-5.1"),
    "gpt-5.2-codex": ModelSpec("gpt-5.2-codex", provider="openai", model="gpt-5.2-codex"),
    "gpt-5.3-codex": ModelSpec("gpt-5.3-codex", provider="openai", model="gpt-5.3-codex"),
    "gpt54": ModelSpec("gpt54", provider="openai", model="gpt-5.4"),
    "gpt-5.4": ModelSpec("gpt-5.4", provider="openai", model="gpt-5.4"),
    "gpt-5.4-ptu": ModelSpec("gpt-5.4-ptu", provider="openai", model="gpt-5.4-ptu"),
    "gpt-5.5": ModelSpec("gpt-5.5", provider="openai", model="gpt-5.5"),
    "kimi-k2": ModelSpec("kimi-k2", provider="openai", model="kimi-k2"),
    "kimi-k2.5": ModelSpec("kimi-k2.5", provider="openai", model="kimi-k2.5"),
    "minimax-m2.5": ModelSpec("minimax-m2.5", provider="openai", model="minimax-m2.5"),
    "minimax-m2.7": ModelSpec("minimax-m2.7", provider="openai", model="minimax-m2.7"),
    "gemini": ModelSpec("gemini", provider="google", model="gemini-3.6-flash"),
    "openrouter": ModelSpec("openrouter", provider="openrouter", model="qwen/qwen3.5-27b"),
}


def _safe_name(text: str) -> str:
    text = text.strip().replace("/", "_").replace(":", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "model"


def get_model(ref: str, *, role: Role | None = None) -> ModelSpec:
    """Resolve a roster name or an inline ``provider:model`` spec.

    Inline specs may optionally provide a display name:
    ``name=provider:model``.
    """
    raw = ref.strip()
    if not raw:
        raise ValueError("empty model reference")
    if raw in REGISTRY:
        spec = REGISTRY[raw]
    else:
        name = ""
        provider_model = raw
        if "=" in raw:
            name, provider_model = raw.split("=", 1)
            name = _safe_name(name)
        if ":" not in provider_model:
            known = ", ".join(sorted(REGISTRY))
            raise ValueError(
                f"Unknown model {raw!r}. Use one of [{known}] or provider:model."
            )
        provider, model = provider_model.split(":", 1)
        provider = provider.strip()
        model = model.strip()
        if not name:
            name = _safe_name(model)
        spec = ModelSpec(name=name, provider=provider, model=model)

    if role == "detective" and not spec.can_detective:
        raise ValueError(f"{spec.name!r} cannot play detective")
    if role == "culprit" and not spec.can_culprit:
        raise ValueError(f"{spec.name!r} cannot play culprit")
    return spec


def parse_model_list(spec: str, *, role: Role) -> list[ModelSpec]:
    return [get_model(part, role=role) for part in spec.split(",") if part.strip()]


def make_detective_agent(spec: ModelSpec):
    if spec.kind == "heuristic":
        return HeuristicAgent(agent_id=spec.name)
    if spec.kind == "oracle_min":
        return OracleAgent(agent_id=spec.name, mode="min_action")
    if spec.kind == "oracle_max":
        return OracleAgent(agent_id=spec.name, mode="max_score")
    if not spec.provider or not spec.model:
        raise ValueError(f"{spec.name!r} has no LLM provider/model for detective")
    return LLMDetectiveAgent(
        agent_id=spec.name,
        provider=spec.provider,
        model=spec.model,
    )


def make_culprit_agent(spec: ModelSpec):
    if spec.kind == "passive":
        return None
    if not spec.provider or not spec.model:
        raise ValueError(f"{spec.name!r} has no LLM provider/model for culprit")
    return LLMCulpritAgent(
        agent_id=spec.name,
        provider=spec.provider,
        model=spec.model,
    )
