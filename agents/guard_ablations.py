"""Controlled removals from the existing exposure-bias guard.

All three variants inherit the full guard's prompt, observation history, ledger,
parsing, and trace recording. Only action interception changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from agents.llm_agent import BiasGuardedLLMDetectiveAgent
from mystery_world.world import AgentAction


ABLATION_VERSION = "exposure-bias-guard-ablations-v1"


@dataclass(frozen=True)
class GuardVariant:
    redirect_talk: bool
    redirect_object: bool
    accusation_mode: str
    prompt: str = "exposure-bias-guard-v1"
    ledger: bool = True
    accusation_budget_cutoff: int = 5

    def metadata(self, max_accuse_blocks: int = 2) -> dict[str, Any]:
        return {**asdict(self), "max_accuse_blocks": max_accuse_blocks}


VARIANTS = {
    "full_minus_accuse": GuardVariant(True, True, "off"),
    "delay_control": GuardVariant(False, False, "fixed_delay"),
    "prompt_ledger": GuardVariant(False, False, "off"),
}


class GuardAblationDetectiveAgent(BiasGuardedLLMDetectiveAgent):
    def __init__(self, *args: Any, variant: str, **kwargs: Any) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown guard ablation: {variant!r}")
        super().__init__(*args, **kwargs)
        self.variant = variant

    def _guard_action(
        self,
        action: AgentAction,
        action_args: dict[str, Any],
        observation: str,
        budget: int,
    ) -> tuple[AgentAction, dict[str, Any]]:
        policy = VARIANTS[self.variant]
        if action == AgentAction.ACCUSE:
            if (
                policy.accusation_mode == "fixed_delay"
                and budget > policy.accusation_budget_cutoff
                and self._blocked_accusations < self.max_accuse_blocks
            ):
                self._blocked_accusations += 1
                self._last_guard_feedback = (
                    "delay control: postponed accusation without checking evidence "
                    f"({self._blocked_accusations}/{self.max_accuse_blocks})"
                )
                return self._fallback_investigation_action(observation)
            return action, action_args

        if action == AgentAction.TALK_TO and policy.redirect_talk:
            return super()._guard_action(action, action_args, observation, budget)
        if action == AgentAction.EXAMINE_OBJECT and policy.redirect_object:
            return super()._guard_action(action, action_args, observation, budget)
        return action, action_args
