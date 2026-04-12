"""
Base agent interface and belief-state tracking.

All agents implement ``BaseAgent`` which defines the observe → think → act loop.
The ``BeliefState`` tracks the agent's probabilistic beliefs over suspects,
weapons, and locations, enabling evaluation of reasoning quality.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from mystery_world.world import AgentAction, MysteryEnvironment


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
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Abstract base class for all mystery-solving agents."""

    def __init__(self, agent_id: str = "agent"):
        self.agent_id = agent_id
        self.belief_state = BeliefState()
        self.observation_history: list[str] = []
        self.action_history: list[dict[str, Any]] = []
        self.total_tokens_used: int = 0  # for token cost metric

    @abstractmethod
    def initialize(self, observation: str) -> None:
        """Receive the initial briefing and set up internal state."""
        ...


    @abstractmethod
    def decide_action(self, observation: str) -> tuple[AgentAction, dict[str, str]]:
        """
        Given the latest observation, decide the next action.

        Returns
        -------
        action : AgentAction
        kwargs : dict
            Action parameters (e.g. target_location, character_name).
        """
        ...

    
    @abstractmethod
    def update_beliefs(self, observation: str) -> None:
        """Update internal belief state based on new observation."""
        ...
    


    def record_action(self, action: AgentAction, kwargs: dict, observation: str) -> None:
        self.observation_history.append(observation)
        self.action_history.append({
            "action": action.name,
            "kwargs": kwargs,
        })

    
    def get_belief_snapshot(self) -> dict[str, Any]:
        return self.belief_state.snapshot()
