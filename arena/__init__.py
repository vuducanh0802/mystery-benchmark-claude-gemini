"""Arena utilities for cross-role MysteryArena evaluation."""

from arena.aggregate import aggregate_matches, load_matches, write_outputs
from arena.trueskill import compute_role_trueskill
from arena.metrics import match_from_episode, match_from_trajectory
from arena.roster import ModelSpec, get_model, make_culprit_agent, make_detective_agent

__all__ = [
    "ModelSpec",
    "aggregate_matches",
    "compute_role_trueskill",
    "get_model",
    "load_matches",
    "make_culprit_agent",
    "make_detective_agent",
    "match_from_episode",
    "match_from_trajectory",
    "write_outputs",
]
