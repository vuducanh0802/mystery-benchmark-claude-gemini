"""
Benchmark execution harness

Runs agent on generated mystery instances, collecting per-step belief
snapshots and final metrics. Supports parallel execution and checkpointing.
"""

from __future__ import annotations

import json
import structlog
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from evaluation.metrics import EpisodeMetrics, compute_episode_metrics
from mystery_world.entities import CharacterRole
from mystery_world.narrator import (
    render_character_summary,
    render_evidence_summary,
    render_initial_briefing,
    render_step_observation,
)
from mystery_world.world import AgentAction, MysteryEnvironment, WorldState

logger = structlog.get_logger()


@dataclass
class EpisodeResult:
    """Full result of running one agent on one mystery instance."""
    instance_id: str = ""
    seed: int = 0
    complexity_level: int = 1
    metrics: EpisodeMetrics | None = None
    episode_summary: dict[str, Any] = field(default_factory=dict)
    belief_trace: list[dict[str, Any]] = field(default_factory=list)
    action_trace: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "seed": self.seed,
            "complexity_level": self.complexity_level,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "belief_trace": self.belief_trace,
            "action_trace": self.action_trace,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }


def run_episode(
    agent: BaseAgent,
    world_state: WorldState,
    complexity_level: int = 1,
    verbose: bool = False,
    npc_responder=None,
):
    """
    Run a single-agent-vs-environment episode.

    Parameters
    ----------
    agent: BaseAgent
        The agent to evaluate.
    world_state: WorldState
        A fully generated mystery world.
    complexity_level: int
        For tagging metrics.
    verbose: bool
        Print step-by-step output
    
    Returns
    -------
    EpisodeResult
    """
    result = EpisodeResult(
        instance_id=f"seed_{world_state.seed}",
        seed=world_state.seed,
        complexity_level=complexity_level,
    )
    t0 = time.time()

    try:
        env = MysteryEnvironment(world_state)
        if npc_responder is not None:
            env.set_npc_responder(npc_responder)
        briefing = render_initial_briefing(env)

        if verbose:
            print(briefing)
            print("=" * 60)
        
        agent.initialize(env, briefing)

        # Collect belief snapshot at step 0
        result.belief_trace.append(agent.get_belief_snapshot())

        # Main loop
        max_steps = world_state.config.max_agent_actions + 5   # safety margin
        for step_idx in range(max_steps):
            if env.is_solved:
                break

            if env.budget_remaining <= 0:
                # Force accusation
                action = AgentAction.ACCUSE
                bs = agent.belief_state
                kwargs = {
                    "suspect_name": bs.top_suspect() or "",
                    "weapon_name": bs.top_weapon() or "",
                    "location_name": bs.top_location() or "",
                }
            else:
                # Get current observation context
                obs_context = env.observe_location()
                # Add evidence and character summaries periodically
                if step_idx % 3 == 0:
                    obs_context += "\n" + render_evidence_summary(env)
                    obs_context += "\n" + render_character_summary(env)

                # Agent decides
                action, kwargs = agent.decide_action(obs_context)
            
            # Execute action
            action_result = env.step(action, **kwargs)
            observation = render_step_observation(env, action_result.observation)

            if verbose:
                print(f"\n[Step {step_idx + 1}] Action: {action.name} {kwargs}")
                print(f"Result: {observation[:300]}")
            
            # Record
            agent.record_action(action, kwargs, observation)
            agent.update_beliefs(observation)

            result.belief_trace.append(agent.get_belief_snapshot())
            result.action_trace.append({
                "step": step_idx,
                "action": action.name,
                "kwargs": kwargs,
                "success": action_result.success,
                "observation_preview": observation[:200],
            })
        
        # Compute metrics
        episode_summary = env.get_episode_summary()
        result.episode_summary = episode_summary

        # Build ground truth for metrics
        culprit = world_state.get_culprit()
        weapon = world_state.objects.get(world_state.murder_weapon_id)
        murder_loc = world_state.locations.get(world_state.murder_location_id)
        ground_truth = {
            "culprit_name": culprit.full_name if culprit else "",
            "weapon_name": weapon.name if weapon else "",
            "location_name": murder_loc.name if murder_loc else "",
        }

        result.metrics = compute_episode_metrics(
            episode_summary=episode_summary,
            belief_snapshots=result.belief_trace,
            ground_truth=ground_truth,
            total_tokens=agent.total_tokens_used,
            complexity_level=complexity_level,
        )
    except Exception as e:
        logger.exception(f"Error in episode seed={world_state.seed}")
        result.error = str(e)
    
    result.elapsed_seconds = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_benchmark(
    agent_factory,  # Callable[[], BaseAgent]
    instances: list[tuple[WorldState, int]],  # (world_state, complexity_level)
    output_dir: str | Path,
    verbose: bool = False,
    npc_responder=None,
):
    """
    Run a full benchmark suite.

    Parameters
    ----------
    agent_factory: callable
        Function that returns a fresh BaseAgent instance for each episode
    instances: list of (WorldState, complexity_level)
        Generated benchmark instances.
    output_dir: str or Path
        Directory to save per-episode results.
    verbose: bool
        Print progress
    
    Returns:
    -------
    list[EpisodeResult]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[EpisodeResult] = []

    for i, (ws, level) in enumerate(instances):
        logger.info(f"Running instance {i+1}/{len(instances)} (seed={ws.seed}, level={level})")
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Instance {i + 1}/{len(instances)} — Seed: {ws.seed}, Level: {level}")
            print(f"{'=' * 60}")
        
        agent = agent_factory()
        result = run_episode(agent, ws, complexity_level=level, verbose=verbose, npc_responder=npc_responder)
        results.append(result)

        # Save individual result
        result_path = output_dir / f"episode_{ws.seed}.json"
        result_path.write_text(json.dumps(result.to_dict(), indent=2))
    
    # Save aggregate results
    all_metrics = [r.metrics.to_dict() for r in results if r.metrics]
    (output_dir / "all_metrics.json").write_text(json.dumps(all_metrics, indent=2))

    return results
