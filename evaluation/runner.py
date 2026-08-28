"""
Benchmark execution harness

Runs detective agents on generated mystery instances, collecting per-step belief
snapshots and final metrics. Supports parallel execution and checkpointing.
"""

from __future__ import annotations

import json
import structlog
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from agents.base_agent import BaseAgent
from evaluation.metrics import EpisodeMetrics, compute_episode_metrics
from evaluation.trajectory import TrajectoryWriter, world_state_hash
from mystery_world.entities import CharacterRole
from mystery_world.narrator import (
    render_culprit_briefing,
    render_character_summary,
    render_evidence_summary,
    render_initial_briefing,
    render_step_observation,
)
from mystery_world.world import AgentAction, MysteryEnvironment, WorldState

logger = structlog.get_logger()


@dataclass
class EpisodeResult:
    """Full result of running one detective on one mystery instance."""
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
    detective_agent: BaseAgent | None = None,
    world_state: WorldState | None = None,
    complexity_level: int = 1,
    verbose: bool = False,
    npc_responder=None,
    culprit_agent: BaseAgent | None = None,
    trajectory_writer: TrajectoryWriter | None = None,
    *,
    agent: BaseAgent | None = None,
    step_callback: Callable[[dict[str, Any]], None] | None = None,
):
    """
    Run a detective-vs-environment episode.

    Parameters
    ----------
    detective_agent: BaseAgent
        The detective agent to evaluate.
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
    if detective_agent is None:
        detective_agent = agent
    if detective_agent is None:
        raise ValueError("run_episode requires detective_agent")
    if world_state is None:
        raise ValueError("run_episode requires world_state")

    result = EpisodeResult(
        instance_id=f"seed_{world_state.seed}",
        seed=world_state.seed,
        complexity_level=complexity_level,
    )
    t0 = time.time()

    def _notify_step(payload: dict[str, Any]) -> None:
        if step_callback is None:
            return
        try:
            step_callback(payload)
        except Exception:
            logger.exception("Step progress callback failed")

    try:
        env = MysteryEnvironment(world_state)
        if npc_responder is not None:
            env.set_npc_responder(npc_responder)
        if culprit_agent is not None:
            env.enable_free_culprit()
        briefing = render_initial_briefing(env)

        if verbose:
            print(briefing)
            print("=" * 60)
        
        detective_agent.initialize(env, briefing)
        if culprit_agent is not None:
            culprit_agent.initialize(env, render_culprit_briefing(env))

        # Collect belief snapshot at step 0
        result.belief_trace.append(detective_agent.get_belief_snapshot())

        # Main loop
        max_steps = world_state.config.max_agent_actions + 5   # safety margin
        for step_idx in range(max_steps):
            if env.is_solved:
                break

            model_called = env.budget_remaining > 0
            if not model_called:
                # Force accusation
                action = AgentAction.ACCUSE
                bs = detective_agent.belief_state
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

                # Detective decides
                action, kwargs = detective_agent.decide_action(obs_context)
            
            # Capture observation passed INTO the detective for this step (for replay)
            input_obs = (
                obs_context
                if model_called
                else "[budget exhausted; forcing accusation]"
            )

            # Execute action
            action_result = env.step(action, **kwargs)
            observation = render_step_observation(env, action_result.observation)

            if verbose:
                print(f"\n[Step {step_idx + 1}] Action: {action.name} {kwargs}")
                print(f"Result: {observation[:300]}")

            # Record
            detective_agent.record_action(action, kwargs, observation)
            detective_agent.update_beliefs(observation)

            result.belief_trace.append(detective_agent.get_belief_snapshot())
            result.action_trace.append({
                "step": step_idx,
                "actor_id": "detective",
                "role": "detective",
                "action": action.name,
                "kwargs": kwargs,
                "success": action_result.success,
                "observation_preview": observation[:200],
            })

            if trajectory_writer is not None:
                trajectory_writer.write_step(
                    step=step_idx,
                    action=action.name,
                    action_kwargs=kwargs,
                    observation=input_obs,
                    model_response=(
                        getattr(detective_agent, "last_raw_response", None)
                        if model_called else None
                    ),
                    result_observation=observation,
                    success=action_result.success,
                    post_state_hash=world_state_hash(world_state),
                    model_called=model_called,
                    input_tokens=(
                        detective_agent.last_input_tokens if model_called else 0
                    ),
                    output_tokens=(
                        detective_agent.last_output_tokens if model_called else 0
                    ),
                    proposed_action=(
                        getattr(detective_agent, "last_proposed_action", action.name)
                        if model_called else action.name
                    ),
                    proposed_action_kwargs=(
                        getattr(detective_agent, "last_proposed_action_args", kwargs)
                        if model_called else kwargs
                    ),
                    guard_intervention=(
                        getattr(detective_agent, "last_guard_intervention", None)
                        if model_called else None
                    ),
                )
            _notify_step({
                "step": step_idx,
                "actor_id": "detective",
                "role": "detective",
                "action": action.name,
                "success": action_result.success,
            })

            if culprit_agent is not None and not env.is_solved:
                culprit_id = world_state.culprit_id
                culprit_obs = env.observe_location(culprit_id)
                culprit_action, culprit_kwargs = culprit_agent.decide_action(culprit_obs)
                culprit_result = env.step_for_actor(
                    culprit_id, culprit_action, **culprit_kwargs,
                )
                culprit_observation = render_step_observation(
                    env, culprit_result.observation, actor_id=culprit_id,
                )
                culprit_agent.record_action(
                    culprit_action, culprit_kwargs, culprit_observation,
                )
                try:
                    culprit_agent.update_beliefs(culprit_observation)
                except NotImplementedError:
                    pass
                result.action_trace.append({
                    "step": step_idx,
                    "actor_id": culprit_id,
                    "role": "culprit",
                    "action": culprit_action.name,
                    "kwargs": culprit_kwargs,
                    "success": culprit_result.success,
                    "observation_preview": culprit_observation[:200],
                })

                if trajectory_writer is not None:
                    trajectory_writer.write_step(
                        step=step_idx,
                        actor_id=culprit_id,
                        role="culprit",
                        action=culprit_action.name,
                        action_kwargs=culprit_kwargs,
                        observation=culprit_obs,
                        model_response=getattr(culprit_agent, "last_raw_response", None),
                        result_observation=culprit_observation,
                        success=culprit_result.success,
                        post_state_hash=world_state_hash(world_state),
                        model_called=True,
                        input_tokens=getattr(culprit_agent, "last_input_tokens", 0),
                        output_tokens=getattr(culprit_agent, "last_output_tokens", 0),
                        proposed_action=getattr(
                            culprit_agent, "last_proposed_action", culprit_action.name,
                        ),
                        proposed_action_kwargs=getattr(
                            culprit_agent, "last_proposed_action_args", culprit_kwargs,
                        ),
                    )
                _notify_step({
                    "step": step_idx,
                    "actor_id": culprit_id,
                    "role": "culprit",
                    "action": culprit_action.name,
                    "success": culprit_result.success,
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
            total_tokens=detective_agent.total_tokens_used,
            complexity_level=complexity_level,
            input_tokens=detective_agent.total_input_tokens,
            output_tokens=detective_agent.total_output_tokens,
        )
    except Exception as e:
        logger.exception(f"Error in episode seed={world_state.seed}")
        result.error = f"{type(e).__name__}: {e}"
    
    result.elapsed_seconds = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_benchmark(
    detective_agent_factory=None,  # Callable[[], BaseAgent]
    instances: list[tuple[WorldState, int]] | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = False,
    npc_responder=None,
    culprit_agent_factory=None,
    trajectory_dir: str | Path | None = None,
    trajectory_meta: dict | None = None,
    skip_existing: bool = False,
    *,
    agent_factory=None,
):
    """
    Run a full benchmark suite.

    Parameters
    ----------
    detective_agent_factory: callable
        Function that returns a fresh detective BaseAgent instance for each episode
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
    if detective_agent_factory is None:
        detective_agent_factory = agent_factory
    if detective_agent_factory is None:
        raise ValueError("run_benchmark requires detective_agent_factory")
    if instances is None:
        raise ValueError("run_benchmark requires instances")
    if output_dir is None:
        raise ValueError("run_benchmark requires output_dir")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = Path(trajectory_dir) if trajectory_dir else None
    if traj_dir is not None:
        traj_dir.mkdir(parents=True, exist_ok=True)
    meta = trajectory_meta or {}

    results: list[EpisodeResult] = []

    for i, (ws, level) in enumerate(instances):
        traj_path = traj_dir / f"level_{level}_seed_{ws.seed}.jsonl" if traj_dir else None
        if skip_existing and traj_path is not None and traj_path.exists():
            logger.info(f"Skipping seed={ws.seed} level={level} (trajectory exists)")
            continue

        logger.info(f"Running instance {i+1}/{len(instances)} (seed={ws.seed}, level={level})")
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Instance {i + 1}/{len(instances)} — Seed: {ws.seed}, Level: {level}")
            print(f"{'=' * 60}")

        detective_agent = detective_agent_factory()
        culprit_agent = culprit_agent_factory() if culprit_agent_factory is not None else None
        if culprit_agent is not None:
            ws.config = replace(ws.config, free_culprit_actions=True)
        writer = None
        if traj_path is not None:
            writer = TrajectoryWriter(traj_path)
            writer.write_header(
                state=ws,
                level=str(level),
                agent=meta.get("detective_agent", meta.get("agent", "unknown")),
                model=meta.get("detective_model", meta.get("model")),
                provider=meta.get("detective_provider", meta.get("provider")),
                npc_provider=meta.get("npc_provider"),
                npc_model=meta.get("npc_model"),
                npc_seed=meta.get("npc_seed"),
                instance_id=f"seed_{ws.seed}",
            )
        try:
            result = run_episode(
                detective_agent, ws, complexity_level=level, verbose=verbose,
                npc_responder=npc_responder, culprit_agent=culprit_agent,
                trajectory_writer=writer,
            )
            if writer is not None:
                writer.write_footer(
                    episode_summary=result.episode_summary,
                    metrics=result.metrics.to_dict() if result.metrics else None,
                    elapsed_seconds=result.elapsed_seconds,
                    error=result.error,
                )
        finally:
            if writer is not None:
                writer.close()
        results.append(result)

        # Save individual result
        result_path = output_dir / f"episode_{ws.seed}.json"
        result_path.write_text(json.dumps(result.to_dict(), indent=2))
    
    # Save aggregate results
    all_metrics = [r.metrics.to_dict() for r in results if r.metrics]
    (output_dir / "all_metrics.json").write_text(json.dumps(all_metrics, indent=2))

    return results
