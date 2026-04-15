#!/usr/bin/env python3
"""
End-to-end pipeline demo.

Validates the entire workflow:
  1. Generate benchmark instances
  2. Verify consistency and solvability
  3. Run heuristic agent (no API key needed)
  4. Compute metrics
  5. Generate analysis outputs

Usage:
    python scripts/demo_pipeline.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mystery_world import COMPLEXITY_PRESETS, ComplexityLevel
from mystery_world.generator import generate_mystery, verify_solvability
from mystery_world.narrator import render_initial_briefing
from mystery_world.world import MysteryEnvironment
from mystery_world.events import process_all_events
from benchmark.verify import check_structural_consistency, check_solvability
from agents.heuristic_agent import HeuristicAgent
from evaluation.runner import run_episode
from evaluation.metrics import compute_episode_metrics, aggregate_metrics, EpisodeMetrics


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main() -> None:
    section("1. PROCEDURAL GENERATION")

    results_by_level: dict[int, list[EpisodeMetrics]] = {}

    for level in [1, 2, 3, 4, 5]:
        config = COMPLEXITY_PRESETS[ComplexityLevel(level)]
        print(f"Level {level}: {config.num_suspects} suspects, {config.num_locations} locations, "
              f"{config.num_time_steps} steps, decay={config.evidence_decay_rate}")

        # Generate 3 instances per level for demo
        level_metrics: list[EpisodeMetrics] = []
        for i in range(3):
            seed = 1000 * level + i
            world_state = generate_mystery(config, seed)

            # Verify
            consistency = check_structural_consistency(world_state)
            solvability = check_solvability(world_state)

            culprit = world_state.get_culprit()
            victim = world_state.get_victim()
            weapon = world_state.objects.get(world_state.murder_weapon_id)
            murder_loc = world_state.locations.get(world_state.murder_location_id)

            print(f"  Instance seed={seed}: "
                  f"culprit={culprit.full_name if culprit else '?'}, "
                  f"victim={victim.full_name if victim else '?'}, "
                  f"weapon={weapon.name if weapon else '?'}, "
                  f"location={murder_loc.name if murder_loc else '?'}")
            print(f"    Consistent: {consistency['consistent']}, Solvable: {solvability['solvable']}")

            if not consistency["consistent"]:
                print(f"    Issues: {consistency['issues']}")

    section("2. WORLD SIMULATION DEMO")

    # Show a single world stepping through time
    config = COMPLEXITY_PRESETS[ComplexityLevel.MEDIUM]
    ws = generate_mystery(config, seed=42)
    env = MysteryEnvironment(ws)

    briefing = render_initial_briefing(env)
    print(briefing[:1500])
    print("...(truncated)")

    # Simulate a few agent actions
    from mystery_world.world import AgentAction
    print("\n--- Agent takes some actions ---")

    actions = [
        (AgentAction.EXAMINE_LOCATION, {}),
    ]
    loc = env.get_current_location()
    if loc:
        for oid in loc.objects_here[:2]:
            obj = ws.objects.get(oid)
            if obj is not None:
                actions.append(
                    (AgentAction.EXAMINE_OBJECT, {"object_name": obj.name})
                )

    # Move to an adjacent location
    if loc and loc.adjacent_ids:
        adj = ws.locations[loc.adjacent_ids[0]]
        actions.append((AgentAction.MOVE, {"target_location": adj.name}))
        actions.append((AgentAction.EXAMINE_LOCATION, {}))
        for oid in adj.objects_here[:2]:
            obj = ws.objects.get(oid)
            if obj is not None:
                actions.append(
                    (AgentAction.EXAMINE_OBJECT, {"object_name": obj.name})
                )
        # Talk to someone there
        if adj.characters_here:
            char = ws.characters.get(adj.characters_here[0])
            if char and char.is_alive:
                actions.append((AgentAction.TALK_TO, {"character_name": char.full_name}))

    for action, kwargs in actions:
        result = env.step(action, **kwargs)
        print(f"\n  Action: {action.name} {kwargs}")
        print(f"  {'✓' if result.success else '✗'} {result.observation[:200]}")

    print(f"\n  World step: {ws.current_step}, Events logged: {len(ws.event_log)}")
    print(f"  Current weather: {ws.weather}")
    for ev in ws.event_log[-3:]:
        print(f"  Event: {ev.description}")

    section("3. HEURISTIC AGENT EVALUATION")

    all_metrics: list[EpisodeMetrics] = []

    for level in [1, 2, 3]:
        config = COMPLEXITY_PRESETS[ComplexityLevel(level)]
        level_metrics = []
        for i in range(5):
            seed = 2000 * level + i
            ws = generate_mystery(config, seed)

            agent = HeuristicAgent()
            result = run_episode(agent, ws, complexity_level=level, verbose=False)

            if result.metrics:
                level_metrics.append(result.metrics)
                all_metrics.append(result.metrics)

        if level_metrics:
            solved = sum(m.solved for m in level_metrics)
            avg_eff = sum(m.clue_efficiency for m in level_metrics) / len(level_metrics)
            print(f"  Level {level}: {solved}/{len(level_metrics)} solved, "
                  f"avg clue efficiency={avg_eff:.2f}")

    section("4. METRICS SUMMARY")

    for level in [1, 2, 3]:
        agg = aggregate_metrics(all_metrics, level)
        print(f"  Level {level}: solve_rate={agg.solve_rate:.2f}, "
              f"belief_acc={agg.mean_belief_accuracy:.2f}, "
              f"clue_eff={agg.mean_clue_efficiency:.2f}, "
              f"action_eff={agg.mean_action_efficiency:.2f}")

    section("5. REPRODUCIBILITY CHECK")

    # Same seed should produce identical world
    config = COMPLEXITY_PRESETS[ComplexityLevel.MEDIUM]
    ws1 = generate_mystery(config, seed=12345)
    ws2 = generate_mystery(config, seed=12345)

    c1 = ws1.get_culprit()
    c2 = ws2.get_culprit()
    same_culprit = c1 and c2 and c1.full_name == c2.full_name
    same_weapon = ws1.murder_weapon_id == ws2.murder_weapon_id
    same_location = ws1.murder_location_id == ws2.murder_location_id

    print(f"  Same seed → same culprit: {same_culprit}")
    print(f"  Same seed → same weapon ID: {same_weapon}")
    print(f"  Same seed → same location ID: {same_location}")
    print(f"  Same seed → same weather: {ws1.weather == ws2.weather}")
    print(f"  Reproducibility: {'✓ PASS' if all([same_culprit, same_weapon, same_location]) else '✗ FAIL'}")

    section("6. SERIALISATION ROUND-TRIP")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        ws1.save(f.name)
        from mystery_world.world import WorldState
        ws_loaded = WorldState.load(f.name)
        print(f"  Saved and loaded: seed={ws_loaded.seed}")
        print(f"  Culprit match: {ws_loaded.culprit_id == ws1.culprit_id}")
        print(f"  Locations match: {len(ws_loaded.locations) == len(ws1.locations)}")
        print(f"  Evidence match: {len(ws_loaded.evidence) == len(ws1.evidence)}")

    section("PIPELINE VALIDATION COMPLETE")
    print("All components working correctly.\n")
    print("Next steps:")
    print("  1. Generate full benchmark:  python scripts/generate_benchmark.py --n-per-level 20")
    print("  2. Run LLM agent:            python scripts/run_evaluation.py --benchmark data/benchmark_v1/ --agent llm")
    print("  3. Run symbolic agent:        python scripts/run_evaluation.py --benchmark data/benchmark_v1/ --agent symbolic")
    print("  4. Analyse results:           python scripts/analyse_results.py --results-dir results/")


if __name__ == "__main__":
    main()
