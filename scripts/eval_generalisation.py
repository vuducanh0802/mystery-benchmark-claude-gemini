#!/usr/bin/env python3
"""
Generalisation evaluation (RQ3):
  Do agents trained/prompted on one set of narrative world rules generalise
  to novel worlds with different rules, entity distributions, and structures?

Methodology:
  - Train/prompt agent on "standard" world rules (standard asset pool)
  - Evaluate on "novel" world rules with:
    (a) Different entity distributions (exotic locations, unusual weapons)
    (b) Modified dynamics (faster evidence decay, aggressive culprit)
    (c) Structural changes (larger/smaller worlds, different graph topologies)
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mystery_world import AssetPool, ComplexityConfig, COMPLEXITY_PRESETS, ComplexityLevel
from mystery_world.generator import generate_mystery
from agents.heuristic_agent import HeuristicAgent
from evaluation.runner import run_episode
from evaluation.metrics import EpisodeMetrics, aggregate_metrics


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Novel world configurations
# ---------------------------------------------------------------------------

NOVEL_ASSET_POOL = AssetPool(
    location_templates=[
        "Abandoned Subway Station", "Rooftop Helipad", "Underground Lab",
        "Floating Casino", "Arctic Research Bunker", "Desert Oasis Camp",
        "Sunken Ballroom", "Volcanic Observatory", "Bamboo Tea House",
        "Orbital Lounge", "Midnight Bazaar", "Frozen Library",
    ],
    first_names=[
        "Zephyr", "Isolde", "Kairo", "Vesper", "Mireille", "Talon",
        "Sable", "Orin", "Calista", "Dax", "Elara", "Fenwick",
        "Gwynn", "Huxley", "Io", "Jael", "Kestrel", "Liora",
    ],
    last_names=[
        "Vex", "Onyx", "Frost", "Tempest", "Shade", "Quill",
        "Rune", "Shard", "Thorn", "Wisp", "Ember", "Gale",
        "Hex", "Iris", "Jade", "Kai", "Lynx", "Mist",
    ],
    weapon_templates=[
        "cryogenic injector", "sonic disruptor", "nano-wire garrotte",
        "paralytic dart", "gravity hammer", "plasma cutter",
        "neural scrambler", "containment sphere", "biometric lock override",
    ],
    motive_templates=[
        "stolen research patent", "AI ethics disagreement",
        "sabotaged space mission", "cryptocurrency heist gone wrong",
        "time-loop paradox grudge", "cloning identity crisis",
        "simulation escape plan", "quantum entanglement betrayal",
    ],
    object_templates=[
        "cracked holographic display", "expired access badge",
        "encrypted data chip", "smashed communicator",
        "torn hazmat suit", "depleted energy cell",
        "suspicious biometric log", "modified atmospheric sensor",
        "disassembled drone parts", "contaminated sample vial",
    ],
)

# Novel dynamics: faster decay, more aggressive culprit
NOVEL_DYNAMICS_CONFIG = ComplexityConfig(
    num_locations=5, num_suspects=4, num_innocents=2, num_weapons=3,
    num_objects=8, num_red_herrings=3, num_time_steps=12,
    evidence_decay_rate=0.3,          # 3x standard
    witness_memory_half_life=2,       # 3x faster decay
    weather_change_prob=0.3,          # 2x standard
    npc_move_prob=0.5,                # higher movement
    culprit_tamper_prob=0.5,          # very aggressive culprit
    alibi_complexity=2, motive_layers=2,
    requires_deduction=True, requires_abduction=True,
    max_agent_actions=30,
)

# Novel structure: many small rooms
NOVEL_STRUCTURE_CONFIG = ComplexityConfig(
    num_locations=12, num_suspects=6, num_innocents=4, num_weapons=5,
    num_objects=15, num_red_herrings=4, num_time_steps=10,
    evidence_decay_rate=0.1, witness_memory_half_life=6,
    weather_change_prob=0.15, npc_move_prob=0.3, culprit_tamper_prob=0.2,
    alibi_complexity=3, motive_layers=2,
    requires_deduction=True, requires_abduction=True,
    max_agent_actions=35,
)


def run_generalisation_eval(
    n_instances: int = 10,
    output_dir: str = "results/generalisation",
) -> None:
    """Run the generalisation evaluation."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    conditions = {
        "standard": {
            "config": COMPLEXITY_PRESETS[ComplexityLevel.MEDIUM],
            "pool": None,  # default pool
        },
        "novel_entities": {
            "config": COMPLEXITY_PRESETS[ComplexityLevel.MEDIUM],
            "pool": NOVEL_ASSET_POOL,
        },
        "novel_dynamics": {
            "config": NOVEL_DYNAMICS_CONFIG,
            "pool": None,
        },
        "novel_structure": {
            "config": NOVEL_STRUCTURE_CONFIG,
            "pool": None,
        },
        "novel_all": {
            "config": NOVEL_DYNAMICS_CONFIG,
            "pool": NOVEL_ASSET_POOL,
        },
    }

    all_results: dict[str, list[dict]] = {}

    for condition_name, params in conditions.items():
        logger.info(f"\n--- Condition: {condition_name} ---")
        metrics_list: list[EpisodeMetrics] = []

        for i in range(n_instances):
            seed = 5000 + i
            ws = generate_mystery(
                params["config"], seed,
                asset_pool=params["pool"],
            )
            agent = HeuristicAgent()
            result = run_episode(agent, ws, complexity_level=3, verbose=False)
            if result.metrics:
                metrics_list.append(result.metrics)

        solved = sum(m.solved for m in metrics_list)
        avg_eff = sum(m.clue_efficiency for m in metrics_list) / max(len(metrics_list), 1)
        avg_belief = sum(m.final_belief_accuracy for m in metrics_list) / max(len(metrics_list), 1)

        logger.info(f"  Solved: {solved}/{len(metrics_list)} ({solved / max(len(metrics_list), 1):.1%})")
        logger.info(f"  Avg clue efficiency: {avg_eff:.3f}")
        logger.info(f"  Avg belief accuracy: {avg_belief:.3f}")

        all_results[condition_name] = [m.to_dict() for m in metrics_list]

    # Save results
    results_path = output_dir_path / "generalisation_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    logger.info(f"\nResults saved to {results_path}")

    # Summary table
    print("\n=== GENERALISATION RESULTS (RQ3) ===")
    print(f"{'Condition':<20} {'Solve Rate':>12} {'Clue Eff':>12} {'Belief Acc':>12}")
    print("-" * 60)
    for cond, metrics in all_results.items():
        sr = sum(m["solved"] for m in metrics) / max(len(metrics), 1)
        ce = sum(m["clue_efficiency"] for m in metrics) / max(len(metrics), 1)
        ba = sum(m["final_belief_accuracy"] for m in metrics) / max(len(metrics), 1)
        print(f"{cond:<20} {sr:>12.3f} {ce:>12.3f} {ba:>12.3f}")


if __name__ == "__main__":
    run_generalisation_eval(n_instances=10)
