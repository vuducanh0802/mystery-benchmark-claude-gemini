"""
Benchmark instance generator.

Generates a controlled set of mystery instances across complexity levels,
ensuring each is solvable and reproducible via random seeds.
"""

from __future__ import annotations

import json
import structlog
from pathlib import Path
from typing import Any

from mystery_world import COMPLEXITY_PRESETS, ComplexityConfig, ComplexityLevel
from mystery_world.generator import generate_mystery, verify_solvability

logger = structlog.get_logger()


def generate_benchmark_suite(
    n_per_level: int = 20,
    levels: list[int] | None = None,
    base_seed: int = 42,
    output_dir: str | Path = "data/benchmark_v1",
    custom_configs: dict[int, ComplexityConfig] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate a full benchmark suite.

    Parameters
    ----------
    n_per_level: int
        Number of instances per complexity level.
    levels: list[int], optional
        Which complexity levels to generate (default: all 5)
    base_seed: int
        Starting seed for reproducibility
    output_dir: Path
        Where to save generated instances.
    custom_configs: dict, optional
        Override preset configs for specific levels

    Returns
    ----------
    list[dict]
        Manifest of generated instances
    """
    if levels is None:
        levels = [1, 2, 3, 4, 5]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []

    for level in levels:
        level_dir = output_dir / f"level_{level}"
        level_dir.mkdir(exist_ok=True)

        config = (
            custom_configs.get(level)
            if custom_configs
            else COMPLEXITY_PRESETS.get(ComplexityLevel(level))
        )
        if config is None:
            config = COMPLEXITY_PRESETS[ComplexityLevel.MEDIUM]
        
        logger.info(f"Generating {n_per_level} instances for level {level}...")
        generated = 0
        seed = base_seed + level * 10000

        while generated < n_per_level:
            instance_seed = seed + generated
            try:
                world_state = generate_mystery(config, instance_seed)
                check = verify_solvability(world_state)

                if not check["solvable"]:
                    logger.warning(f"Seed {instance_seed} not solvable, skipping.")
                    seed += 1
                    continue

                # Save instance
                filename = f"instance_{instance_seed}.json"
                world_state.save(level_dir / filename)

                # Save solution separately (for human annotation reference)
                culprit = world_state.get_culprit()
                victim = world_state.get_victim()
                weapon = world_state.objects.get(world_state.murder_weapon_id)
                murder_loc = world_state.locations.get(world_state.murder_location_id)

                solution = {
                    "seed": instance_seed,
                    "level": level,
                    "culprit": culprit.full_name if culprit else "",
                    "victim": victim.full_name if victim else "",
                    "weapon": weapon.name if weapon else "",
                    "location": murder_loc.name if murder_loc else "",
                    "motive": world_state.motive,
                    "murder_step": world_state.murder_step,
                    "solvability_check": check,
                    "num_evidence": len(world_state.evidence),
                    "num_characters": len(world_state.characters),
                    "num_locations": len(world_state.locations),
                }
                (level_dir / f"solution_{instance_seed}.json").write_text(
                    json.dumps(solution, indent=2)
                )

                manifest.append({
                    "instance_file": str(level_dir / filename),
                    "solution_file": str(level_dir / f"solution_{instance_seed}.json"),
                    **solution
                })

                generated += 1
                logger.info(f"Generated instance {generated}/{n_per_level} (seed={instance_seed})")
            
            except Exception as e:
                logger.error(f"Error generating seed {instance_seed}: {e}")
                seed += 1

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Benchmark suite saved to {output_dir} ({len(manifest)} instances)")
    return manifest



def load_benchmark_suite(
    benchmark_dir: str | Path
) -> list[tuple["WorldState", int]]:
    """Load a benchmark suite from disk for evaluation."""
    from mystery_world.world import WorldState

    benchmark_dir = Path(benchmark_dir)
    manifest_path = benchmark_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {benchmark_dir}")
    
    manifest = json.loads(manifest_path.read_text())
    instances = []
    for entry in manifest:
        ws = WorldState.load(entry["instance_file"])
        level = entry.get("level", 1)
        instances.append((ws, level))
    
    return instances
    