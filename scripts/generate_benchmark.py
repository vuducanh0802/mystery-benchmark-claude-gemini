"""
Generate a benchmark suite of mystery instances.

Usage:
    uv run scripts/generate_benchmark.py --levels TRIVIAL EASY MEDIUM --instances-per-level 5 --seed 42 --output-dir data/benchmark_v1
"""

import argparse
import sys
from pathlib import Path
import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.generate import generate_benchmark_suite
from benchmark.verify import (
    check_solvability,
    check_structural_consistency,
    compute_diversity_metrics,
    export_annotation_sheet
)
from mystery_world import ComplexityLevel
from mystery_world.world import WorldState

LEVEL_NAMES = {lvl.name: lvl.value for lvl in ComplexityLevel}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MysteryArena benchmark suite")
    parser.add_argument(
        "--levels", nargs="+", default=["TRIVIAL", "EASY", "MEDIUM", "HARD", "EXPERT"],
        metavar="LEVEL",
        help=f"Complexity levels to generate. Choices: {list(LEVEL_NAMES)}",
    )
    parser.add_argument("--instances-per-level", type=int, default=20, dest="n_per_level",
                        help="Instances per complexity level")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--output-dir", type=str, default="data/benchmark_v1", dest="output",
                        help="Output directory")
    parser.add_argument("--expert-annotations", action="store_true", help="Export human annotation sheets")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = structlog.get_logger()

    # Resolve level names → int values
    level_list = []
    for name in args.levels:
        name_upper = name.upper()
        if name_upper not in LEVEL_NAMES:
            parser.error(f"Unknown level '{name}'. Choices: {list(LEVEL_NAMES)}")
        level_list.append(LEVEL_NAMES[name_upper])

    logger.info(f"Generating {args.n_per_level} instances x {len(level_list)} levels = {args.n_per_level * len(level_list)} total")
    manifest = generate_benchmark_suite(
        n_per_level=args.n_per_level,
        levels=level_list,
        base_seed=args.seed,
        output_dir=args.output,
    )

    # Verification pass
    logger.info("Running verification pass...")
    n_consistent = 0
    n_solvable = 0
    for entry in manifest:
        ws = WorldState.load(entry["instance_file"])
        consistency = check_structural_consistency(ws)
        solvability = check_solvability(ws)
        if consistency["consistent"]:
            n_consistent += 1
        else:
            logger.warning(f"Inconsistent: seed={entry['seed']}: {consistency['issues']}")
        if solvability["solvable"]:
            n_solvable += 1
        else:
            logger.warning(f"Not solvable: seed={entry['seed']}: {solvability}")
    
    logger.info(f"Consistency: {n_consistent}/{len(manifest)} passed")
    logger.info(f"Solvability: {n_solvable}/{len(manifest)} passed")

    # Diversity
    diversity = compute_diversity_metrics(args.output)
    logger.info(f"Diversity metrics: {diversity}")

    # Annotation sheets
    if args.expert_annotations:
        ann_dir = Path(args.output) / "annotations"
        ann_dir.mkdir(exist_ok=True)
        for entry in manifest:
            ws = WorldState.load(entry["instance_file"])
            export_annotation_sheet(ws, ann_dir / f"annotation_{entry['seed']}.txt")
        logger.info(f"Annotation sheets exported to {ann_dir}")
    
    logger.info("Done.")



if __name__ == "__main__":
    main()
