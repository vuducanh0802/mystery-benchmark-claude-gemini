"""
Generate a benchmark suite of mystery instances.

Usage:
    python scripts/generate_benchmark.py --n-per-level 20 --levels 5 --seed 42 --output data/benchmark_v1/
"""

import argparse
import logging
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
from mystery_world.world import WorldState


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MysteryBench benchmark suite")
    parser.add_argument("--n-per-level", type=int, default=20, help="Instances per complexity level")
    parser.add_argument("--levels", type=int, default=5, help="Number of complexity levels (1-5)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--output", type=str, default="data/benchmark_v1", help="Output directory")
    parser.add_argument("--expert-annotations", action="store_true", help="Export human annotation sheets")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = structlog.get_logger()

    # Generate
    level_list = list(range(1, args.levels + 1))
    logger.info(f"Generating {args.n_per_level} instances x {len(level_list)} levels {args.n_per_level * len(level_list)} total")
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
