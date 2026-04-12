#!/usr/bin/env python3
"""
Analyse benchmark results and generate paper figures / tables.

Usage:
    python scripts/analyse_results.py \\
        --results-dir results/ \\
        --output figures/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.analysis import (
    compute_aggregate_table,
    generate_all_figures,
    load_results,
    significance_test,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse MysteryBench results")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing agent result subdirs")
    parser.add_argument("--output", type=str, default="figures/", help="Output directory for figures and tables")
    parser.add_argument("--agents", type=str, default=None, help="Comma-separated agent subdirectory names")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)

    # Auto-discover agent result directories
    if args.agents:
        agent_dirs = {name: results_dir / name for name in args.agents.split(",")}
    else:
        agent_dirs = {
            d.name: d for d in results_dir.iterdir()
            if d.is_dir() and (d / "all_metrics.json").exists()
        }

    if not agent_dirs:
        logger.error(f"No agent result directories found in {results_dir}")
        logger.info("Expected structure: results/<agent_name>/all_metrics.json")
        sys.exit(1)

    logger.info(f"Found agents: {list(agent_dirs.keys())}")

    # Load all results
    dfs = {}
    for name, path in agent_dirs.items():
        df = load_results(path)
        logger.info(f"  {name}: {len(df)} episodes loaded")
        dfs[name] = df

    # Generate figures
    logger.info("Generating figures...")
    generate_all_figures(agent_dirs, output_dir)

    # Generate aggregate table
    table = compute_aggregate_table(dfs)
    table_path = output_dir / "table2_results.csv"
    table.to_csv(table_path, index=False)
    logger.info(f"Aggregate table saved to {table_path}")
    print("\n=== AGGREGATE RESULTS (Table 2) ===")
    print(table.to_string(index=False))

    # Statistical significance tests
    agent_names = list(dfs.keys())
    if len(agent_names) >= 2:
        print(f"\n=== SIGNIFICANCE TESTS ===")
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                a, b = agent_names[i], agent_names[j]
                test = significance_test(dfs[a], dfs[b], "solved")
                print(f"  {a} vs {b} (solve rate): p={test['p_value']:.4f}, effect={test['effect_size']:.3f}")

                test_belief = significance_test(dfs[a], dfs[b], "final_belief_accuracy")
                print(f"  {a} vs {b} (belief acc): p={test_belief['p_value']:.4f}, d={test_belief.get('cohens_d', 0):.3f}")

    logger.info(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
