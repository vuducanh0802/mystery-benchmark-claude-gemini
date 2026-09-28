#!/usr/bin/env python3
"""Show API-backed GPT-4o completion counts for a targeted paired run."""

import argparse
import json
from collections import Counter
from pathlib import Path

from run_claude_gemini_baselines import (
    LEVEL_NAMES,
    ModelSpec,
    build_jobs,
    load_cases,
    select_target_jobs,
    trajectory_path,
    validate_trajectory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-per-level", type=int, default=20)
    args = parser.parse_args()
    if args.target_per_level <= 0:
        parser.error("--target-per-level must be positive")

    config = json.loads((args.output_dir / "run_config.json").read_text())
    cases = load_cases(Path(config["benchmark_dir"]), set(LEVEL_NAMES), None)
    models = [ModelSpec(**model) for model in config["models"]]
    jobs = select_target_jobs(
        build_jobs(cases, models, config["policies"], config["config_fingerprint"]),
        args.target_per_level,
    )
    valid = Counter()
    invalid = Counter()
    for job in jobs:
        ok, _, _ = validate_trajectory(trajectory_path(args.output_dir, job), job)
        key = (job.model.name, job.policy, job.case.level)
        if ok:
            valid[key] += 1
        else:
            invalid[key] += 1

    print("model   policy   level     valid  target  missing")
    for model in models:
        for policy in config["policies"]:
            for level, label in LEVEL_NAMES.items():
                key = (model.name, policy, level)
                total = valid[key] + invalid[key]
                if total:
                    print(f"{model.name:<7} {policy:<8} {label:<9} {valid[key]:>5} {total:>7} {invalid[key]:>8}")
    print(f"TOTAL: {sum(valid.values())}/{len(jobs)} valid")


if __name__ == "__main__":
    main()
