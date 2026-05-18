"""
Run a small set of (level, seed) pairs twice with the heuristic agent and
assert that the action+state-hash signatures of the two runs are identical.

Exit 0 on success, 1 on first divergence.

Usage:
    uv run scripts/verify_reproducibility.py
    uv run scripts/verify_reproducibility.py --levels TRIVIAL EASY --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.heuristic_agent import HeuristicAgent
from evaluation.runner import run_episode
from evaluation.trajectory import TrajectoryWriter, trajectory_hash
from mystery_world import COMPLEXITY_PRESETS, ComplexityLevel
from mystery_world.generator import generate_mystery


def _run_once(level_name: str, seed: int, out: Path) -> str:
    config = COMPLEXITY_PRESETS[ComplexityLevel[level_name]]
    state = generate_mystery(seed=seed, config=config)
    detective_agent = HeuristicAgent(agent_id="heuristic")
    with TrajectoryWriter(out) as w:
        w.write_header(
            state=state, level=level_name,
            agent="heuristic", model=None, provider=None,
            instance_id=f"seed_{seed}",
        )
        result = run_episode(
            detective_agent,
            state,
            complexity_level=ComplexityLevel[level_name].value,
            trajectory_writer=w,
        )
        w.write_footer(
            episode_summary=result.episode_summary,
            metrics=result.metrics.to_dict() if result.metrics else None,
            elapsed_seconds=result.elapsed_seconds,
            error=result.error,
        )
    return trajectory_hash(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--levels", nargs="+", default=["TRIVIAL", "EASY"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="mb_repro_"))
    try:
        ok = True
        for lvl in args.levels:
            for seed in args.seeds:
                a = _run_once(lvl, seed, tmp / f"{lvl}_{seed}_a.jsonl")
                b = _run_once(lvl, seed, tmp / f"{lvl}_{seed}_b.jsonl")
                tag = f"{lvl} seed={seed}"
                if a == b:
                    print(f"  OK   {tag}  hash={a[:12]}")
                else:
                    print(f"  FAIL {tag}  a={a[:12]}  b={b[:12]}")
                    ok = False
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
