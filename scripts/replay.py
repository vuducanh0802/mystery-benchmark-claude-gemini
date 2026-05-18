"""
Replay a logged trajectory and verify state-hash equality at every step.

Usage:
    uv run scripts/replay.py path/to/trajectory.jsonl
    uv run scripts/replay.py path/to/trajectory.jsonl --inspect
    uv run scripts/replay.py path/to/trajectory.jsonl --no-verify   # just walk through

Exit code 0 on success, 1 on first hash mismatch (or any error).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.trajectory import read_trajectory, world_state_hash
from mystery_world import COMPLEXITY_PRESETS, ComplexityLevel
from mystery_world.generator import generate_mystery
from mystery_world.world import AgentAction, MysteryEnvironment


def _config_from_header(header: dict):
    """Reconstruct a ComplexityConfig matching the logged config."""
    from mystery_world import ComplexityConfig
    return ComplexityConfig.from_dict(header["config"])


def main() -> int:
    p = argparse.ArgumentParser(description="Replay a JSONL trajectory.")
    p.add_argument("trajectory")
    p.add_argument("--inspect", action="store_true", help="Print each step.")
    p.add_argument("--no-verify", action="store_true", help="Skip world-state hash assertions.")
    args = p.parse_args()

    recs = read_trajectory(args.trajectory)
    if not recs or recs[0].get("kind") != "header":
        print("ERROR: missing header line", file=sys.stderr)
        return 1

    header = recs[0]
    steps = [r for r in recs if r.get("kind") == "step"]

    config = _config_from_header(header)
    state = generate_mystery(seed=header["seed"], config=config)
    env = MysteryEnvironment(state)
    if header.get("config", {}).get("free_culprit_actions") or any(
        r.get("actor_id", "detective") != "detective" for r in steps
    ):
        env.enable_free_culprit()

    detective_name = header.get("detective_agent", header.get("agent"))
    detective_model = header.get("detective_model", header.get("model"))
    print(f"Replaying seed={header['seed']} level={header['level']} "
          f"detective={detective_name} model={detective_model} ({len(steps)} steps)")

    mismatches = 0
    for rec in steps:
        try:
            action = AgentAction[rec["action"]]
        except KeyError:
            print(f"  ! step {rec['step']}: unknown action {rec['action']}", file=sys.stderr)
            return 1
        actor_id = rec.get("actor_id", "detective")
        if actor_id == "detective":
            env.step(action, **(rec.get("action_kwargs") or {}))
        else:
            env.step_for_actor(actor_id, action, **(rec.get("action_kwargs") or {}))
        post_hash = world_state_hash(state)
        ok = (post_hash == rec["world_state_hash"])
        if args.inspect:
            mark = "OK" if ok else "MISMATCH"
            role = rec.get("role", actor_id)
            print(f"  [step {rec['step']:>3}] {role:<10} {rec['action']:<22} {mark}")
        if not ok:
            mismatches += 1
            if not args.no_verify:
                print(f"  ! hash mismatch at step {rec['step']}: "
                      f"expected {rec['world_state_hash'][:12]}, got {post_hash[:12]}",
                      file=sys.stderr)
                return 1

    if args.no_verify:
        print(f"Walked {len(steps)} steps (verify disabled, {mismatches} hash diffs).")
    else:
        print(f"OK: {len(steps)} steps replayed, all world-state hashes match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
