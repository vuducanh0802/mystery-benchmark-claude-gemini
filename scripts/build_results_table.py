"""
Aggregate JSONL trajectories under {trajectory_dir}/{agent}/{level}/seed_*.jsonl
into a per-(agent,level) table.

Outputs:
    {output}/summary.csv
    {output}/summary.md

Columns: agent, level, n, solve_rate, composite, accusation, triangle,
         alibi, elimination, avg_actions, avg_tokens.

Usage:
    uv run scripts/build_results_table.py \
        --trajectory-dir results/trajectories \
        --output results/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


_LEVEL_ORDER = ["TRIVIAL", "EASY", "MEDIUM", "HARD", "EXPERT"]


def _load_footer(path: Path) -> dict | None:
    """Return the footer record of a JSONL trajectory, or None if missing."""
    last = None
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") == "footer":
                last = rec
    return last


def _safe(d: dict | None, *keys, default=0.0):
    cur = d or {}
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _collect(traj_dir: Path) -> dict[tuple[str, str], list[dict]]:
    """Return {(agent, level): [metrics_dict, ...]}."""
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for f in traj_dir.rglob("seed_*.jsonl"):
        try:
            footer = _load_footer(f)
        except Exception as e:
            print(f"  ! skipping unreadable {f}: {e}")
            continue
        if not footer:
            continue
        # agent / level live in the directory tree: .../{agent}/{level}/seed_*.jsonl
        try:
            level = f.parent.name
            agent = f.parent.parent.name
        except Exception:
            continue
        m = footer.get("metrics") or {}
        summary = footer.get("episode_summary") or {}
        # Budget-exhaustion penalty: agents that ran out of actions and
        # were force-accused get all reward metrics zeroed at report time.
        # This applies retroactively to existing trajectories.
        actions_taken = int(_safe(summary, "actions_taken", default=0)
                            or _safe(m, "actions_used", default=0))
        budget = int(_safe(summary, "budget", default=0)
                     or _safe(m, "action_budget", default=0))
        budget_exhausted = budget > 0 and actions_taken >= budget
        zero = budget_exhausted
        out[(agent, level)].append({
            "solved": False if zero else bool(m.get("solved")),
            "composite": 0.0 if zero else float(_safe(m, "composite_score")),
            "accusation": 0.0 if zero else float(_safe(m, "accusation_score")),
            "triangle": 0.0 if zero else float(_safe(m, "triangle_score")),
            "alibi": 0.0 if zero else float(_safe(m, "alibi_score")),
            "elimination": 0.0 if zero else float(
                _safe(summary, "scoring", "elimination_score")
                or _safe(summary, "elimination_score")
            ),
            "actions": actions_taken,
            "tokens": int(_safe(m, "total_tokens", default=0)),
            "error": bool(footer.get("error")),
            "budget_exhausted": budget_exhausted,
        })
    return out


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    f = lambda key: round(mean(r[key] for r in rows), 4)
    return {
        "n": n,
        "solve_rate": round(mean(1.0 if r["solved"] else 0.0 for r in rows), 4),
        "composite": f("composite"),
        "accusation": f("accusation"),
        "triangle": f("triangle"),
        "alibi": f("alibi"),
        "elimination": f("elimination"),
        "avg_actions": round(mean(r["actions"] for r in rows), 2),
        "avg_tokens": round(mean(r["tokens"] for r in rows), 1),
        "errors": sum(1 for r in rows if r["error"]),
        "budget_exhausted": sum(1 for r in rows if r.get("budget_exhausted")),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    traj_dir = Path(args.trajectory_dir)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    grouped = _collect(traj_dir)
    if not grouped:
        print(f"No trajectories found under {traj_dir}")
        return 1

    agents = sorted({a for (a, _) in grouped.keys()})
    levels_present = sorted({l for (_, l) in grouped.keys()},
                            key=lambda x: (_LEVEL_ORDER.index(x) if x in _LEVEL_ORDER else 99, x))

    cols = ["agent", "level", "n", "solve_rate", "composite", "accusation",
            "triangle", "alibi", "elimination", "avg_actions", "avg_tokens",
            "errors", "budget_exhausted"]

    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"

    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for a in agents:
            for l in levels_present:
                stats = _agg(grouped.get((a, l), []))
                if stats.get("n", 0) == 0:
                    continue
                w.writerow({"agent": a, "level": l, **stats})

    with md_path.open("w") as fh:
        fh.write("# MysteryArena results\n\n")
        for a in agents:
            fh.write(f"## {a}\n\n")
            fh.write("| level | n | solve | composite | accuse | triangle | alibi | elim | actions | tokens | err | budget_exh |\n")
            fh.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for l in levels_present:
                s = _agg(grouped.get((a, l), []))
                if s.get("n", 0) == 0:
                    continue
                fh.write(
                    f"| {l} | {s['n']} | {s['solve_rate']:.3f} | {s['composite']:.3f} | "
                    f"{s['accusation']:.3f} | {s['triangle']:.3f} | {s['alibi']:.3f} | "
                    f"{s['elimination']:.3f} | {s['avg_actions']:.1f} | "
                    f"{s['avg_tokens']:.0f} | {s['errors']} | {s['budget_exhausted']} |\n"
                )
            fh.write("\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
