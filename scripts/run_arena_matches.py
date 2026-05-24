"""Unified MysteryArena runner for heuristic baselines, model battles, and matrices.

Heuristic baseline:

    uv run python scripts/run_arena_matches.py \
        --matchup heuristic \
        --levels TRIVIAL EASY \
        --seeds 0-9 \
        --workers 4

Two-way model battle:

    uv run python scripts/run_arena_matches.py \
        --matchup two-way \
        --model-a kimi-k2.5 \
        --model-b glm-4.7 \
        --levels TRIVIAL \
        --seeds 0-2 \
        --workers 2

Custom detective x culprit matrix:

    uv run python scripts/run_arena_matches.py \
        --matchup matrix \
        --detectives heuristic,kimi-k2.5 \
        --culprits passive,glm-4.7 \
        --levels TRIVIAL EASY \
        --seeds 0-2 \
        --workers 4
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.base_agent import BaseAgent
from arena.aggregate import write_outputs
from arena.hf_publish import HFPublishError, publish_run_to_hf
from arena.roster import ModelSpec, get_model, parse_model_list
from scripts.arena_run import (
    _episode_actor_step_budget,
    _load_env_file,
    _parse_levels,
    _parse_seeds,
    _run_one,
)

ArenaJob = tuple[ModelSpec, ModelSpec, str, int]


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_") or "arena"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _default_run_id(matchup: str, detectives: list[ModelSpec], culprits: list[ModelSpec]) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    if matchup == "two-way" and len(detectives) == 2:
        return f"battle_{_slug(detectives[0].name)}_vs_{_slug(detectives[1].name)}_{stamp}"
    if matchup == "heuristic" and len(culprits) == 1:
        return f"heuristic_{_slug(detectives[0].name)}_vs_{_slug(culprits[0].name)}_{stamp}"
    return f"arena_{_slug(matchup)}_{len(detectives)}x{len(culprits)}_{stamp}"


def _unique_specs(specs: list[ModelSpec]) -> list[ModelSpec]:
    out: list[ModelSpec] = []
    seen: set[tuple[str, str | None, str | None, str]] = set()
    for spec in specs:
        key = (spec.name, spec.provider, spec.model, spec.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out


def _has_llm(specs: list[ModelSpec]) -> bool:
    return any(spec.kind == "llm" for spec in specs)


def _sanitize_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ")
    lowered = message.lower()
    if "model not allowed" in lowered:
        return "model not allowed by the configured gateway/account"
    if "unauthorized" in lowered or "forbidden" in lowered or "permission" in lowered:
        return "gateway rejected the request due to authorization or permission"
    return message[:300]


def _preflight_model(spec: ModelSpec, *, base_url: str, api_key_env: str | None) -> tuple[bool, str]:
    if spec.kind != "llm":
        return True, "local baseline"
    if not spec.model:
        return False, "model ref has no provider model id"
    api_key = os.environ.get(api_key_env or "")
    if not api_key:
        return False, f"missing local environment variable: {api_key_env}"
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)
        client.chat.completions.create(
            model=spec.model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=4,
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001 - preflight should surface provider failures cleanly.
        return False, _sanitize_error(exc)
    return True, "ok"


def _configure_llm(args: argparse.Namespace, specs: list[ModelSpec]) -> int:
    if not _has_llm(specs):
        return 0
    if not args.litellm_url:
        args.litellm_url = os.environ.get(args.gateway_url_env)
    if not args.litellm_key_env and os.environ.get(args.gateway_key_env):
        args.litellm_key_env = args.gateway_key_env
    if not args.litellm_url:
        print("error: LLM matchups require LLM_GATEWAY_URL or --litellm-url", file=sys.stderr)
        return 2

    BaseAgent.configure_litellm(
        args.litellm_url,
        api_key_env=args.litellm_key_env,
        model=args.litellm_model,
    )

    if not args.preflight:
        return 0
    failures = []
    for spec in _unique_specs([s for s in specs if s.kind == "llm"]):
        ok, message = _preflight_model(spec, base_url=args.litellm_url, api_key_env=args.litellm_key_env)
        print(f"Preflight {spec.name} ({spec.model}): {message}")
        if not ok:
            failures.append((spec.name, message))
    if failures:
        print("error: preflight failed; no matches were run", file=sys.stderr)
        return 1
    return 0


def _resolve_matchup(args: argparse.Namespace) -> tuple[list[ModelSpec], list[ModelSpec], list[ArenaJob]]:
    levels = _parse_levels(args.levels)
    seeds = _parse_seeds(args.seeds)

    if args.matchup == "two-way":
        model_a_detective = get_model(args.model_a, role="detective")
        model_a_culprit = get_model(args.model_a, role="culprit")
        model_b_detective = get_model(args.model_b, role="detective")
        model_b_culprit = get_model(args.model_b, role="culprit")
        detectives = [model_a_detective, model_b_detective]
        culprits = [model_a_culprit, model_b_culprit]
        jobs = []
        for level in levels:
            for seed in seeds:
                jobs.append((model_a_detective, model_b_culprit, level, seed))
                jobs.append((model_b_detective, model_a_culprit, level, seed))
        return detectives, culprits, jobs

    if args.matchup == "heuristic":
        detective = get_model(args.detective, role="detective")
        culprits = parse_model_list(args.culprits, role="culprit")
        jobs = [
            (detective, culprit, level, seed)
            for culprit in culprits
            for level in levels
            for seed in seeds
        ]
        return [detective], culprits, jobs

    detectives = parse_model_list(args.detectives, role="detective")
    culprits = parse_model_list(args.culprits, role="culprit")
    jobs = [
        (detective, culprit, level, seed)
        for detective in detectives
        for culprit in culprits
        for level in levels
        for seed in seeds
    ]
    return detectives, culprits, jobs


def _model_scores(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores: dict[str, list[float]] = defaultdict(list)
    role_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"detective": 0, "culprit": 0})
    for match in matches:
        detective = match.get("detective", {}).get("name", "unknown")
        culprit = match.get("culprit", {}).get("name", "unknown")
        scores[detective].append(float(match.get("detective_payoff", 0.0)))
        scores[culprit].append(float(match.get("culprit_payoff", 0.0)))
        role_counts[detective]["detective"] += 1
        role_counts[culprit]["culprit"] += 1

    rows = []
    for model, values in scores.items():
        rows.append(
            {
                "model": model,
                "score": round(mean(values), 6) if values else 0.0,
                "n": len(values),
                "detective_games": role_counts[model]["detective"],
                "culprit_games": role_counts[model]["culprit"],
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    for idx, row in enumerate(rows, 1):
        row["rank"] = idx
    return rows


def _write_report(path: Path, *, run_id: str, matchup: str, matches: list[dict[str, Any]], scores: list[dict[str, Any]]) -> None:
    lines = [
        "# MysteryArena Run Report",
        "",
        f"- run_id: `{run_id}`",
        f"- matchup: `{matchup}`",
        f"- matches: `{len(matches)}`",
        "",
        "## Overall Scores",
        "",
        "| Rank | Model | Score | N | Detective Games | Culprit Games |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in scores:
        lines.append(
            "| {rank} | {model} | {score:.3f} | {n} | {detective_games} | {culprit_games} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Matches",
            "",
            "| Detective | Culprit | Level | Seed | Detective Payoff | Culprit Payoff | Solved | Actions | Error |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for match in matches:
        error_lines = str(match.get("error") or "").splitlines()
        error = error_lines[0] if error_lines else ""
        lines.append(
            "| {detective} | {culprit} | {level} | {seed} | {d:.3f} | {c:.3f} | {solved} | {actions} | {error} |".format(
                detective=match.get("detective", {}).get("name", "unknown"),
                culprit=match.get("culprit", {}).get("name", "unknown"),
                level=match.get("level", ""),
                seed=int(match.get("seed", 0) or 0),
                d=float(match.get("detective_payoff", 0.0)),
                c=float(match.get("culprit_payoff", 0.0)),
                solved=bool(match.get("solved", False)),
                actions=int(match.get("actions_taken", 0) or 0)
                + int(match.get("culprit_actions_taken", 0) or 0),
                error=error.replace("|", "/"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _materialize_run(
    *,
    out_dir: Path,
    run_id: str,
    matchup: str,
    matches: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    matches.sort(
        key=lambda m: (
            str(m.get("level", "")),
            int(m.get("seed", 0) or 0),
            m.get("detective", {}).get("name", ""),
            m.get("culprit", {}).get("name", ""),
        )
    )
    with (out_dir / "matches.jsonl").open("w", encoding="utf-8") as fh:
        for match in matches:
            fh.write(json.dumps(match, ensure_ascii=False, default=str) + "\n")

    outputs = write_outputs(out_dir, bootstrap_samples=bootstrap_samples)
    scores = _model_scores(matches)
    summary = {
        "run_id": run_id,
        "matchup": matchup,
        "out_dir": str(out_dir),
        "matches": matches,
        "model_scores": scores,
        "outputs": outputs,
    }
    _write_json(out_dir / "arena_summary.json", summary)
    _write_report(out_dir / "report.md", run_id=run_id, matchup=matchup, matches=matches, scores=scores)
    return summary


def _publish_snapshot(args: argparse.Namespace, *, out_dir: Path, match_count: int) -> None:
    try:
        payload = publish_run_to_hf(
            out_dir,
            repo_id=args.repo_id,
            private=args.private,
            revision=args.revision,
            create_pr=args.create_pr,
            include_model_responses=args.include_model_responses,
            commit_message=f"Publish Arena run {out_dir.name} match {match_count}",
        )
    except HFPublishError as exc:
        if args.fail_on_publish_error:
            raise
        print(f"publish failed after match {match_count}: {exc}", file=sys.stderr)
        return
    print(f"Published {match_count} match(es) to {payload['repo_id']}: {payload['upload']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified MysteryArena runner with per-match HF publishing.")
    parser.add_argument(
        "--matchup",
        choices=["heuristic", "two-way", "matrix"],
        default="heuristic",
        help="Run preset. heuristic is a detective baseline, two-way swaps two models, matrix runs all pairs.",
    )
    parser.add_argument("--model-a", default="kimi-k2.5", help="First model for --matchup two-way.")
    parser.add_argument("--model-b", default="glm-4.7", help="Second model for --matchup two-way.")
    parser.add_argument("--detective", default="heuristic", help="Detective for --matchup heuristic.")
    parser.add_argument("--detectives", default="heuristic", help="Comma-separated detectives for --matchup matrix.")
    parser.add_argument("--culprits", default="passive", help="Comma-separated culprits for heuristic or matrix.")
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["TRIVIAL"],
        help="Difficulty levels, e.g. TRIVIAL EASY or TRIVIAL,EASY.",
    )
    parser.add_argument("--seeds", default="0", help="Seed spec such as 0, 0-2, or 0,3,5.")
    parser.add_argument("--out", default=None, help="Output directory. Defaults under arena/results/.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--gateway-url-env", default="LLM_GATEWAY_URL")
    parser.add_argument("--gateway-key-env", default="LLM_GATEWAY_API_KEY")
    parser.add_argument("--litellm-url", default=None)
    parser.add_argument("--litellm-key-env", default=None)
    parser.add_argument("--litellm-model", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1, help="Number of matches to run concurrently.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--publish-hf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish to Hugging Face after every completed match.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Target Hugging Face Dataset repo. Defaults to ARENA_HF_DATASET or Elfsong/Mystery_Arena_Results.",
    )
    parser.add_argument("--private", action="store_true", help="Create target dataset as private if needed.")
    parser.add_argument("--revision", default=None, help="Target Dataset branch or revision.")
    parser.add_argument("--create-pr", action="store_true", help="Publish through a Dataset pull request.")
    parser.add_argument(
        "--include-model-responses",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include raw model_response fields in published trajectories.",
    )
    parser.add_argument(
        "--fail-on-publish-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the run if per-match publishing fails.",
    )
    parser.add_argument(
        "--preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check LLM model access with a tiny chat completion before running.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_env_file(args.env_file)
    args.repo_id = args.repo_id or os.environ.get("ARENA_HF_DATASET", "Elfsong/Mystery_Arena_Results")

    levels = _parse_levels(args.levels)
    seeds = _parse_seeds(args.seeds)
    detectives, culprits, jobs = _resolve_matchup(args)
    all_specs = _unique_specs([*detectives, *culprits])
    llm_status = _configure_llm(args, all_specs)
    if llm_status:
        return llm_status

    run_id = args.run_id or _default_run_id(args.matchup, detectives, culprits)
    out_dir = Path(args.out) if args.out else Path("arena/results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    npc = {
        "provider": "fallback",
        "model": "gpt-4o-mini",
        "url": None,
        "seed": 42,
        "prompt_policy": "role_facts_only_no_strategy",
    }
    _write_json(
        out_dir / "config.json",
        {
            "run_id": run_id,
            "mode": args.matchup,
            "levels": levels,
            "seeds": seeds,
            "npc": npc,
            "detectives": [detective.to_dict() for detective in detectives],
            "culprits": [culprit.to_dict() for culprit in culprits],
            "rating": {"bootstrap_samples": args.bootstrap_samples},
            "llm_gateway": {
                "url": args.litellm_url,
                "key_env": args.litellm_key_env,
                "model_override": args.litellm_model,
            },
        },
    )
    _write_json(
        out_dir / "roster.json",
        {
            "detectives": [detective.to_dict() for detective in detectives],
            "culprits": [culprit.to_dict() for culprit in culprits],
        },
    )

    print(f"Arena run: {run_id}")
    print(f"Matchup: {args.matchup}")
    print(f"Detectives: {', '.join(d.name for d in detectives)}")
    print(f"Culprits: {', '.join(c.name for c in culprits)}")
    print(
        f"Levels: {','.join(levels)} | seeds={','.join(str(seed) for seed in seeds)} | "
        f"jobs={len(jobs)} | workers={args.workers}"
    )
    print(f"Out: {out_dir}")
    print(f"Publish HF: {args.publish_hf} | repo={args.repo_id}")

    matches: list[dict[str, Any]] = []
    publish_lock = threading.Lock()
    started = time.monotonic()

    def run_job(idx: int, job: ArenaJob) -> tuple[int, dict[str, Any]]:
        detective, culprit, level, seed = job
        print(
            f"[{idx}/{len(jobs)}] detective={detective.name} culprit={culprit.name} "
            f"level={level} seed={seed} budget={_episode_actor_step_budget(level, culprit)}",
            flush=True,
        )
        match = _run_one(
            run_id=run_id,
            out_dir=out_dir,
            detective=detective,
            culprit=culprit,
            npc=npc,
            level=level,
            seed=seed,
            skip_existing=args.resume,
            progress=None,
        )
        return idx, match

    def record_completed(idx: int, match: dict[str, Any]) -> dict[str, Any]:
        matches.append(match)
        with publish_lock:
            current_summary = _materialize_run(
                out_dir=out_dir,
                run_id=run_id,
                matchup=args.matchup,
                matches=matches,
                bootstrap_samples=args.bootstrap_samples,
            )
            if args.publish_hf:
                _publish_snapshot(args, out_dir=out_dir, match_count=len(matches))
        elapsed = time.monotonic() - started
        print(
            f"Recorded {len(matches)}/{len(jobs)} match(es) after {elapsed:.1f}s; "
            f"completed job #{idx}",
            flush=True,
        )
        return current_summary

    summary: dict[str, Any] = {"model_scores": []}
    workers = max(1, int(args.workers))
    if workers == 1:
        for idx, job in enumerate(jobs, 1):
            completed_idx, match = run_job(idx, job)
            summary = record_completed(completed_idx, match)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_job, idx, job): idx for idx, job in enumerate(jobs, 1)}
            for future in as_completed(futures):
                completed_idx, match = future.result()
                summary = record_completed(completed_idx, match)

    print("\nOverall scores:")
    for row in summary["model_scores"]:
        print(
            f"  #{row['rank']} {row['model']}: {row['score']:.3f} "
            f"(n={row['n']}, detective={row['detective_games']}, culprit={row['culprit_games']})"
        )
    print(f"\nReport: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
