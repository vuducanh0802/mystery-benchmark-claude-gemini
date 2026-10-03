#!/usr/bin/env python3
"""Run three guarded GPT-OSS ablations against an existing compatible endpoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tarfile
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.guard_ablations import ABLATION_VERSION, VARIANTS, GuardAblationDetectiveAgent
from agents.llm_agent import BIAS_GUARDED_SYSTEM_PROMPT
from scripts import run_claude_gemini_baselines as core
from scripts import run_gpt_oss_kimi_baselines as paired


class Ablation(paired.SanitizedTransport, GuardAblationDetectiveAgent):
    pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    p.add_argument("--benchmark-dir", type=Path, default=ROOT / "data/benchmark_v1")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/gpt_oss_guarded_ablations")
    p.add_argument("--experiment-id", default="gpt_oss_guarded_ablations_v1")
    p.add_argument("--levels", nargs="+", default=list(core.LEVEL_NAMES.values()))
    p.add_argument("--per-level", type=int, default=200)
    p.add_argument("--gpt-oss-model", default=os.getenv("GPT_OSS_MODEL", "gpt-oss-120b"))
    p.add_argument("--gpt-oss-base-url", default=os.getenv("GPT_OSS_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--gpt-oss-workers", type=int, default=1)
    p.add_argument("--temperature", type=float, help="Override GPT_OSS_TEMPERATURE; default 0")
    p.add_argument("--reasoning-effort", choices=("low", "medium", "high"),
                   help="Override GPT_OSS_EXTRA_BODY reasoning_effort; default low")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--history-window", type=int, default=10)
    p.add_argument("--max-accuse-blocks", type=int, default=2)
    p.add_argument("--timeout-seconds", type=float, default=180)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--retry-backoff-seconds", type=float, default=1)
    p.add_argument("--retry-rounds", type=int, default=2)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="Offline checks without model calls")
    mode.add_argument("--preflight-only", action="store_true", help="One endpoint completion without episodes")
    mode.add_argument("--report-only", action="store_true", help="Rebuild reports without model calls")
    args = p.parse_args(argv)
    if len(set(args.variants)) != len(args.variants):
        p.error("variants must not contain duplicates")
    if min(args.per_level, args.gpt_oss_workers, args.max_tokens) <= 0:
        p.error("per-level, workers, and max-tokens must be positive")
    if min(args.history_window, args.max_accuse_blocks, args.max_retries, args.retry_rounds) < 0:
        p.error("history, accusation blocks, and retries must be nonnegative")
    if (not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0
            or not math.isfinite(args.retry_backoff_seconds) or args.retry_backoff_seconds < 0):
        p.error("invalid timeout or retry backoff")
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        p.error("temperature must be between 0 and 2")
    args.models = ["gpt-oss"]
    args.policies = args.variants
    args.kimi_workers = 1  # Shared executor; no Kimi endpoint is selected.
    return args


def endpoint_from_args(args):
    endpoint = paired.endpoints_from_args(args)["gpt-oss"]
    extra = endpoint.extra_body.copy()
    effort = args.reasoning_effort or extra.get("reasoning_effort", "low")
    if effort not in {"low", "medium", "high"}:
        raise ValueError("GPT-OSS reasoning_effort must be low, medium, or high")
    template_effort = extra.get("chat_template_kwargs", {}).get("reasoning_effort")
    if template_effort is not None and template_effort != effort:
        raise ValueError("Conflicting reasoning_effort and chat_template_kwargs")
    extra.update(reasoning_effort=effort, response_format={"type": "json_object"})
    temperature = args.temperature if args.temperature is not None else endpoint.temperature
    return replace(endpoint, temperature=0.0 if temperature is None else temperature, extra_body=extra)


def make_agent(job, args):
    endpoint = args.endpoints[job.model.name]
    agent = Ablation(
        variant=job.policy, max_accuse_blocks=args.max_accuse_blocks,
        agent_id=job.cell_id, provider="openai", model=job.model.model,
        base_url=endpoint.base_url,
        api_key=os.getenv(endpoint.api_key_env) or "EMPTY",
        max_tokens=args.max_tokens, max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    agent.config.temperature = endpoint.temperature
    agent.config.extra_body = endpoint.extra_body.copy()
    return agent


def reference_case_hashes():
    """Canonical instance identities, independent of manifest path formatting."""
    bundle = ROOT / "benchmark_suites/claude_gemini_cases_v1.tar.gz"
    metadata = json.loads(bundle.with_suffix("").with_suffix(".json").read_text())
    if core._sha256(bundle) != metadata["archive_sha256"]:
        raise ValueError("Frozen benchmark bundle checksum mismatch")
    with tarfile.open(bundle, "r:gz") as archive:
        raw = archive.extractfile("manifest.json").read()
    if hashlib.sha256(raw).hexdigest() != metadata["files"]["manifest.json"]:
        raise ValueError("Frozen manifest checksum mismatch")
    ordinals = Counter()
    expected = {}
    for entry in json.loads(raw):
        level = int(entry["level"])
        ordinal = ordinals[level]
        ordinals[level] += 1
        identity = f"level_{level}_case_{ordinal:04d}_seed_{int(entry['seed'])}"
        relative = f"level_{level}/{Path(entry['instance_file']).name}"
        expected[identity] = metadata["files"][relative]
    return expected


def build_config(args, cases):
    expected = reference_case_hashes()
    for case in cases:
        if expected.get(case.instance_id) != case.sha256:
            raise ValueError(f"Case differs from the fixed GPT-OSS baseline suite: {case.instance_id}")
    config = paired.build_config(args, cases)
    config.pop("config_fingerprint")
    created = config.pop("created_at")
    config.update(
        ablation_version=ABLATION_VERSION,
        variants={name: VARIANTS[name].metadata(args.max_accuse_blocks) for name in args.variants},
        guarded_system_prompt_sha256=hashlib.sha256(BIAS_GUARDED_SYSTEM_PROMPT.encode()).hexdigest(),
        byte_distinct_worlds=len({case.sha256 for case in cases}),
        environment={
            "patch": "inventory-relocation-missing-room-v1",
            "events_sha256": core._sha256(ROOT / "mystery_world/events.py"),
        },
        reference_release={
            "dataset": "Elfsong/Mystery-Benchmark-Full",
            "revision": "a4e88a7ca2fd7530e703f76a73dde87488c9a680",
            "model_revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
            "case_identity_and_bytes_verified": True,
        },
    )
    config["config_fingerprint"] = core._stable_hash(config)
    config["created_at"] = created
    return config


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(output_dir, jobs):
    valid_rows = []
    validation = core.write_reports(output_dir, jobs, episode_rows=valid_rows)
    valid_by_id = {(r["model_identity"], r["policy"], r["instance_id"]): r for r in valid_rows}
    invalid = {r["job_id"]: r["reason"] for r in validation["invalid"]}
    rows = []
    groups = defaultdict(list)
    for job in jobs:
        valid = valid_by_id.get((job.model.identity, job.policy, job.case.instance_id))
        reason = invalid.get(job.job_id, "")
        row = {
            "model": job.model.model, "variant": job.policy,
            "level": core.LEVEL_NAMES[job.case.level], "instance_id": job.case.instance_id,
            "source_sha256": job.case.sha256, "valid": bool(valid), "error": reason,
            "solved": int(valid["solved"]) if valid else 0,
            "composite": valid["composite"] if valid else 0,
            "actions": valid["actions"] if valid else 0,
            "input_tokens": valid["input_tokens"] if valid else 0,
            "output_tokens": valid["output_tokens"] if valid else 0,
            "guard_interventions": valid["guard_interventions"] if valid else 0,
            "guard_reasons": json.dumps(valid["guard_reasons"] if valid else []),
            "budget_exhausted": bool(valid and valid["budget_exhausted"]),
        }
        rows.append(row)
        groups[(job.policy, row["level"])].append(row)
        groups[(job.policy, "OVERALL")].append(row)
    summaries = []
    for (variant, level), items in groups.items():
        valid = [r for r in items if r["valid"]]
        solved = sum(r["solved"] for r in items)
        summaries.append({
            "variant": variant, "level": level, "expected_n": len(items),
            "valid_n": len(valid), "invalid_n": len(items) - len(valid),
            "solved_n": solved, "solve_rate": solved / len(items),
            "solve_rate_valid_only": solved / len(valid) if valid else None,
            "composite": sum(r["composite"] for r in items) / len(items),
            "avg_actions_valid": mean(r["actions"] for r in valid) if valid else None,
            "avg_input_tokens_valid": mean(r["input_tokens"] for r in valid) if valid else None,
            "avg_output_tokens_valid": mean(r["output_tokens"] for r in valid) if valid else None,
            "avg_guard_interventions_valid": mean(r["guard_interventions"] for r in valid) if valid else None,
            "budget_exhausted_n": sum(r["budget_exhausted"] for r in valid),
        })
    write_csv(output_dir / "episode_metrics.csv", rows)
    write_csv(output_dir / "ablation_summary.csv", summaries)
    return validation


def main(argv=None):
    args = parse_args(argv)
    args.endpoints = {"gpt-oss": endpoint_from_args(args)}
    return paired.execute(
        args, agent_factory=make_agent, config_factory=build_config,
        report_writer=write_reports,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2)
