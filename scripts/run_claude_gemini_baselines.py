#!/usr/bin/env python3
"""Run paired API-model Vanilla and Guarded detective baselines.

The runner consumes one benchmark manifest, gives every model/policy cell the
same serialized worlds, keeps identities collision-free, resumes only complete
API-backed trajectories, and never converts provider failures into game actions.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.base_agent import LLMConfig  # noqa: E402
from agents.llm_agent import (  # noqa: E402
    BiasGuardedLLMDetectiveAgent,
    LLMDetectiveAgent,
)
from evaluation.runner import run_episode  # noqa: E402
from evaluation.trajectory import TrajectoryWriter  # noqa: E402
from mystery_world.world import WorldState  # noqa: E402


LEVEL_NAMES = {1: "TRIVIAL", 2: "EASY", 3: "MEDIUM", 4: "HARD", 5: "EXPERT"}
LEVEL_VALUES = {name: value for value, name in LEVEL_NAMES.items()}
POLICIES = ("vanilla", "guarded")
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_GPT4O_MODEL = "gpt-4o"
GUARD_VERSION = "exposure-bias-guard-v1"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    model: str

    @property
    def identity(self) -> str:
        digest = hashlib.sha256(self.model.encode()).hexdigest()[:8]
        slug = re.sub(r"[^a-z0-9]+", "-", self.model.lower()).strip("-")
        return f"{self.name}__{slug}-{digest}"


@dataclass(frozen=True)
class BenchmarkCase:
    level: int
    ordinal: int
    benchmark_seed: int
    path: Path
    source_label: str
    sha256: str

    @property
    def instance_id(self) -> str:
        return (
            f"level_{self.level}_case_{self.ordinal:04d}_"
            f"seed_{self.benchmark_seed}"
        )


@dataclass(frozen=True)
class Job:
    model: ModelSpec
    policy: str
    case: BenchmarkCase
    run_fingerprint: str = ""

    @property
    def cell_id(self) -> str:
        return f"{self.model.identity}__{self.policy}"

    @property
    def job_id(self) -> str:
        return f"{self.cell_id}__{self.case.instance_id}"


@dataclass(frozen=True)
class JobResult:
    job: Job
    status: str
    error: str = ""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _parse_levels(raw_levels: list[str]) -> set[int]:
    levels: set[int] = set()
    for raw in raw_levels:
        value = raw.strip().upper()
        if value.isdigit() and int(value) in LEVEL_NAMES:
            levels.add(int(value))
        elif value in LEVEL_VALUES:
            levels.add(LEVEL_VALUES[value])
        else:
            raise ValueError(f"unknown level {raw!r}")
    return levels


def _resolve_instance_path(
    benchmark_dir: Path,
    raw_path: str,
    level: int,
) -> Path:
    raw = Path(raw_path)
    candidates = [raw] if raw.is_absolute() else [
        ROOT / raw,
        benchmark_dir / raw,
        benchmark_dir / f"level_{level}" / raw.name,
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"cannot resolve benchmark instance; tried: {rendered}")


def load_cases(
    benchmark_dir: Path,
    levels: set[int],
    per_level: int | None,
) -> list[BenchmarkCase]:
    manifest_path = benchmark_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing benchmark manifest: {manifest_path}")
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("benchmark manifest must be a JSON list")

    ordinals: Counter[int] = Counter()
    selected: Counter[int] = Counter()
    cases: list[BenchmarkCase] = []
    for entry in entries:
        level = int(entry.get("level", 0))
        if level not in LEVEL_NAMES:
            raise ValueError(f"invalid manifest level: {level}")
        ordinal = ordinals[level]
        ordinals[level] += 1
        if level not in levels:
            continue
        if per_level is not None and selected[level] >= per_level:
            continue
        source_label = str(entry["instance_file"])
        path = _resolve_instance_path(benchmark_dir, source_label, level)
        benchmark_seed = int(entry.get("seed", ordinal))
        cases.append(BenchmarkCase(
            level=level,
            ordinal=ordinal,
            benchmark_seed=benchmark_seed,
            path=path,
            source_label=source_label,
            sha256=_sha256(path),
        ))
        selected[level] += 1

    missing = sorted(levels - set(selected))
    if missing:
        raise ValueError(f"manifest has no cases for levels: {missing}")
    if per_level is not None:
        short = {level: selected[level] for level in levels if selected[level] < per_level}
        if short:
            raise ValueError(f"manifest has fewer than --per-level cases: {short}")
    return cases


def build_jobs(
    cases: list[BenchmarkCase],
    models: list[ModelSpec],
    policies: list[str],
    run_fingerprint: str = "",
) -> list[Job]:
    jobs = [
        Job(
            model=model,
            policy=policy,
            case=case,
            run_fingerprint=run_fingerprint,
        )
        for case in cases
        for model in models
        for policy in policies
    ]
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("job identities collide")
    return jobs


def trajectory_path(output_dir: Path, job: Job) -> Path:
    return (
        output_dir / "trajectories" / job.model.identity / job.policy
        / LEVEL_NAMES[job.case.level] / f"{job.case.instance_id}.jsonl"
    )


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("missing or empty trajectory")
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON at line {line_number}: {exc}") from exc
    return records


def validate_trajectory(path: Path, job: Job) -> tuple[bool, str, list[dict[str, Any]]]:
    try:
        records = _read_records(path)
        if not records or records[0].get("kind") != "header":
            return False, "missing header", records
        if records[-1].get("kind") != "footer":
            return False, "missing terminal footer", records
        if sum(record.get("kind") == "header" for record in records) != 1:
            return False, "header count is not one", records
        if sum(record.get("kind") == "footer" for record in records) != 1:
            return False, "footer count is not one", records

        header, footer = records[0], records[-1]
        expected = {
            "detective_agent": job.cell_id,
            "detective_model": job.model.model,
            "detective_provider": job.model.provider,
            "detective_policy": job.policy,
            "benchmark_seed": job.case.benchmark_seed,
            "instance_id": job.case.instance_id,
            "source_instance_sha256": job.case.sha256,
        }
        if job.run_fingerprint:
            expected["experiment_config_hash"] = job.run_fingerprint
        for key, value in expected.items():
            if header.get(key) != value:
                return False, f"header {key} mismatch", records
        if int(header.get("schema_version", 0)) < 2:
            return False, "trajectory schema is older than 2", records
        if str(footer.get("error") or "").strip():
            return False, f"terminal error: {footer['error']}", records
        metrics = footer.get("metrics")
        if not isinstance(metrics, dict):
            return False, "missing metrics", records

        steps = [record for record in records if record.get("kind") == "step"]
        called_steps = [record for record in steps if record.get("model_called")]
        if not called_steps:
            return False, "no API-backed detective step", records
        for record in called_steps:
            if not str(record.get("model_response") or "").strip():
                return False, "API-backed step has no model response", records
            input_tokens = int(record.get("input_tokens") or 0)
            output_tokens = int(record.get("output_tokens") or 0)
            if input_tokens + output_tokens <= 0:
                return False, "API-backed step has zero token usage", records
        if int(metrics.get("total_tokens") or 0) <= 0:
            return False, "episode has zero total tokens", records
        return True, "", records
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}", []


def _make_agent(job: Job, args: argparse.Namespace) -> LLMDetectiveAgent:
    agent_class = (
        BiasGuardedLLMDetectiveAgent if job.policy == "guarded"
        else LLMDetectiveAgent
    )
    return agent_class(
        agent_id=job.cell_id,
        provider=job.model.provider,
        model=job.model.model,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        timeout_seconds=args.timeout_seconds,
    )


def _run_job(output_dir: Path, experiment_id: str, job: Job, args: argparse.Namespace) -> JobResult:
    final_path = trajectory_path(output_dir, job)
    valid, _, _ = validate_trajectory(final_path, job)
    if valid:
        return JobResult(job, "skipped")

    state = WorldState.load(job.case.path)
    agent = _make_agent(job, args)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_suffix(
        final_path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp",
    )
    try:
        with TrajectoryWriter(temp_path) as writer:
            writer.write_header(
                state=state,
                level=LEVEL_NAMES[job.case.level],
                agent=job.cell_id,
                model=job.model.model,
                provider=job.model.provider,
                npc_provider="fallback",
                npc_model=None,
                npc_seed=42,
                instance_id=job.case.instance_id,
                benchmark_seed=job.case.benchmark_seed,
                source_instance=job.case.source_label,
                source_instance_sha256=job.case.sha256,
                detective_policy=job.policy,
                experiment_id=experiment_id,
                experiment_config_hash=job.run_fingerprint,
            )
            result = run_episode(
                detective_agent=agent,
                world_state=state,
                complexity_level=job.case.level,
                verbose=False,
                npc_responder=None,
                culprit_agent=None,
                trajectory_writer=writer,
            )
            result.instance_id = job.case.instance_id
            if result.metrics is not None:
                result.metrics.instance_id = job.case.instance_id
            result.episode_summary["benchmark_seed"] = job.case.benchmark_seed
            result.episode_summary["embedded_rng_seed"] = state.seed
            writer.write_footer(
                episode_summary=result.episode_summary,
                metrics=result.metrics.to_dict() if result.metrics else None,
                elapsed_seconds=result.elapsed_seconds,
                error=result.error,
            )
        os.replace(temp_path, final_path)
        if result.error:
            return JobResult(job, "error", result.error)
        valid, reason, _ = validate_trajectory(final_path, job)
        if not valid:
            return JobResult(job, "error", reason)
        return JobResult(job, "ok")
    except Exception as exc:  # noqa: BLE001
        temp_path.unlink(missing_ok=True)
        return JobResult(job, "crash", f"{type(exc).__name__}: {exc}")


def _is_fatal_provider_error(error: str) -> bool:
    lowered = error.lower()
    return any(marker in lowered for marker in (
        "authenticationerror", "permissiondenied", "invalid_api_key",
        "incorrect api key", "status code: 401", "error code: 401",
        "status code: 403", "error code: 403", "status code: 404",
        "error code: 404", "model not found", "unknown model",
    ))


def _safe_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    return mean(float(row[key]) for row in rows) if rows else None


def write_reports(output_dir: Path, jobs: list[Job]) -> dict[str, Any]:
    expanded: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    invalid_reasons: dict[str, str] = {}
    valid_job_ids: set[str] = set()
    for job in jobs:
        path = trajectory_path(output_dir, job)
        valid, reason, records = validate_trajectory(path, job)
        if not valid:
            invalid.append({"job_id": job.job_id, "reason": reason})
            invalid_reasons[job.job_id] = reason
            continue
        valid_job_ids.add(job.job_id)
        footer = records[-1]
        metrics = footer["metrics"]
        summary = footer.get("episode_summary") or {}
        steps = [record for record in records if record.get("kind") == "step"]
        actions = int(metrics.get("actions_used") or summary.get("actions_taken") or 0)
        budget = int(metrics.get("action_budget") or summary.get("budget") or 0)
        expanded.append({
            "model_identity": job.model.identity,
            "model_name": job.model.name,
            "provider": job.model.provider,
            "model": job.model.model,
            "policy": job.policy,
            "level": LEVEL_NAMES[job.case.level],
            "solved": bool(metrics.get("solved")),
            "composite": float(metrics.get("composite_score") or 0.0),
            "accusation": float(metrics.get("accusation_score") or 0.0),
            "triangle": float(metrics.get("triangle_score") or 0.0),
            "alibi": float(metrics.get("alibi_score") or 0.0),
            "elimination": float(metrics.get("elimination_score") or 0.0),
            "actions": actions,
            "input_tokens": int(metrics.get("input_tokens") or 0),
            "output_tokens": int(metrics.get("output_tokens") or 0),
            "total_tokens": int(metrics.get("total_tokens") or 0),
            "budget_exhausted": budget > 0 and actions >= budget,
            "guard_interventions": sum(
                bool(step.get("guard_intervention")) for step in steps
            ),
        })

    expected_groups: dict[tuple[str, str, str], list[Job]] = defaultdict(list)
    for job in jobs:
        expected_groups[(job.model.identity, job.policy, LEVEL_NAMES[job.case.level])].append(job)
    valid_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in expanded:
        valid_groups[(
            row["model_identity"], row["policy"], row["level"],
        )].append(row)

    rows: list[dict[str, Any]] = []
    model_by_identity = {job.model.identity: job.model for job in jobs}
    for key in sorted(
        expected_groups,
        key=lambda value: (value[0], value[1], LEVEL_VALUES[value[2]]),
    ):
        expected = expected_groups[key]
        valid = valid_groups.get(key, [])
        model = model_by_identity[key[0]]
        group_errors = [
            invalid_reasons[job.job_id]
            for job in expected
            if job.job_id in invalid_reasons
        ]
        rows.append({
            "model_name": model.name,
            "provider": model.provider,
            "model": model.model,
            "policy": key[1],
            "level": key[2],
            "attempted_n": len(expected),
            "n": len(valid),
            "solve_rate": _safe_mean(valid, "solved"),
            "composite": _safe_mean(valid, "composite"),
            "accusation": _safe_mean(valid, "accusation"),
            "triangle": _safe_mean(valid, "triangle"),
            "alibi": _safe_mean(valid, "alibi"),
            "elimination": _safe_mean(valid, "elimination"),
            "avg_actions": _safe_mean(valid, "actions"),
            "avg_input_tokens": _safe_mean(valid, "input_tokens"),
            "avg_output_tokens": _safe_mean(valid, "output_tokens"),
            "avg_total_tokens": _safe_mean(valid, "total_tokens"),
            "errors": len(expected) - len(valid),
            "timeouts": sum("timeout" in reason.lower() for reason in group_errors),
            "budget_exhausted": sum(row["budget_exhausted"] for row in valid),
            "avg_guard_interventions": _safe_mean(valid, "guard_interventions"),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8",
    )
    if rows:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    validation = {
        "expected": len(jobs),
        "valid": len(expanded),
        "complete": len(expanded) == len(jobs),
        "invalid": invalid,
        "valid_job_ids_sha256": _stable_hash(sorted(valid_job_ids)),
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8",
    )
    return validation


def _models_from_args(args: argparse.Namespace) -> list[ModelSpec]:
    available = {
        "claude": ModelSpec("claude", "anthropic", args.claude_model),
        "gemini": ModelSpec("gemini", "google", args.gemini_model),
        "gpt4o": ModelSpec("gpt4o", "openai", args.gpt4o_model),
    }
    return [available[name] for name in args.models]


def _verify_credentials(models: list[ModelSpec]) -> None:
    for model in models:
        # Resolve only to verify presence. Never log, serialize, or return keys.
        LLMConfig(provider=model.provider, model=model.model).resolved_api_key()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired API-model Vanilla vs Guarded benchmark",
    )
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "data/benchmark_v1")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/claude_gemini_vanilla_guarded",
    )
    parser.add_argument("--experiment-id", default="claude_gemini_vanilla_guarded_v1")
    parser.add_argument(
        "--models", nargs="+", choices=("claude", "gemini", "gpt4o"),
        default=["claude", "gemini"],
    )
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--levels", nargs="+", default=list(LEVEL_NAMES.values()))
    parser.add_argument("--per-level", type=int, default=None)
    parser.add_argument("--claude-model", default=os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL))
    parser.add_argument("--gemini-model", default=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
    parser.add_argument("--gpt4o-model", default=os.environ.get("GPT4O_MODEL", DEFAULT_GPT4O_MODEL))
    parser.add_argument("--claude-workers", type=int, default=4)
    parser.add_argument("--gemini-workers", type=int, default=8)
    parser.add_argument("--gpt4o-workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--history-window", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument(
        "--retry-rounds", type=int, default=1,
        help="Whole-episode retries after per-call retries are exhausted",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate manifest, matrix, and key presence without API calls",
    )
    args = parser.parse_args()

    if args.per_level is not None and args.per_level <= 0:
        parser.error("--per-level must be positive")
    if min(args.claude_workers, args.gemini_workers, args.gpt4o_workers) <= 0:
        parser.error("provider worker counts must be positive")
    if args.history_window < 0 or args.max_tokens <= 0:
        parser.error("history window must be nonnegative and max tokens positive")

    levels = _parse_levels(args.levels)
    models = _models_from_args(args)
    _verify_credentials(models)
    cases = load_cases(args.benchmark_dir.resolve(), levels, args.per_level)
    os.environ["MYSTERY_LLM_HISTORY_WINDOW"] = str(args.history_window)

    config = {
        "experiment_id": args.experiment_id,
        "benchmark_dir": str(args.benchmark_dir.resolve()),
        "manifest_sha256": _sha256(args.benchmark_dir.resolve() / "manifest.json"),
        "models": [model.__dict__ for model in models],
        "policies": args.policies,
        "guard_version": GUARD_VERSION,
        "levels": [LEVEL_NAMES[level] for level in sorted(levels)],
        "cases": len(cases),
        "jobs": len(cases) * len(models) * len(args.policies),
        "npc": {"provider": "fallback", "seed": 42},
        "decoding": {
            "max_tokens": args.max_tokens,
            "history_window": args.history_window,
            "temperature": None,
        },
        "retries": {
            "per_call": args.max_retries,
            "whole_episode_rounds": args.retry_rounds,
            "timeout_seconds": args.timeout_seconds,
        },
    }
    config["config_fingerprint"] = _stable_hash({
        key: config[key]
        for key in (
            "experiment_id", "manifest_sha256", "models", "policies",
            "guard_version", "levels", "cases", "npc", "decoding",
        )
    })
    config["created_at"] = _now()
    jobs = build_jobs(
        cases, models, args.policies, config["config_fingerprint"],
    )
    print(
        f"Matrix: {len(cases)} cases x {len(models)} models x "
        f"{len(args.policies)} policies = {len(jobs)} jobs"
    )
    for model in models:
        print(f"  {model.name}: provider={model.provider} model={model.model}")
    print("  credentials: present (values are never logged)")
    if args.validate_only:
        print("Validation-only complete; no API calls were made.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8",
    )

    limits = {
        "claude": args.claude_workers,
        "gemini": args.gemini_workers,
        "gpt4o": args.gpt4o_workers,
    }
    semaphores = {
        name: threading.BoundedSemaphore(limits[name])
        for name in limits
    }
    blocked_models: set[str] = set()
    block_lock = threading.Lock()

    def run_limited(job: Job) -> JobResult:
        with block_lock:
            if job.model.name in blocked_models:
                return JobResult(job, "blocked", "provider blocked after fatal error")
        with semaphores[job.model.name]:
            with block_lock:
                if job.model.name in blocked_models:
                    return JobResult(job, "blocked", "provider blocked after fatal error")
            result = _run_job(args.output_dir, args.experiment_id, job, args)
            if result.error and _is_fatal_provider_error(result.error):
                with block_lock:
                    blocked_models.add(job.model.name)
            return result

    for retry_round in range(args.retry_rounds + 1):
        pending = [
            job for job in jobs
            if not validate_trajectory(trajectory_path(args.output_dir, job), job)[0]
            and job.model.name not in blocked_models
        ]
        if not pending:
            break
        print(
            f"Round {retry_round + 1}/{args.retry_rounds + 1}: "
            f"{len(pending)} incomplete jobs"
        )
        max_workers = sum(limits[model.name] for model in models)
        counts: Counter[str] = Counter()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_limited, job): job for job in pending}
            for completed, future in enumerate(as_completed(futures), 1):
                result = future.result()
                counts[result.status] += 1
                if result.error and result.status in {"error", "crash"}:
                    print(
                        f"  ! {result.job.job_id}: "
                        f"{result.error[:240].replace(chr(10), ' ')}"
                    )
                if completed % 25 == 0 or completed == len(pending):
                    print(f"  [{completed}/{len(pending)}] {dict(counts)}")
        validation = write_reports(args.output_dir, jobs)
        print(f"  valid: {validation['valid']}/{validation['expected']}")
        if blocked_models:
            print(
                "Fatal provider configuration error; blocked remaining jobs for: "
                + ", ".join(sorted(blocked_models))
            )
            break

    validation = write_reports(args.output_dir, jobs)
    print(f"Output: {args.output_dir}")
    print(f"Complete: {validation['complete']} ({validation['valid']}/{validation['expected']})")
    return 0 if validation["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
