#!/usr/bin/env python3
"""Paired GPT-OSS/Kimi evaluation against already running compatible endpoints.

This program never downloads weights or starts a model server.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import run_claude_gemini_baselines as core
from agents.llm_agent import BiasGuardedLLMDetectiveAgent, LLMDetectiveAgent


@dataclass(frozen=True)
class Endpoint:
    model: core.ModelSpec
    base_url: str
    api_key_env: str
    temperature: float | None
    extra_body: dict
    revision: str

    def metadata(self) -> dict:
        return {
            **self.model.__dict__, "base_url": self.base_url,
            "api_key_env": self.api_key_env, "temperature": self.temperature,
            "extra_body": self.extra_body, "revision": self.revision,
        }


class SanitizedTransport:
    """Keep credentials out of exception strings persisted by the episode runner."""
    def _complete(self, system, user):
        try:
            return super()._complete(system, user)
        except Exception as exc:
            message = str(exc)
            key = self.config.resolved_api_key()
            if key and key != "EMPTY":
                message = message.replace(key, "[REDACTED]")
            raise RuntimeError(f"{type(exc).__name__}: {message}") from None


class Vanilla(SanitizedTransport, LLMDetectiveAgent):
    pass


class Guarded(SanitizedTransport, BiasGuardedLLMDetectiveAgent):
    pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", choices=["gpt-oss", "kimi"], default=["gpt-oss", "kimi"])
    p.add_argument("--policies", nargs="+", choices=core.POLICIES, default=list(core.POLICIES))
    p.add_argument("--levels", nargs="+", default=list(core.LEVEL_NAMES.values()))
    p.add_argument("--per-level", type=int)
    p.add_argument("--benchmark-dir", type=Path, default=ROOT / "data/benchmark_v1")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/gpt_oss_kimi_vanilla_guarded")
    p.add_argument("--experiment-id", default="gpt_oss_kimi_paired_v1")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--history-window", type=int, default=10)
    p.add_argument("--timeout-seconds", type=float, default=300)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--retry-backoff-seconds", type=float, default=1)
    p.add_argument("--retry-rounds", type=int, default=0)
    p.add_argument("--validate-only", action="store_true", help="Offline checks; no endpoint calls")
    p.add_argument("--preflight-only", action="store_true", help="One short completion per selected endpoint; no episodes")
    p.add_argument("--report-only", action="store_true", help="Rebuild reports without endpoint calls or credentials")
    for name, prefix, model, port in (
        ("gpt-oss", "GPT_OSS", "openai/gpt-oss-120b", 8000),
        ("kimi", "KIMI", "moonshotai/Kimi-K2.5", 8001),
    ):
        p.add_argument(f"--{name}-model", default=os.getenv(f"{prefix}_MODEL", model))
        p.add_argument(f"--{name}-base-url", default=os.getenv(f"{prefix}_BASE_URL", f"http://localhost:{port}/v1"))
        p.add_argument(f"--{name}-workers", type=int, default=1)
    args = p.parse_args(argv)
    if len(set(args.models)) != len(args.models) or len(set(args.policies)) != len(args.policies):
        p.error("models and policies must not contain duplicates")
    if args.per_level is not None and args.per_level <= 0:
        p.error("per-level must be positive")
    if min(args.gpt_oss_workers, args.kimi_workers, args.max_tokens) <= 0:
        p.error("workers and max-tokens must be positive")
    if min(args.history_window, args.max_retries, args.retry_rounds, args.retry_backoff_seconds) < 0 or args.timeout_seconds <= 0:
        p.error("invalid history/retry/timeout settings")
    if sum([args.validate_only, args.preflight_only, args.report_only]) > 1:
        p.error("choose only one of validate-only/preflight-only/report-only")
    return args


def endpoints_from_args(args):
    endpoints = {}
    for name in args.models:
        prefix = "GPT_OSS" if name == "gpt-oss" else "KIMI"
        attr = name.replace("-", "_")
        url = getattr(args, attr + "_base_url").rstrip("/")
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"{prefix}_BASE_URL must be an HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(f"{prefix}_BASE_URL must not contain credentials, query parameters, or fragments")
        extra = json.loads(os.getenv(f"{prefix}_EXTRA_BODY", "{}"))
        if not isinstance(extra, dict):
            raise ValueError(f"{prefix}_EXTRA_BODY must be a JSON object")
        # Only decoding controls: prevent accidental key persistence or overriding
        # the case messages/model through OpenAI extra_body's merge behavior.
        allowed = {"reasoning_effort", "thinking", "chat_template_kwargs", "top_p", "top_k", "min_p", "repetition_penalty"}
        if set(extra) - allowed:
            raise ValueError(f"{prefix}_EXTRA_BODY only accepts decoding controls: {sorted(allowed)}")
        for key, value in extra.items():
            if key == "thinking" and (not isinstance(value, dict) or set(value) - {"type"} or value.get("type") not in {"enabled", "disabled"}):
                raise ValueError("thinking must contain type=enabled or disabled")
            if key == "chat_template_kwargs" and (not isinstance(value, dict) or set(value) - {"enable_thinking", "reasoning_effort"}):
                raise ValueError("chat_template_kwargs only accepts enable_thinking and reasoning_effort")
        raw_temp = os.getenv(f"{prefix}_TEMPERATURE", "").strip()
        temperature = float(raw_temp) if raw_temp else None
        if temperature is not None and not 0 <= temperature <= 2:
            raise ValueError(f"{prefix}_TEMPERATURE must be between 0 and 2")
        endpoints[name] = Endpoint(
            core.ModelSpec(name, "openai", getattr(args, attr + "_model")),
            url, f"{prefix}_API_KEY", temperature, extra,
            os.getenv(f"{prefix}_REVISION", "unspecified"),
        )
    return endpoints


def check_credentials(endpoints):
    for endpoint in endpoints.values():
        host = urlsplit(endpoint.base_url).hostname
        if host not in {"localhost", "127.0.0.1", "::1"} and not os.getenv(endpoint.api_key_env):
            raise ValueError(f"Set {endpoint.api_key_env}; use EMPTY explicitly for an unauthenticated remote server")


def make_agent(job, args):
    endpoint = args.endpoints[job.model.name]
    cls = Guarded if job.policy == "guarded" else Vanilla
    agent = cls(
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


def source_hash():
    files = sorted({
        p for directory in ("agents", "mystery_world", "evaluation", "scripts")
        for p in (ROOT / directory).rglob("*.py")
    })
    return core._stable_hash({str(p.relative_to(ROOT)): core._sha256(p) for p in files})


def build_config(args, cases):
    config = {
        "experiment_id": args.experiment_id,
        "manifest_sha256": core._sha256(args.benchmark_dir / "manifest.json"),
        "case_hashes": {case.instance_id: case.sha256 for case in cases},
        "models": [e.metadata() for e in args.endpoints.values()],
        "policies": args.policies,
        "guard_version": core.GUARD_VERSION,
        "source_sha256": source_hash(),
        "npc": {"provider": "fallback", "seed": 42, "culprit": "passive"},
        "decoding": {"max_tokens": args.max_tokens, "history_window": args.history_window},
        "retry_settings": {"max_retries": args.max_retries, "retry_backoff_seconds": args.retry_backoff_seconds, "timeout_seconds": args.timeout_seconds},
    }
    config["config_fingerprint"] = core._stable_hash(config)
    config["cases"] = len(cases)
    config["jobs"] = len(cases) * len(args.models) * len(args.policies)
    config["created_at"] = core._now()
    return config


def main(argv=None):
    args = parse_args(argv)
    args.endpoints = endpoints_from_args(args)
    return execute(args)


def execute(
    args, *, agent_factory=make_agent, config_factory=build_config,
    report_writer=core.write_reports,
):
    """Shared execution loop for paired baselines and controlled ablations."""
    args.benchmark_dir = args.benchmark_dir.resolve()
    if not args.report_only and not args.validate_only:
        check_credentials(args.endpoints)
    levels = core._parse_levels(args.levels)
    cases = core.load_cases(args.benchmark_dir, levels, args.per_level)
    os.environ["MYSTERY_LLM_HISTORY_WINDOW"] = str(args.history_window)
    # Parse all selected worlds before any endpoint call.
    for case in cases:
        core.WorldState.load(case.path)
    config = config_factory(args, cases)
    jobs = core.build_jobs(cases, [e.model for e in args.endpoints.values()], args.policies, config["config_fingerprint"])
    print(f"Matrix: {len(cases)} cases x {len(args.models)} models x {len(args.policies)} policies = {len(jobs)} jobs")
    print("Transport: OpenAI-compatible endpoints; models are open-weight, not OpenAI-hosted.")
    for name, endpoint in args.endpoints.items():
        print(f"  {name}: {endpoint.model.model}; credential values omitted")
    if args.validate_only:
        print("Offline validation passed. No model calls or downloads.")
        return 0
    if args.preflight_only:
        for name in args.models:
            job = next(j for j in jobs if j.model.name == name)
            agent = agent_factory(job, args)
            text, tokens = agent._complete(
                'Return one JSON object with action="WAIT", action_args={}, beliefs={}.',
                "Connectivity check. Return the requested JSON object only.",
            )
            agent._parse_response(text)
            print(f"  {name}: JSON response and token usage verified ({tokens} tokens)")
        return 0

    path = args.output_dir / "run_config.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("config_fingerprint") != config["config_fingerprint"]:
            raise ValueError("Output directory belongs to a different configuration. Use a new OUTPUT_DIR; existing results were not modified.")
    elif args.report_only:
        raise ValueError("No run_config.json; nothing to report")
    elif any(args.output_dir.glob("trajectories/**/*.jsonl")):
        raise ValueError("Existing trajectories have no run_config.json; choose a new OUTPUT_DIR")
    if args.report_only:
        validation = report_writer(args.output_dir, jobs)
        print(f"Complete: {validation['complete']} ({validation['valid']}/{validation['expected']})")
        return 0 if validation["complete"] else 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(config, indent=2))

    limits = {"gpt-oss": args.gpt_oss_workers, "kimi": args.kimi_workers}
    semaphores = {name: threading.BoundedSemaphore(limits[name]) for name in args.models}
    blocked = set()
    lock = threading.Lock()

    def run_limited(job):
        with semaphores[job.model.name]:
            with lock:
                if job.model.name in blocked:
                    return core.JobResult(job, "blocked", "endpoint blocked after fatal error")
            result = core._run_job(args.output_dir, args.experiment_id, job, args, agent_factory=agent_factory)
            if result.error and (core._is_fatal_provider_error(result.error) or "BadRequestError" in result.error):
                with lock:
                    blocked.add(job.model.name)
            return result

    for round_number in range(args.retry_rounds + 1):
        pending = [j for j in jobs if j.model.name not in blocked and not core.validate_trajectory(core.trajectory_path(args.output_dir, j), j)[0]]
        if not pending:
            break
        print(f"Round {round_number + 1}: {len(pending)} incomplete episodes", flush=True)
        counts = Counter()
        with ThreadPoolExecutor(max_workers=sum(limits[n] for n in args.models)) as pool:
            futures = [pool.submit(run_limited, job) for job in pending]
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                counts[result.status] += 1
                if result.error and result.status != "blocked":
                    print(f"  ! {result.job.job_id}: {result.error[:240]}", flush=True)
                if index % 10 == 0 or index == len(pending):
                    print(f"  {index}/{len(pending)}: {dict(counts)}", flush=True)
        report_writer(args.output_dir, jobs)
    validation = report_writer(args.output_dir, jobs)
    print(f"Output: {args.output_dir}")
    print(f"Complete: {validation['complete']} ({validation['valid']}/{validation['expected']})")
    return 0 if validation["complete"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2)
