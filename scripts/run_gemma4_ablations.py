#!/usr/bin/env python3
"""Gemma-4-31B guarded ablations on the frozen, paired 1,000-case suite."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import threading
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.guard_ablations import VARIANTS, GuardVariant
from scripts import run_gpt_oss_ablations as ablation
from scripts import run_gpt_oss_kimi_baselines as paired
from scripts import run_claude_gemini_baselines as core


MODEL = "google/gemma-4-31B-it"
SOURCE_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
INFERENCE_REVISION = "c1ac76e99d5513b141e8adde7288b85c3f9c32ec"
STRING_OR_NULL = {"anyOf": [{"type": "string"}, {"type": "null"}]}
# Same structured output constraints as the earlier local Gemma baseline.
DETECTIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 240},
        "beliefs": {
            "type": "object",
            "properties": {
                "top_suspect": STRING_OR_NULL,
                "suspect_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "top_weapon": STRING_OR_NULL,
                "weapon_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "top_location": STRING_OR_NULL,
                "location_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "eliminated_suspects": {"type": "array", "items": {"type": "string"}},
                "new_facts": {"type": "array", "maxItems": 5,
                              "items": {"type": "string", "maxLength": 160}},
            },
            "required": ["top_suspect", "suspect_confidence", "top_weapon", "weapon_confidence",
                         "top_location", "location_confidence", "eliminated_suspects", "new_facts"],
            "additionalProperties": False,
        },
        "action": {"type": "string", "enum": [
            "MOVE", "EXAMINE_LOCATION", "EXAMINE_OBJECT", "TALK_TO", "TAKE_OBJECT",
            "CHECK_INVENTORY", "WAIT", "ACCUSE",
        ]},
        "action_args": {"type": "object"},
    },
    "required": ["reasoning", "beliefs", "action", "action_args"],
    "additionalProperties": False,
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    p.add_argument("--include-guarded-reference", action="store_true",
                   help="Also run Full Guarded on these exact cases and current guard implementation")
    p.add_argument("--benchmark-dir", type=Path, default=ROOT / "data/benchmark_v1")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/gemma4_guarded_ablations")
    p.add_argument("--experiment-id", default="gemma4_guarded_ablations_v1")
    p.add_argument("--levels", nargs="+", default=list(core.LEVEL_NAMES.values()))
    p.add_argument("--per-level", type=int, default=200)
    p.add_argument("--model", default=os.getenv("GEMMA4_MODEL", MODEL))
    p.add_argument("--base-urls", nargs="+", default=os.getenv(
        "GEMMA4_BASE_URLS", "http://127.0.0.1:8332/v1").split(","))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--history-window", type=int, default=4)
    p.add_argument("--max-accuse-blocks", type=int, default=2)
    p.add_argument("--timeout-seconds", type=float, default=300)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--retry-backoff-seconds", type=float, default=1)
    p.add_argument("--retry-rounds", type=int, default=2)
    p.add_argument("--serving-metadata", type=Path,
                   help="Serving provenance JSON produced by the local launcher")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    args = p.parse_args(argv)
    if len(set(args.variants)) != len(args.variants):
        p.error("variants must not contain duplicates")
    if min(args.per_level, args.workers, args.max_tokens) <= 0:
        p.error("per-level, workers and max-tokens must be positive")
    if min(args.history_window, args.max_accuse_blocks, args.max_retries, args.retry_rounds) < 0:
        p.error("history, accusation blocks and retries must be nonnegative")
    if (not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0
            or not math.isfinite(args.retry_backoff_seconds) or args.retry_backoff_seconds < 0):
        p.error("invalid timeout or retry backoff")
    if not 0 <= args.temperature <= 2:
        p.error("temperature must be between 0 and 2")
    args.base_urls = [url.strip().rstrip("/") for url in args.base_urls]
    if not args.base_urls or len(set(args.base_urls)) != len(args.base_urls):
        p.error("base-urls must be nonempty and distinct")
    for url in args.base_urls:
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            p.error("base-urls must be HTTP(S) URLs without credentials, queries or fragments")
    args.models = ["gemma4"]
    args.policies = (["guarded"] if args.include_guarded_reference else []) + args.variants
    args.worker_limits = {"gemma4": args.workers}
    args.replica_counter = itertools.count()
    args.replica_lock = threading.Lock()
    args.endpoints = {"gemma4": paired.Endpoint(
        core.ModelSpec("gemma4", "openai", args.model), args.base_urls[0],
        "GEMMA4_API_KEY", args.temperature,
        {"response_format": {"type": "json_schema", "json_schema": {
            "name": "mystery_detective_action", "strict": True, "schema": DETECTIVE_SCHEMA,
        }}},
        os.getenv("GEMMA4_REVISION", "unspecified"),
    )}
    return args


def make_agent(job, args):
    endpoint = args.endpoints[job.model.name]
    with args.replica_lock:
        index = next(args.replica_counter)
    endpoint = replace(endpoint, base_url=args.base_urls[index % len(args.base_urls)])
    if job.policy == "guarded":
        cls, extra = paired.Guarded, {}
    else:
        cls, extra = ablation.Ablation, {"variant": job.policy}
    agent = cls(
        **extra, max_accuse_blocks=args.max_accuse_blocks,
        agent_id=job.cell_id, provider="openai", model=job.model.model,
        base_url=endpoint.base_url, api_key=os.getenv(endpoint.api_key_env) or "EMPTY",
        max_tokens=args.max_tokens, max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds, timeout_seconds=args.timeout_seconds,
    )
    agent.config.temperature = endpoint.temperature
    agent.config.extra_body = endpoint.extra_body.copy()
    return agent


def build_config(args, cases):
    variants = {name: VARIANTS[name].metadata(args.max_accuse_blocks) for name in args.variants}
    if args.include_guarded_reference:
        variants["guarded"] = GuardVariant(True, True, "evidence_gate").metadata(args.max_accuse_blocks)
    config = ablation.build_ablation_config(args, cases, variants=variants, reference_release={
        "case_suite": "benchmark_suites/claude_gemini_cases_v1.tar.gz",
        "case_identity_and_bytes_verified": True,
        "historical_gemma_guarded_is_paired": False,
        "historical_comparison": "Different serialized worlds and guard implementation; use a matched Full Guarded arm",
    })
    config.pop("config_fingerprint")
    created = config.pop("created_at")
    config["replica_base_urls"] = args.base_urls
    config["serving"] = (json.loads(args.serving_metadata.read_text()) if args.serving_metadata else {
        "weights": "not verified by endpoint client", "reasoning": "must be disabled on the server",
    })
    config["config_fingerprint"] = core._stable_hash(config)
    config["created_at"] = created
    return config


def main(argv=None):
    args = parse_args(argv)
    # Check credentials for every replica, including nonlocal secondary URLs.
    if not args.validate_only and not args.report_only:
        paired.check_credentials({str(i): replace(args.endpoints["gemma4"], base_url=url)
                                  for i, url in enumerate(args.base_urls)})
    if args.preflight_only:
        for url in args.base_urls:
            single = parse_args(argv)
            single.base_urls = [url]
            single.endpoints["gemma4"] = replace(single.endpoints["gemma4"], base_url=url)
            code = paired.execute(single, agent_factory=make_agent, config_factory=build_config,
                                  report_writer=ablation.write_reports)
            if code:
                return code
        return 0
    return paired.execute(args, agent_factory=make_agent, config_factory=build_config,
                          report_writer=ablation.write_reports)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2)
