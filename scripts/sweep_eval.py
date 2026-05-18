"""
Sweep one detective agent across N levels x M seeds with concurrency + resume.

Each (detective, level, seed) emits one JSONL trajectory at:
    {trajectory_dir}/{detective}/{level}/seed_{seed}.jsonl

Re-running the same command skips seeds whose JSONL already exists.

Examples:
    # Heuristic baseline, all 5 levels, 200 seeds, 8 workers
    uv run scripts/sweep_eval.py --detective-agent heuristic \
        --levels TRIVIAL EASY MEDIUM HARD EXPERT --seeds 0-199 \
        --trajectory-dir results/trajectories --workers 8

    # Claude via Anthropic native
    uv run scripts/sweep_eval.py --detective-agent claude --detective-model claude-sonnet-4-6 \
        --levels TRIVIAL EASY MEDIUM HARD EXPERT --seeds 0-199 \
        --trajectory-dir results/trajectories --workers 8

    # Qwen3.5 via OpenRouter
    uv run scripts/sweep_eval.py --detective-agent openrouter --detective-model qwen/qwen3.5-27b \
        --levels TRIVIAL EASY MEDIUM HARD EXPERT --seeds 0-199 \
        --trajectory-dir results/trajectories --workers 8

    # With OpenAI-direct NPCs (gpt-4o-mini)
    uv run scripts/sweep_eval.py --detective-agent claude ... \
        --npc-provider openai --npc-model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.culprit_agent import LLMCulpritAgent
from agents.detective_agent import HeuristicAgent, LLMDetectiveAgent, OracleAgent
from evaluation.runner import run_episode
from evaluation.trajectory import TrajectoryWriter
from mystery_world import COMPLEXITY_PRESETS, ComplexityLevel
from mystery_world.generator import generate_mystery


AGENT_CONFIGS = {
    "heuristic":          {"provider": None,         "model": None},
    "oracle_min":         {"provider": None,         "model": None},
    "oracle_max":         {"provider": None,         "model": None},
    "claude":             {"provider": "anthropic",  "model": "claude-sonnet-4-6"},
    "claude-opus":        {"provider": "anthropic",  "model": "claude-opus-4-7"},
    "chatgpt":            {"provider": "openai",     "model": "gpt-4o"},
    "chatgpt-mini":       {"provider": "openai",     "model": "gpt-4o-mini"},
    "gemini":             {"provider": "google",     "model": "gemini-2.0-flash"},
    "openrouter":         {"provider": "openrouter", "model": "qwen/qwen3.5-27b"},
}


def _make_detective_agent(detective_agent_name: str, model: str | None):
    cfg = AGENT_CONFIGS[detective_agent_name]
    if detective_agent_name == "heuristic":
        return HeuristicAgent(agent_id="heuristic")
    if detective_agent_name == "oracle_min":
        return OracleAgent(agent_id="oracle_min", mode="min_action")
    if detective_agent_name == "oracle_max":
        return OracleAgent(agent_id="oracle_max", mode="max_score")
    return LLMDetectiveAgent(
        agent_id=detective_agent_name,
        provider=cfg["provider"],
        model=model or cfg["model"],
    )


def _parse_seeds(spec: str) -> list[int]:
    """Accepts '0-199' or '0,1,5,10' or single int."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _build_npc_responder(args) -> object | None:
    if not args.npc_provider or args.npc_provider == "fallback":
        return None
    from mystery_world.npc_responder import NPCResponder
    if args.npc_provider == "openai":
        # OpenAI direct: leave base_url=None (default api.openai.com).
        return NPCResponder(base_url=None, model=args.npc_model, seed=args.npc_seed,
                            api_key_env="OPENAI_API_KEY")
    if args.npc_provider == "openrouter":
        return NPCResponder(base_url="https://openrouter.ai/api/v1",
                            model=args.npc_model, seed=args.npc_seed,
                            api_key_env="OPENROUTER_API_KEY")
    if args.npc_provider == "vllm":
        return NPCResponder(base_url=args.npc_url, model=args.npc_model, seed=args.npc_seed)
    raise ValueError(f"unknown --npc-provider {args.npc_provider}")


def _build_culprit_agent(args) -> object | None:
    if not args.culprit_model:
        return None
    return LLMCulpritAgent(
        agent_id="culprit",
        provider=args.culprit_provider,
        model=args.culprit_model,
    )


def _run_one(args, level_name: str, seed: int, traj_path: Path) -> tuple[str, int, str]:
    if traj_path.exists():
        return (level_name, seed, "skipped")
    config = COMPLEXITY_PRESETS[ComplexityLevel[level_name]]
    state = generate_mystery(seed=seed, config=config)
    detective_agent = _make_detective_agent(
        args.detective_agent, args.detective_model,
    )
    npc = _build_npc_responder(args)
    culprit = _build_culprit_agent(args)
    if culprit is not None:
        state.config = replace(state.config, free_culprit_actions=True)
    cfg = AGENT_CONFIGS[args.detective_agent]

    # Provenance: when routed through a LiteLLM gateway, --litellm-model (if set)
    # is the actual model for EVERY role, and the transport is the gateway — record
    # that, not the AGENT_CONFIGS default, so trajectory headers aren't mislabeled.
    eff_model = args.litellm_model or args.detective_model or cfg.get("model")
    eff_provider = "litellm" if args.litellm_url else cfg.get("provider")
    npc_active = args.npc_provider not in (None, "fallback")
    eff_npc_model = (args.litellm_model or args.npc_model) if npc_active else None

    traj_path.parent.mkdir(parents=True, exist_ok=True)
    with TrajectoryWriter(traj_path) as w:
        w.write_header(
            state=state,
            level=level_name,
            agent=args.detective_agent,
            model=eff_model,
            provider=eff_provider,
            npc_provider=args.npc_provider or "fallback",
            npc_model=eff_npc_model,
            npc_seed=args.npc_seed if npc_active else None,
            instance_id=f"seed_{seed}",
        )
        try:
            result = run_episode(
                detective_agent, state, complexity_level=ComplexityLevel[level_name].value,
                verbose=False, npc_responder=npc, culprit_agent=culprit,
                trajectory_writer=w,
            )
            w.write_footer(
                episode_summary=result.episode_summary,
                metrics=result.metrics.to_dict() if result.metrics else None,
                elapsed_seconds=result.elapsed_seconds,
                error=result.error,
            )
            return (level_name, seed, "ok" if not result.error else f"error:{result.error[:60]}")
        except Exception as e:
            w.write_footer(
                episode_summary={}, metrics=None, elapsed_seconds=0.0,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1000:]}",
            )
            return (level_name, seed, f"crash:{type(e).__name__}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--detective-agent",
        "--agent",
        dest="detective_agent",
        required=True,
        choices=list(AGENT_CONFIGS.keys()),
    )
    p.add_argument("--detective-model", "--model", dest="detective_model", default=None)
    p.add_argument("--levels", nargs="+", default=["TRIVIAL", "EASY", "MEDIUM", "HARD", "EXPERT"])
    p.add_argument("--seeds", default="0-199", help="e.g. '0-199' or '0,1,2'")
    p.add_argument("--trajectory-dir", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--npc-provider", default="fallback",
                   choices=["fallback", "openai", "openrouter", "vllm"])
    p.add_argument("--npc-model", default="gpt-4o-mini")
    p.add_argument("--npc-url", default=None, help="vLLM base URL (only used with --npc-provider vllm)")
    p.add_argument("--npc-seed", type=int, default=42)
    p.add_argument("--culprit-provider", default="openai",
                   choices=["anthropic", "openai", "google", "openrouter"])
    p.add_argument("--culprit-model", default=None,
                   help="Enable a free-acting LLM culprit using this model.")
    p.add_argument("--litellm-url", default=None,
                   help="Route ALL roles (detective + NPC) through this LiteLLM "
                        "(OpenAI-compatible) gateway. Single injection point.")
    p.add_argument("--litellm-key-env", default=None,
                   help="Env var holding the LiteLLM key (omit for keyless proxy).")
    p.add_argument("--litellm-model", default=None,
                   help="Optional model alias applied to every role; if omitted, "
                        "each role keeps its own (--model / --npc-model).")
    args = p.parse_args()

    if args.litellm_url:
        from agents.base_agent import BaseAgent
        BaseAgent.configure_litellm(
            args.litellm_url,
            api_key_env=args.litellm_key_env,
            model=args.litellm_model,
        )
        print(f"LiteLLM gateway: {args.litellm_url} "
              f"(key_env={args.litellm_key_env or 'none'}, "
              f"model={args.litellm_model or 'per-role'})")

    seeds = _parse_seeds(args.seeds)
    base = Path(args.trajectory_dir) / args.detective_agent
    jobs: list[tuple[str, int, Path]] = []
    for lvl in args.levels:
        for s in seeds:
            jobs.append((lvl, s, base / lvl / f"seed_{s}.jsonl"))

    total = len(jobs)
    banner_model = (
        args.litellm_model
        or args.detective_model
        or AGENT_CONFIGS[args.detective_agent].get("model")
    )
    via = " via litellm" if args.litellm_url else ""
    print(f"Sweep: detective={args.detective_agent} model={banner_model}{via} "
          f"jobs={total} workers={args.workers}")
    print(f"Output: {base}")

    counts = {"ok": 0, "skipped": 0, "error": 0, "crash": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, args, lvl, s, p): (lvl, s) for (lvl, s, p) in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            lvl, s = futs[fut]
            _, _, status = fut.result()
            head = status.split(":", 1)[0]
            counts[head] = counts.get(head, 0) + 1
            if i % 25 == 0 or i == total:
                print(f"  [{i}/{total}] last={lvl} seed={s} status={status}")
    print(f"Done. {counts}")
    return 0 if counts.get("crash", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
