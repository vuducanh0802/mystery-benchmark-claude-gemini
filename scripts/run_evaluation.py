"""
Run agent evaluation on a generated benchmark suite.

Usage examples:
    # Heuristic baseline (no API key needed):
    uv run scripts/run_evaluation.py --benchmark-dir data/benchmark_v1/ --agent heuristic --output-dir results/heuristic/

    # Claude (Anthropic):
    uv run scripts/run_evaluation.py --benchmark-dir data/benchmark_v1/ --agent claude --output-dir results/claude/

    # ChatGPT (OpenAI):
    uv run scripts/run_evaluation.py --benchmark-dir data/benchmark_v1/ --agent chatgpt --output-dir results/chatgpt/

    # Gemini (Google):
    uv run scripts/run_evaluation.py --benchmark-dir data/benchmark_v1/ --agent gemini --output-dir results/gemini/

    # With LLM-powered NPC interviews (requires vLLM or compatible endpoint):
    uv run scripts/run_evaluation.py --benchmark-dir data/benchmark_v1/ --agent claude \\
        --npc-url http://localhost:8000/v1 --npc-model Qwen/Qwen2.5-27B-Instruct \\
        --output-dir results/claude_npc/
"""

from __future__ import annotations

import argparse
import json
import structlog
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.heuristic_agent import HeuristicAgent
from agents.llm_agent import LLMAgent
from benchmark.generate import load_benchmark_suite
from evaluation.runner import run_benchmark

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

AGENT_CONFIGS = {
    "heuristic": {
        "provider": None,
        "model": None,
        "description": "Rule-based heuristic agent (no LLM, no API key needed)",
    },
    "claude": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "description": "Claude Sonnet via Anthropic API (ANTHROPIC_API_KEY)",
    },
    "chatgpt": {
        "provider": "openai",
        "model": "gpt-4o",
        "description": "GPT-4o via OpenAI API (OPENAI_API_KEY)",
    },
    "gemini": {
        "provider": "google",
        "model": "gemini-2.0-flash",
        "description": "Gemini 2.0 Flash via Google API (GOOGLE_API_KEY)",
    },
}


def make_agent_factory(agent_name: str, model_override: str | None):
    cfg = AGENT_CONFIGS[agent_name]

    def factory():
        if agent_name == "heuristic":
            return HeuristicAgent(agent_id="heuristic")
        model = model_override or cfg["model"]
        return LLMAgent(
            agent_id=agent_name,
            provider=cfg["provider"],
            model=model,
        )

    return factory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MysteryBench / MysteryArena evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {name:10s} — {cfg['description']}"
            for name, cfg in AGENT_CONFIGS.items()
        ),
    )
    parser.add_argument("--benchmark-dir", dest="benchmark", required=True, help="Path to benchmark directory")
    parser.add_argument(
        "--agent",
        choices=list(AGENT_CONFIGS.keys()),
        default="heuristic",
        help="Agent to evaluate",
    )
    parser.add_argument("--model", default=None, help="Override default model for the chosen agent")
    parser.add_argument("--output-dir", dest="output", required=True, help="Output directory for results")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument(
        "--levels",
        default=None,
        help="Comma-separated complexity levels to run, e.g. '1,2,3'",
    )
    # NPC interview LLM
    parser.add_argument(
        "--npc-url",
        default=None,
        help="OpenAI-compatible base URL for NPC responder (e.g. http://localhost:8000/v1). "
            "If omitted, interviews use the deterministic fallback.",
    )
    parser.add_argument(
        "--npc-model",
        default="Qwen/Qwen2.5-27B-Instruct",
        help="Model served at --npc-url (default: Qwen/Qwen2.5-27B-Instruct)",
    )
    parser.add_argument(
        "--npc-seed",
        type=int,
        default=42,
        help="Fixed seed for NPC responses (default: 42)",
    )
    parser.add_argument(
        "--trajectory-dir",
        default=None,
        help="If set, write JSONL trajectory logs (one file per episode) for replay/reproducibility.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip seeds whose trajectory file already exists (resume-on-crash).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load instances
    logger.info(f"Loading benchmark from {args.benchmark}")
    instances = load_benchmark_suite(args.benchmark)

    if args.levels:
        target = {int(l) for l in args.levels.split(",")}
        instances = [(ws, lvl) for ws, lvl in instances if lvl in target]

    if args.max_instances:
        instances = instances[: args.max_instances]

    logger.info(f"Instances: {len(instances)} | Agent: {args.agent}")

    # NPC responder
    npc_responder = None
    if args.npc_url:
        from mystery_world.npc_responder import NPCResponder
        npc_responder = NPCResponder(
            base_url=args.npc_url,
            model=args.npc_model,
            seed=args.npc_seed,
        )
        logger.info(f"NPC responder: {args.npc_model} @ {args.npc_url} (seed={args.npc_seed})")
    else:
        logger.info("NPC responder: deterministic fallback (no --npc-url provided)")

    # Run
    cfg = AGENT_CONFIGS[args.agent]
    traj_meta = {
        "agent": args.agent,
        "model": args.model or cfg.get("model"),
        "provider": cfg.get("provider"),
        "npc_provider": "vllm" if args.npc_url else "fallback",
        "npc_model": args.npc_model if args.npc_url else None,
        "npc_seed": args.npc_seed if args.npc_url else None,
    }
    results = run_benchmark(
        agent_factory=make_agent_factory(args.agent, args.model),
        instances=instances,
        output_dir=args.output,
        verbose=args.verbose,
        npc_responder=npc_responder,
        trajectory_dir=args.trajectory_dir,
        trajectory_meta=traj_meta,
        skip_existing=args.skip_existing,
    )

    # Summary
    solved = sum(1 for r in results if r.metrics and r.metrics.solved)
    total = len(results)
    errors = sum(1 for r in results if r.error)
    partial_scores = [
        r.metrics.to_dict().get("partial_score", 0.0)
        for r in results
        if r.metrics
    ]
    avg_partial = sum(partial_scores) / max(len(partial_scores), 1)

    summary = {
        "agent": args.agent,
        "model": args.model or AGENT_CONFIGS[args.agent].get("model"),
        "total_instances": total,
        "solved": solved,
        "solve_rate": round(solved / max(total, 1), 4),
        "avg_partial_score": round(avg_partial, 4),
        "errors": errors,
        "npc_model": args.npc_model if args.npc_url else "fallback",
    }

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 50)

    Path(args.output, "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()