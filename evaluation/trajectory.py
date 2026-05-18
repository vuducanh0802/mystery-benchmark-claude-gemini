"""
JSONL trajectory logging for reproducibility & replay.

Format
------
File contains one JSON object per line:

  Line 0 (header):
    {
      "kind": "header",
      "schema_version": 1,
      "seed": int,
      "level": str,                  # ComplexityLevel.name
      "config": dict,                # ComplexityConfig.to_dict()
      "config_hash": str,            # sha256 of canonical config JSON
      "detective_agent": str,
      "detective_model": str | null,
      "detective_provider": str | null,
      "agent": str,                   # backward-compatible alias
      "model": str | null,            # backward-compatible alias
      "provider": str | null,         # backward-compatible alias
      "npc_provider": str | null,
      "npc_model": str | null,
      "npc_seed": int | null,
      "git_sha": str,
      "started_at": iso8601 str,
      "instance_id": str
    }

  Lines 1..N (steps):
    {
      "kind": "step",
      "step": int,
      "actor_id": str,
      "role": str,                    # detective | culprit | npc-like role
      "action": str,                 # AgentAction.name
      "action_kwargs": dict,
      "observation": str,            # observation passed INTO the agent for this step
      "model_response": str | null,  # raw LLM text, when available
      "result_observation": str,     # rendered observation AFTER the action
      "success": bool,
      "world_state_hash": str,       # sha256 of WorldState.to_dict() AFTER the action
      "timestamp": iso8601 str
    }

  Final line (footer):
    {
      "kind": "footer",
      "episode_summary": dict,
      "metrics": dict | null,
      "ended_at": iso8601 str,
      "elapsed_seconds": float,
      "error": str | null
    }
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from mystery_world.world import WorldState


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def world_state_hash(state: WorldState) -> str:
    return hashlib.sha256(_stable_json(state.to_dict()).encode()).hexdigest()


def config_hash(config_dict: dict) -> str:
    return hashlib.sha256(_stable_json(config_dict).encode()).hexdigest()


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class TrajectoryWriter:
    """Append-only JSONL writer for one episode."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._closed = False

    def write_header(
        self,
        *,
        state: WorldState,
        level: str,
        agent: str,
        model: str | None,
        provider: str | None,
        npc_provider: str | None = None,
        npc_model: str | None = None,
        npc_seed: int | None = None,
        instance_id: str = "",
    ) -> None:
        cfg = state.config.to_dict()
        rec = {
            "kind": "header",
            "schema_version": 1,
            "seed": state.seed,
            "level": level,
            "config": cfg,
            "config_hash": config_hash(cfg),
            "detective_agent": agent,
            "detective_model": model,
            "detective_provider": provider,
            "agent": agent,
            "model": model,
            "provider": provider,
            "npc_provider": npc_provider,
            "npc_model": npc_model,
            "npc_seed": npc_seed,
            "git_sha": _git_sha(),
            "started_at": _now(),
            "instance_id": instance_id or f"seed_{state.seed}",
        }
        self._write(rec)

    def write_step(
        self,
        *,
        step: int,
        action: str,
        action_kwargs: dict,
        observation: str,
        model_response: str | None,
        result_observation: str,
        success: bool,
        post_state_hash: str,
        actor_id: str = "detective",
        role: str = "detective",
    ) -> None:
        self._write({
            "kind": "step",
            "step": step,
            "actor_id": actor_id,
            "role": role,
            "action": action,
            "action_kwargs": action_kwargs,
            "observation": observation,
            "model_response": model_response,
            "result_observation": result_observation,
            "success": success,
            "world_state_hash": post_state_hash,
            "timestamp": _now(),
        })

    def write_footer(
        self,
        *,
        episode_summary: dict,
        metrics: dict | None,
        elapsed_seconds: float,
        error: str | None = None,
    ) -> None:
        self._write({
            "kind": "footer",
            "episode_summary": episode_summary,
            "metrics": metrics,
            "ended_at": _now(),
            "elapsed_seconds": elapsed_seconds,
            "error": error,
        })

    def _write(self, rec: dict) -> None:
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._closed:
            self._fh.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_trajectory(path: str | Path) -> list[dict]:
    """Load a trajectory file as a list of records."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def trajectory_hash(path: str | Path) -> str:
    """Hash the sequence of (action, kwargs, world_state_hash) — independent of timestamps."""
    recs = read_trajectory(path)
    sig = []
    for r in recs:
        if r.get("kind") == "step":
            sig.append((
                r["step"],
                r.get("actor_id", "detective"),
                r["action"],
                r["action_kwargs"],
                r["world_state_hash"],
            ))
    return hashlib.sha256(_stable_json(sig).encode()).hexdigest()
