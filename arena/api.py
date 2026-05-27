"""HTTP API layer for MysteryArena results and interactive sessions."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from arena.aggregate import aggregate_matches, load_matches, write_outputs
from arena.jobs import ArenaJobManager
from arena.metrics import (
    PAYOFF_SCHEMA,
    culprit_payoff,
    detective_payoff,
    match_from_trajectory,
    read_jsonl,
)
from arena.roster import (
    REGISTRY,
    ModelSpec,
    get_model,
    make_culprit_agent,
    make_detective_agent,
)
from evaluation.metrics import compute_episode_metrics
from evaluation.trajectory import config_hash, world_state_hash
from mystery_world import COMPLEXITY_PRESETS, ComplexityLevel
from mystery_world.generator import generate_mystery
from mystery_world.narrator import (
    render_character_summary,
    render_culprit_briefing,
    render_evidence_summary,
    render_initial_briefing,
    render_step_observation,
)
from mystery_world.world import AgentAction, MysteryEnvironment


PlayerRole = Literal["detective", "culprit", "both"]
ActionRole = Literal["detective", "culprit"]


class NPCConfig(BaseModel):
    provider: Literal["fallback", "openai", "openrouter", "vllm"] = "fallback"
    model: str = "gpt-4o-mini"
    url: str | None = None
    seed: int = 42
    prompt_policy: str = "role_facts_only_no_strategy"


class SessionCreateRequest(BaseModel):
    player_role: PlayerRole = "detective"
    detective: str = "api_player"
    culprit: str = "passive"
    level: str = "TRIVIAL"
    seed: int = 0
    npc: NPCConfig = Field(default_factory=NPCConfig)


class SessionActionRequest(BaseModel):
    action: str
    action_args: dict[str, Any] = Field(default_factory=dict)
    role: ActionRole | None = None


class SessionCommitRequest(BaseModel):
    run_id: str | None = None
    match_id: str | None = None
    publish_hf: bool = False
    repo_id: str | None = None
    private: bool = False
    revision: str | None = None
    create_pr: bool = False
    include_model_responses: bool = True


class ArenaRunRequest(BaseModel):
    mode: Literal["detective", "culprit", "matrix"] = "matrix"
    detectives: str = "heuristic"
    culprits: str = "passive"
    levels: list[str] = Field(default_factory=lambda: ["TRIVIAL"])
    seeds: str = "0"
    run_id: str | None = None
    workers: int = Field(default=1, ge=1, le=128)
    schedule: Literal["balanced", "row-major"] = "balanced"
    resume: bool = True
    npc: NPCConfig = Field(default_factory=NPCConfig)
    bootstrap_samples: int = Field(default=1000, ge=1)


class ArenaMatchRequest(BaseModel):
    detective: str = "heuristic"
    culprit: str = "passive"
    level: str = "TRIVIAL"
    seed: int = 0
    run_id: str | None = None
    npc: NPCConfig = Field(default_factory=NPCConfig)
    resume: bool = True
    bootstrap_samples: int = Field(default=1000, ge=1)


class HFPublishRequest(BaseModel):
    repo_id: str | None = None
    private: bool = False
    revision: str | None = None
    create_pr: bool = False
    include_model_responses: bool = True


def load_env_file(path: str | Path | None) -> None:
    """Load simple KEY=VALUE pairs from a dotenv file without overwriting env."""
    if not path:
        return
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def configure_gateway(
    *,
    base_url: str | None = None,
    key_env: str | None = None,
    model: str | None = None,
    gateway_url_env: str = "LLM_GATEWAY_URL",
    gateway_key_env: str = "LLM_GATEWAY_API_KEY",
) -> dict[str, Any]:
    """Configure the server-side LLM gateway used by Arena model agents."""
    url = base_url or os.environ.get(gateway_url_env)
    resolved_key_env = key_env
    if not resolved_key_env and os.environ.get(gateway_key_env):
        resolved_key_env = gateway_key_env
    if url:
        BaseAgent.configure_litellm(
            url,
            api_key_env=resolved_key_env,
            model=model,
        )
    return {
        "configured": bool(url),
        "url": url,
        "key_env": resolved_key_env,
        "model_override": model,
    }


def _build_npc_responder(npc: dict[str, Any]):
    provider = npc.get("provider", "fallback")
    if provider in (None, "fallback"):
        return None
    from mystery_world.npc_responder import NPCResponder

    if provider == "openai":
        return NPCResponder(
            base_url=None,
            model=npc["model"],
            seed=int(npc["seed"]),
            api_key_env="OPENAI_API_KEY",
        )
    if provider == "openrouter":
        return NPCResponder(
            base_url="https://openrouter.ai/api/v1",
            model=npc["model"],
            seed=int(npc["seed"]),
            api_key_env="OPENROUTER_API_KEY",
        )
    if provider == "vllm":
        return NPCResponder(
            base_url=npc.get("url"),
            model=npc["model"],
            seed=int(npc["seed"]),
        )
    raise ValueError(f"unknown NPC provider: {provider}")


def _empty_outputs() -> dict[str, Any]:
    return {
        "detective_leaderboard": [],
        "culprit_leaderboard": [],
        "ratings": {"system": "trueskill", "detective": {}, "culprit": {}},
        "matrix": {},
        "summary": {"matches": 0, "detectives": 0, "culprits": 0},
    }


def _rating_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    rating = config.get("rating", {})
    return {
        "bootstrap_samples": int(rating.get("bootstrap_samples", 1000)),
        "trueskill_mu": float(rating.get("trueskill_mu", 25.0)),
        "trueskill_sigma": float(rating.get("trueskill_sigma", 25.0 / 3.0)),
        "trueskill_beta": float(rating.get("trueskill_beta", 25.0 / 6.0)),
        "trueskill_tau": float(rating.get("trueskill_tau", 25.0 / 300.0)),
        "trueskill_draw_threshold": float(rating.get("trueskill_draw_threshold", 0.0)),
    }


def _safe_child(root: Path, name: str) -> Path:
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=404, detail="run not found")
    candidate = (root / name).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise HTTPException(status_code=400, detail="invalid path")
    return candidate


def _trajectory_path_for_match(run_dir: Path, match: dict[str, Any]) -> Path | None:
    raw = match.get("trajectory_path")
    if not raw:
        return None
    candidates = [Path(raw)]
    if not Path(raw).is_absolute():
        candidates.append(run_dir / raw)
    run_root = run_dir.resolve()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists() and (resolved == run_root or run_root in resolved.parents):
            return resolved
    return None


def _public_match(run_dir: Path, match: dict[str, Any], idx: int) -> dict[str, Any]:
    public = dict(match)
    episode_id = str(match.get("match_id") or f"episode_{idx}")
    public["episode_id"] = episode_id
    public["trajectory_available"] = _trajectory_path_for_match(run_dir, match) is not None
    public.pop("trajectory_path", None)
    return public


class ArenaResultsStore:
    """Read-only API facade over Arena result directories."""

    def __init__(self, arena_root: str | Path) -> None:
        self.arena_root = Path(arena_root)

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.arena_root.exists():
            return []
        runs = []
        for path in sorted(self.arena_root.iterdir()):
            if not path.is_dir():
                continue
            config_path = path / "config.json"
            matches_path = path / "matches.jsonl"
            config = {}
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    config = {}
            runs.append({
                "run_id": path.name,
                "path": path.name,
                "has_config": config_path.exists(),
                "has_matches": matches_path.exists(),
                "mode": config.get("mode"),
                "levels": config.get("levels", []),
                "seeds": config.get("seeds", []),
            })
        return runs

    def load_run(self, run_id: str) -> dict[str, Any]:
        run_dir = _safe_child(self.arena_root, run_id)
        if not run_dir.exists() or not run_dir.is_dir():
            raise HTTPException(status_code=404, detail="run not found")
        config_path = run_dir / "config.json"
        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        matches = load_matches(run_dir) if (run_dir / "matches.jsonl").exists() else []
        outputs = aggregate_matches(matches, **_rating_kwargs(config)) if matches else _empty_outputs()
        return {
            "run_id": run_id,
            "config": config,
            "matches": [
                _public_match(run_dir, match, idx)
                for idx, match in enumerate(matches)
            ],
            "outputs": outputs,
        }

    def load_episodes(self, run_id: str) -> list[dict[str, Any]]:
        run = self.load_run(run_id)
        episodes = []
        for match in run["matches"]:
            label = (
                f"{match.get('detective', {}).get('name')} vs "
                f"{match.get('culprit', {}).get('name')} | "
                f"{match.get('level')} seed={match.get('seed')} | "
                f"payoff={float(match.get('detective_payoff', 0.0)):.3f}"
            )
            episodes.append({
                "episode_id": match["episode_id"],
                "label": label,
                "trajectory_available": match.get("trajectory_available", False),
                "match": match,
            })
        return episodes

    def load_trajectory(self, run_id: str, episode_id: str) -> list[dict[str, Any]]:
        run_dir = _safe_child(self.arena_root, run_id)
        matches = load_matches(run_dir) if (run_dir / "matches.jsonl").exists() else []
        for idx, match in enumerate(matches):
            current_id = str(match.get("match_id") or f"episode_{idx}")
            if current_id != episode_id:
                continue
            path = _trajectory_path_for_match(run_dir, match)
            if path is None:
                raise HTTPException(status_code=404, detail="trajectory not found")
            return read_jsonl(path)
        raise HTTPException(status_code=404, detail="episode not found")


def _api_player_spec(name: str, role: ActionRole) -> ModelSpec:
    safe = name.strip() or "api_player"
    return ModelSpec(name=safe, roles=(role,), kind="api")


def _level_name(raw: str) -> str:
    name = raw.strip().upper()
    try:
        ComplexityLevel[name]
    except KeyError as exc:
        known = ", ".join(level.name for level in ComplexityLevel)
        raise HTTPException(status_code=400, detail=f"unknown level {raw!r}; use one of {known}") from exc
    return name


def _agent_action(raw: str) -> AgentAction:
    try:
        return AgentAction[raw.strip().upper()]
    except KeyError as exc:
        known = ", ".join(action.name for action in AgentAction)
        raise HTTPException(status_code=400, detail=f"unknown action {raw!r}; use one of {known}") from exc


def _default_run_id(prefix: str = "arena_api") -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def _clean_run_id(raw: str | None, *, prefix: str = "arena_api") -> str:
    run_id = (raw or "").strip() or _default_run_id(prefix)
    if run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise HTTPException(status_code=400, detail="run_id must be a single path segment")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise HTTPException(
            status_code=400,
            detail="run_id may contain only letters, numbers, underscore, dot, and dash",
        )
    return run_id


@dataclass
class InteractiveArenaSession:
    session_id: str
    player_role: PlayerRole
    level: str
    seed: int
    detective: ModelSpec
    culprit: ModelSpec
    npc: dict[str, Any]
    env: MysteryEnvironment
    briefing: str
    player_observation: str
    detective_agent: Any = None
    culprit_agent: Any = None
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    events: list[dict[str, Any]] = field(default_factory=list)
    belief_trace: list[dict[str, Any]] = field(default_factory=list)
    detective_steps: int = 0
    done: bool = False
    error: str | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    def _role_observations(self) -> dict[str, str]:
        observations = {"detective": self.env.observe_location()}
        if self.env.state.culprit_id:
            observations["culprit"] = self.env.observe_location(self.env.state.culprit_id)
        return observations

    def snapshot(self, *, new_events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        result = self.final_result() if self.done else None
        observations = self._role_observations() if self.player_role == "both" else {}
        return {
            "session_id": self.session_id,
            "player_role": self.player_role,
            "level": self.level,
            "seed": self.seed,
            "detective": self.detective.to_dict(),
            "culprit": self.culprit.to_dict(),
            "npc": self.npc,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "done": self.done,
            "error": self.error,
            "budget_remaining": self.env.budget_remaining,
            "culprit_budget_remaining": self.env.culprit_budget_remaining,
            "briefing": self.briefing,
            "observation": self.player_observation,
            "observations": observations,
            "events": list(self.events),
            "new_events": list(new_events or []),
            "result": result,
        }

    def final_result(self) -> dict[str, Any]:
        summary = self.env.get_episode_summary()
        metrics_dict = None
        if self.belief_trace:
            culprit = self.env.state.get_culprit()
            weapon = self.env.state.objects.get(self.env.state.murder_weapon_id)
            murder_loc = self.env.state.locations.get(self.env.state.murder_location_id)
            metrics = compute_episode_metrics(
                episode_summary=summary,
                belief_snapshots=self.belief_trace,
                ground_truth={
                    "culprit_name": culprit.full_name if culprit else "",
                    "weapon_name": weapon.name if weapon else "",
                    "location_name": murder_loc.name if murder_loc else "",
                },
                total_tokens=getattr(self.detective_agent, "total_tokens_used", 0),
                complexity_level=ComplexityLevel[self.level].value,
            )
            metrics_dict = metrics.to_dict()
        d_payoff = 0.0 if self.error else detective_payoff(summary, metrics_dict)
        c_payoff = 0.0 if self.error else culprit_payoff(summary, metrics_dict)
        return {
            "summary": summary,
            "metrics": metrics_dict,
            "payoff_schema": PAYOFF_SCHEMA,
            "detective_payoff": round(d_payoff, 6),
            "culprit_payoff": round(c_payoff, 6),
            "solved": bool(summary.get("accusation_correct", False)),
        }

    def step(
        self,
        action: AgentAction,
        action_args: dict[str, Any],
        *,
        role: ActionRole | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            if self.done:
                raise HTTPException(status_code=409, detail="session is already complete")
            if self.player_role == "both":
                if role not in {"detective", "culprit"}:
                    raise HTTPException(
                        status_code=400,
                        detail="role is required for player_role='both'",
                    )
                new_events = (
                    self._step_player_detective(action, action_args)
                    if role == "detective"
                    else self._step_player_culprit(action, action_args)
                )
            elif role is not None and role != self.player_role:
                raise HTTPException(
                    status_code=400,
                    detail=f"role must be {self.player_role!r} for this session",
                )
            elif self.player_role == "detective":
                new_events = self._step_player_detective(action, action_args)
            else:
                new_events = self._step_player_culprit(action, action_args)
            self._refresh_player_observation()
            self.updated_at = dt.datetime.now(dt.timezone.utc).isoformat()
            return self.snapshot(new_events=new_events)

    def _append_event(
        self,
        *,
        actor_id: str,
        role: str,
        action: AgentAction,
        action_args: dict[str, Any],
        observation: str,
        result_observation: str,
        success: bool,
        model_response: str | None = None,
    ) -> dict[str, Any]:
        event = {
            "index": len(self.events),
            "step": self.env.state.current_step,
            "actor_id": actor_id,
            "role": role,
            "action": action.name,
            "action_args": dict(action_args),
            "observation": observation,
            "result_observation": result_observation,
            "success": success,
            "model_response": model_response,
            "world_state_hash": world_state_hash(self.env.state),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        self.events.append(event)
        return event

    def _step_player_detective(
        self,
        action: AgentAction,
        action_args: dict[str, Any],
    ) -> list[dict[str, Any]]:
        events = []
        input_obs = self.env.observe_location()
        result = self.env.step(action, **action_args)
        rendered = render_step_observation(self.env, result.observation)
        events.append(self._append_event(
            actor_id="detective",
            role="detective",
            action=action,
            action_args=action_args,
            observation=input_obs,
            result_observation=rendered,
            success=result.success,
        ))
        self.detective_steps += 1
        if self.env.is_solved:
            self.done = True
            return events
        if self.culprit_agent is not None:
            events.extend(self._run_model_culprit_turn())
        return events

    def _step_player_culprit(
        self,
        action: AgentAction,
        action_args: dict[str, Any],
    ) -> list[dict[str, Any]]:
        events = []
        actor_id = self.env.state.culprit_id
        input_obs = self.env.observe_location(actor_id)
        result = self.env.step_for_actor(actor_id, action, **action_args)
        rendered = render_step_observation(self.env, result.observation, actor_id=actor_id)
        events.append(self._append_event(
            actor_id=actor_id,
            role="culprit",
            action=action,
            action_args=action_args,
            observation=input_obs,
            result_observation=rendered,
            success=result.success,
        ))
        if not self.env.is_solved and self.detective_agent is not None:
            events.extend(self._run_model_detective_turn())
        return events

    def _run_model_culprit_turn(self) -> list[dict[str, Any]]:
        actor_id = self.env.state.culprit_id
        obs = self.env.observe_location(actor_id)
        action, kwargs = self.culprit_agent.decide_action(obs)
        result = self.env.step_for_actor(actor_id, action, **kwargs)
        rendered = render_step_observation(self.env, result.observation, actor_id=actor_id)
        self.culprit_agent.record_action(action, kwargs, rendered)
        try:
            self.culprit_agent.update_beliefs(rendered)
        except NotImplementedError:
            pass
        return [self._append_event(
            actor_id=actor_id,
            role="culprit",
            action=action,
            action_args=kwargs,
            observation=obs,
            result_observation=rendered,
            success=result.success,
            model_response=getattr(self.culprit_agent, "last_raw_response", None),
        )]

    def _run_model_detective_turn(self) -> list[dict[str, Any]]:
        if self.env.budget_remaining <= 0:
            action = AgentAction.ACCUSE
            belief = self.detective_agent.belief_state
            kwargs = {
                "suspect_name": belief.top_suspect() or "",
                "weapon_name": belief.top_weapon() or "",
                "location_name": belief.top_location() or "",
            }
            obs_context = "[budget exhausted; forcing accusation]"
        else:
            obs_context = self.env.observe_location()
            if self.detective_steps % 3 == 0:
                obs_context += "\n" + render_evidence_summary(self.env)
                obs_context += "\n" + render_character_summary(self.env)
            action, kwargs = self.detective_agent.decide_action(obs_context)

        result = self.env.step(action, **kwargs)
        rendered = render_step_observation(self.env, result.observation)
        self.detective_agent.record_action(action, kwargs, rendered)
        self.detective_agent.update_beliefs(rendered)
        self.detective_steps += 1
        self.belief_trace.append(self.detective_agent.get_belief_snapshot())
        if self.env.is_solved:
            self.done = True
        return [self._append_event(
            actor_id="detective",
            role="detective",
            action=action,
            action_args=kwargs,
            observation=obs_context,
            result_observation=rendered,
            success=result.success,
            model_response=getattr(self.detective_agent, "last_raw_response", None),
        )]

    def _refresh_player_observation(self) -> None:
        if self.player_role == "detective":
            self.player_observation = self.env.observe_location()
        elif self.player_role == "culprit":
            self.player_observation = self.env.observe_location(self.env.state.culprit_id)
        else:
            self.player_observation = self.env.observe_location()


class ArenaSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, InteractiveArenaSession] = {}
        self._lock = threading.RLock()

    def create(self, request: SessionCreateRequest) -> InteractiveArenaSession:
        level = _level_name(request.level)
        npc = request.npc.model_dump()
        state = generate_mystery(
            config=COMPLEXITY_PRESETS[ComplexityLevel[level]],
            seed=request.seed,
        )

        if request.player_role == "detective":
            detective = _api_player_spec(request.detective, "detective")
            culprit = get_model(request.culprit, role="culprit")
        elif request.player_role == "culprit":
            detective = get_model(request.detective, role="detective")
            culprit = _api_player_spec(request.culprit, "culprit")
            state.config = replace(state.config, free_culprit_actions=True)
        else:
            detective = _api_player_spec(request.detective, "detective")
            culprit = _api_player_spec(request.culprit, "culprit")
            state.config = replace(state.config, free_culprit_actions=True)

        if request.player_role == "detective" and culprit.kind != "passive":
            state.config = replace(state.config, free_culprit_actions=True)

        env = MysteryEnvironment(state)
        npc_responder = _build_npc_responder(npc)
        if npc_responder is not None:
            env.set_npc_responder(npc_responder)
        if state.config.free_culprit_actions:
            env.enable_free_culprit()

        detective_agent = None
        culprit_agent = None
        belief_trace: list[dict[str, Any]] = []

        if request.player_role == "culprit":
            detective_agent = make_detective_agent(detective)
            detective_agent.initialize(env, render_initial_briefing(env))
            belief_trace.append(detective_agent.get_belief_snapshot())
            briefing = render_culprit_briefing(env)
            player_observation = env.observe_location(env.state.culprit_id)
        elif request.player_role == "both":
            briefing = (
                "Detective briefing\n"
                "-------------------\n"
                f"{render_initial_briefing(env)}\n\n"
                "Culprit briefing\n"
                "-----------------\n"
                f"{render_culprit_briefing(env)}"
            )
            player_observation = env.observe_location()
        else:
            if culprit.kind != "passive":
                culprit_agent = make_culprit_agent(culprit)
                if culprit_agent is not None:
                    culprit_agent.initialize(env, render_culprit_briefing(env))
            briefing = render_initial_briefing(env)
            player_observation = env.observe_location()

        session = InteractiveArenaSession(
            session_id=uuid.uuid4().hex,
            player_role=request.player_role,
            level=level,
            seed=request.seed,
            detective=detective,
            culprit=culprit,
            npc=npc,
            env=env,
            briefing=briefing,
            player_observation=player_observation,
            detective_agent=detective_agent,
            culprit_agent=culprit_agent,
            belief_trace=belief_trace,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> InteractiveArenaSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [
            {
                "session_id": session.session_id,
                "player_role": session.player_role,
                "level": session.level,
                "seed": session.seed,
                "detective": session.detective.to_dict(),
                "culprit": session.culprit.to_dict(),
                "done": session.done,
                "updated_at": session.updated_at,
            }
            for session in sessions
        ]


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "episode"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, default=str) + "\n" for record in records),
        encoding="utf-8",
    )


def _session_elapsed_seconds(session: InteractiveArenaSession) -> float:
    try:
        start = dt.datetime.fromisoformat(session.created_at)
        end = dt.datetime.fromisoformat(session.updated_at)
        return max(0.0, (end - start).total_seconds())
    except ValueError:
        return 0.0


def _session_trajectory_records(
    session: InteractiveArenaSession,
    *,
    run_id: str,
    match_id: str,
) -> list[dict[str, Any]]:
    cfg = session.env.state.config.to_dict()
    final = session.final_result()
    header = {
        "kind": "header",
        "schema_version": 1,
        "seed": session.seed,
        "level": session.level,
        "config": cfg,
        "config_hash": config_hash(cfg),
        "detective_agent": session.detective.name,
        "detective_model": session.detective.model,
        "detective_provider": session.detective.provider,
        "agent": session.detective.name,
        "model": session.detective.model,
        "provider": session.detective.provider,
        "npc_provider": session.npc.get("provider", "fallback"),
        "npc_model": session.npc.get("model") if session.npc.get("provider") != "fallback" else None,
        "npc_seed": session.npc.get("seed") if session.npc.get("provider") != "fallback" else None,
        "culprit_agent": session.culprit.name,
        "culprit_model": session.culprit.model,
        "culprit_provider": session.culprit.provider,
        "arena_run_id": run_id,
        "arena_match_id": match_id,
        "git_sha": "api-session",
        "started_at": session.created_at,
        "instance_id": f"session_{session.session_id}",
    }
    steps = []
    for event in session.events:
        steps.append({
            "kind": "step",
            "step": event.get("step", event.get("index", 0)),
            "actor_id": event.get("actor_id", event.get("role", "detective")),
            "role": event.get("role", "detective"),
            "action": event.get("action"),
            "action_kwargs": event.get("action_args", {}),
            "observation": event.get("observation", ""),
            "model_response": event.get("model_response"),
            "result_observation": event.get("result_observation", ""),
            "success": bool(event.get("success", True)),
            "world_state_hash": event.get("world_state_hash", ""),
            "timestamp": event.get("timestamp", session.updated_at),
        })
    footer = {
        "kind": "footer",
        "episode_summary": final.get("summary") or {},
        "metrics": final.get("metrics"),
        "ended_at": session.updated_at,
        "elapsed_seconds": _session_elapsed_seconds(session),
        "error": session.error,
    }
    return [header, *steps, footer]


def _commit_session_to_run(
    session: InteractiveArenaSession,
    *,
    arena_root: Path,
    run_id: str,
    match_id: str,
) -> dict[str, Any]:
    if not session.done:
        raise HTTPException(status_code=409, detail="session must be complete before commit")

    run_dir = arena_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    safe_match = _safe_filename(match_id)
    trajectory_path = run_dir / "trajectories" / f"{safe_match}.jsonl"
    records = _session_trajectory_records(session, run_id=run_id, match_id=match_id)
    _write_jsonl(trajectory_path, records)

    config = {
        "run_id": run_id,
        "mode": "interactive_session",
        "detectives": [session.detective.name],
        "culprits": [session.culprit.name],
        "levels": [session.level],
        "seeds": [session.seed],
        "npc": session.npc,
        "source": {
            "type": "api_session",
            "session_id": session.session_id,
            "player_role": session.player_role,
        },
        "rating": {
            "bootstrap_samples": 1000,
            "trueskill_mu": 25.0,
            "trueskill_sigma": 25.0 / 3.0,
            "trueskill_beta": 25.0 / 6.0,
            "trueskill_tau": 25.0 / 300.0,
            "trueskill_draw_threshold": 0.0,
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (run_dir / "roster.json").write_text(
        json.dumps(
            {
                "detectives": [session.detective.to_dict()],
                "culprits": [session.culprit.to_dict()],
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    match = match_from_trajectory(
        trajectory_path,
        run_id=run_id,
        match_id=match_id,
        detective=session.detective,
        culprit=session.culprit,
        npc=session.npc,
    )
    matches = [
        existing
        for existing in load_matches(run_dir)
        if str(existing.get("match_id")) != match_id
    ]
    matches.append(match)
    matches.sort(
        key=lambda item: (
            str(item.get("detective", {}).get("name", "")),
            str(item.get("culprit", {}).get("name", "")),
            str(item.get("level", "")),
            int(item.get("seed", 0) or 0),
            str(item.get("match_id", "")),
        )
    )
    with (run_dir / "matches.jsonl").open("w", encoding="utf-8") as fh:
        for item in matches:
            fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    outputs = write_outputs(run_dir)
    return {
        "run_id": run_id,
        "match_id": match_id,
        "run_dir": str(run_dir),
        "trajectory_path": str(trajectory_path),
        "match": match,
        "outputs": outputs,
    }


def create_app(
    *,
    arena_root: str | Path = "arena/results",
    env_file: str | Path | None = ".env",
    gateway_url: str | None = None,
    gateway_key_env: str | None = None,
    gateway_model: str | None = None,
    gateway_url_env: str = "LLM_GATEWAY_URL",
    gateway_key_env_default: str = "LLM_GATEWAY_API_KEY",
) -> FastAPI:
    """Build the Arena backend API app."""
    load_env_file(env_file)
    gateway = configure_gateway(
        base_url=gateway_url,
        key_env=gateway_key_env,
        model=gateway_model,
        gateway_url_env=gateway_url_env,
        gateway_key_env=gateway_key_env_default,
    )
    results = ArenaResultsStore(arena_root)
    sessions = ArenaSessionManager()
    jobs = ArenaJobManager(arena_root=results.arena_root, env_file=env_file)

    app = FastAPI(title="MysteryArena API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "arena_root": str(results.arena_root),
            "gateway": gateway,
        }

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        return {
            "models": [spec.to_dict() for spec in REGISTRY.values()],
            "actions": [action.name for action in AgentAction],
        }

    @app.get("/api/runs")
    def runs() -> dict[str, Any]:
        return {"runs": results.list_runs()}

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict[str, Any]:
        return results.load_run(run_id)

    @app.get("/api/runs/{run_id}/episodes")
    def episodes(run_id: str) -> dict[str, Any]:
        return {"episodes": results.load_episodes(run_id)}

    @app.get("/api/runs/{run_id}/episodes/{episode_id}/trajectory")
    def trajectory(run_id: str, episode_id: str) -> dict[str, Any]:
        return {"records": results.load_trajectory(run_id, episode_id)}

    @app.get("/api/arena/jobs")
    def arena_jobs() -> dict[str, Any]:
        return {"jobs": jobs.list_jobs()}

    @app.get("/api/arena/jobs/{job_id}")
    def arena_job(job_id: str) -> dict[str, Any]:
        try:
            return jobs.get(job_id).snapshot()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.post("/api/arena/jobs/{job_id}/cancel")
    def cancel_arena_job(job_id: str) -> dict[str, Any]:
        try:
            return jobs.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.post("/api/arena/runs")
    def create_arena_run(request: ArenaRunRequest) -> dict[str, Any]:
        run_id = _clean_run_id(request.run_id)
        levels = [_level_name(level) for level in request.levels]
        job = jobs.start_run(
            mode=request.mode,
            detectives=request.detectives,
            culprits=request.culprits,
            levels=levels,
            seeds=request.seeds,
            run_id=run_id,
            workers=request.workers,
            schedule=request.schedule,
            resume=request.resume,
            npc=request.npc.model_dump(),
            bootstrap_samples=request.bootstrap_samples,
        )
        return job.snapshot()

    @app.post("/api/arena/matches")
    def create_arena_match(request: ArenaMatchRequest) -> dict[str, Any]:
        level = _level_name(request.level)
        run_id = _clean_run_id(request.run_id, prefix="arena_match")
        job = jobs.start_run(
            mode="matrix",
            detectives=request.detective,
            culprits=request.culprit,
            levels=[level],
            seeds=str(request.seed),
            run_id=run_id,
            workers=1,
            schedule="balanced",
            resume=request.resume,
            npc=request.npc.model_dump(),
            bootstrap_samples=request.bootstrap_samples,
        )
        return job.snapshot()

    @app.post("/api/arena/runs/{run_id}/publish-hf")
    def publish_arena_run(run_id: str, request: HFPublishRequest) -> dict[str, Any]:
        clean_run_id = _clean_run_id(run_id)
        repo_id = request.repo_id or os.environ.get("ARENA_HF_DATASET")
        if not repo_id:
            raise HTTPException(
                status_code=400,
                detail="repo_id is required; set ARENA_HF_DATASET or pass repo_id",
            )
        run_dir = _safe_child(results.arena_root, clean_run_id)
        if not run_dir.exists() or not run_dir.is_dir():
            raise HTTPException(status_code=404, detail="run not found")
        job = jobs.start_publish(
            run_id=clean_run_id,
            repo_id=repo_id,
            private=request.private,
            revision=request.revision,
            create_pr=request.create_pr,
            include_model_responses=request.include_model_responses,
        )
        return job.snapshot()

    @app.get("/api/sessions")
    def list_sessions() -> dict[str, Any]:
        return {"sessions": sessions.list_sessions()}

    @app.post("/api/sessions")
    def create_session(request: SessionCreateRequest) -> dict[str, Any]:
        return sessions.create(request).snapshot()

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        return sessions.get(session_id).snapshot()

    @app.post("/api/sessions/{session_id}/actions")
    def step_session(session_id: str, request: SessionActionRequest) -> dict[str, Any]:
        session = sessions.get(session_id)
        return session.step(
            _agent_action(request.action),
            request.action_args,
            role=request.role,
        )

    @app.post("/api/sessions/{session_id}/commit")
    def commit_session(session_id: str, request: SessionCommitRequest) -> dict[str, Any]:
        session = sessions.get(session_id)
        run_id = _clean_run_id(request.run_id, prefix="arena_session")
        match_id = request.match_id or (
            f"{session.detective.name}__vs__{session.culprit.name}__"
            f"{session.level}__seed_{session.seed}__session_{session.session_id[:8]}"
        )
        result = _commit_session_to_run(
            session,
            arena_root=results.arena_root,
            run_id=run_id,
            match_id=match_id,
        )
        if request.publish_hf:
            repo_id = request.repo_id or os.environ.get("ARENA_HF_DATASET")
            if not repo_id:
                raise HTTPException(
                    status_code=400,
                    detail="repo_id is required; set ARENA_HF_DATASET or pass repo_id",
                )
            job = jobs.start_publish(
                run_id=run_id,
                repo_id=repo_id,
                private=request.private,
                revision=request.revision,
                create_pr=request.create_pr,
                include_model_responses=request.include_model_responses,
            )
            result["publish_job"] = job.snapshot()
        return result

    return app


__all__ = [
    "ArenaMatchRequest",
    "ArenaResultsStore",
    "ArenaRunRequest",
    "ArenaSessionManager",
    "HFPublishRequest",
    "NPCConfig",
    "SessionActionRequest",
    "SessionCommitRequest",
    "SessionCreateRequest",
    "configure_gateway",
    "create_app",
    "load_env_file",
]
