"""Run MysteryArena detective/culprit Arena matches.

Examples:
    uv run python scripts/arena_run.py --mode detective \
        --detectives heuristic,oracle_min --culprits passive \
        --levels TRIVIAL EASY --seeds 0-2 --out arena/results/smoke

    uv run python scripts/arena_run.py --mode matrix \
        --detectives claude,chatgpt --culprits claude,chatgpt \
        --npc-provider fallback --levels TRIVIAL,EASY --seeds 0-9 \
        --out arena/results/run_001 --skip-existing
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from rich import box
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.markup import escape as rich_escape
    from rich.table import Table
except Exception:  # noqa: BLE001 - Rich is an optional UI dependency at runtime.
    box = None
    Console = None
    Group = None
    Live = None
    Panel = None
    Progress = None
    SpinnerColumn = None
    BarColumn = None
    TextColumn = None
    TimeElapsedColumn = None
    TimeRemainingColumn = None
    Table = None
    rich_escape = None

from agents.base_agent import BaseAgent
from arena.aggregate import write_outputs
from arena.metrics import match_from_episode, match_from_trajectory
from arena.roster import ModelSpec, make_culprit_agent, make_detective_agent, parse_model_list
from arena.trueskill import compute_role_trueskill
from evaluation.runner import run_episode
from evaluation.trajectory import TrajectoryWriter
from mystery_world import COMPLEXITY_PRESETS, ComplexityLevel
from mystery_world.generator import generate_mystery

ArenaJob = tuple[ModelSpec, ModelSpec, str, int]


def _parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    return out


def _parse_levels(raw: list[str]) -> list[str]:
    parts: list[str] = []
    for item in raw:
        parts.extend(p.strip() for p in item.split(",") if p.strip())
    return [p.upper() for p in parts]


def _load_env_file(path: str | Path | None) -> None:
    """Load simple KEY=VALUE pairs from .env without overwriting the shell."""
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
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        os.environ[key] = value


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


def _match_id(detective: ModelSpec, culprit: ModelSpec, level: str, seed: int) -> str:
    return f"{detective.name}__vs__{culprit.name}__{level}__seed_{seed}"


def _trajectory_path(
    out_dir: Path,
    detective: ModelSpec,
    culprit: ModelSpec,
    level: str,
    seed: int,
) -> Path:
    return (
        out_dir
        / "trajectories"
        / detective.name
        / culprit.name
        / level
        / f"seed_{seed}.jsonl"
    )


def _episode_detective_step_budget(level: str) -> int:
    return COMPLEXITY_PRESETS[ComplexityLevel[level]].max_agent_actions


def _episode_actor_step_budget(level: str, culprit: ModelSpec) -> int:
    max_detective_steps = _episode_detective_step_budget(level)
    actor_multiplier = 1 if culprit.kind == "passive" else 2
    return max_detective_steps * actor_multiplier


def _trajectory_footer(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        for raw in reversed(path.read_text(encoding="utf-8").splitlines()):
            if not raw.strip():
                continue
            record = json.loads(raw)
            return record if record.get("kind") == "footer" else None
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _trajectory_resume_rejection(path: Path) -> str:
    footer = _trajectory_footer(path)
    if footer is None:
        return "incomplete"
    error = str(footer.get("error") or "").strip()
    if error:
        return f"error: {error.splitlines()[0]}"
    return ""


def _trajectory_is_complete(path: Path) -> bool:
    return path.exists() and not _trajectory_resume_rejection(path)


def _match_actor_steps(match: dict[str, Any]) -> int:
    return int(match.get("actions_taken", 0) or 0) + int(
        match.get("culprit_actions_taken", 0) or 0
    )


def _load_resumable_jobs(
    *,
    jobs: list[ArenaJob],
    out_dir: Path,
    run_id: str,
    npc: dict[str, Any],
) -> tuple[list[tuple[dict[str, Any], int]], list[ArenaJob], list[tuple[Path, str]]]:
    resumed: list[tuple[dict[str, Any], int]] = []
    remaining: list[ArenaJob] = []
    ignored: list[tuple[Path, str]] = []
    for detective, culprit, level, seed in jobs:
        traj_path = _trajectory_path(out_dir, detective, culprit, level, seed)
        if not traj_path.exists():
            remaining.append((detective, culprit, level, seed))
            continue
        resume_rejection = _trajectory_resume_rejection(traj_path)
        if resume_rejection:
            ignored.append((traj_path, resume_rejection))
            remaining.append((detective, culprit, level, seed))
            continue
        try:
            match = match_from_trajectory(
                traj_path,
                run_id=run_id,
                match_id=_match_id(detective, culprit, level, seed),
                detective=detective,
                culprit=culprit,
                npc=npc,
            )
        except Exception as exc:  # noqa: BLE001 - corrupted resume data should not abort the run.
            ignored.append((traj_path, f"{type(exc).__name__}: {exc}"))
            remaining.append((detective, culprit, level, seed))
            continue
        resumed.append((match, _episode_actor_step_budget(level, culprit)))
    return resumed, remaining, ignored


def _run_one(
    *,
    run_id: str,
    out_dir: Path,
    detective: ModelSpec,
    culprit: ModelSpec,
    npc: dict[str, Any],
    level: str,
    seed: int,
    skip_existing: bool,
    progress: ArenaProgress | None = None,
) -> dict[str, Any]:
    match_id = _match_id(detective, culprit, level, seed)
    traj_path = _trajectory_path(out_dir, detective, culprit, level, seed)
    if skip_existing and _trajectory_is_complete(traj_path):
        if progress is not None:
            progress.skip_match(step_budget=_episode_actor_step_budget(level, culprit))
        return match_from_trajectory(
            traj_path,
            run_id=run_id,
            match_id=match_id,
            detective=detective,
            culprit=culprit,
            npc=npc,
        )

    traj_path.parent.mkdir(parents=True, exist_ok=True)
    state = generate_mystery(
        config=COMPLEXITY_PRESETS[ComplexityLevel[level]],
        seed=seed,
    )
    culprit_agent = make_culprit_agent(culprit)
    if culprit_agent is not None:
        state.config = replace(state.config, free_culprit_actions=True)
    detective_agent = make_detective_agent(detective)
    npc_responder = _build_npc_responder(npc)
    if progress is not None:
        progress.start_match(
            match_id=match_id,
            detective=detective.name,
            culprit=culprit.name,
            level=level,
            seed=seed,
            detective_step_budget=_episode_detective_step_budget(level),
            step_budget=_episode_actor_step_budget(level, culprit),
        )

    def on_step(payload: dict[str, Any]) -> None:
        if progress is not None:
            progress.record_step(match_id, payload)

    try:
        with TrajectoryWriter(traj_path) as writer:
            writer.write_header(
                state=state,
                level=level,
                agent=detective.name,
                model=detective.model,
                provider=detective.provider,
                npc_provider=npc.get("provider", "fallback"),
                npc_model=npc.get("model") if npc.get("provider") != "fallback" else None,
                npc_seed=npc.get("seed") if npc.get("provider") != "fallback" else None,
                culprit_agent=culprit.name,
                culprit_model=culprit.model,
                culprit_provider=culprit.provider,
                arena_run_id=run_id,
                arena_match_id=match_id,
                instance_id=f"seed_{seed}",
            )
            try:
                result = run_episode(
                    detective_agent=detective_agent,
                    world_state=state,
                    complexity_level=ComplexityLevel[level].value,
                    npc_responder=npc_responder,
                    culprit_agent=culprit_agent,
                    trajectory_writer=writer,
                    step_callback=on_step,
                )
                writer.write_footer(
                    episode_summary=result.episode_summary,
                    metrics=result.metrics.to_dict() if result.metrics else None,
                    elapsed_seconds=result.elapsed_seconds,
                    error=result.error,
                )
                return match_from_episode(
                    run_id=run_id,
                    match_id=match_id,
                    level=level,
                    seed=seed,
                    detective=detective,
                    culprit=culprit,
                    npc=npc,
                    result=result,
                    trajectory_path=traj_path,
                )
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"
                writer.write_footer(
                    episode_summary={},
                    metrics=None,
                    elapsed_seconds=0.0,
                    error=error,
                )
                return match_from_trajectory(
                    traj_path,
                    run_id=run_id,
                    match_id=match_id,
                    detective=detective,
                    culprit=culprit,
                    npc=npc,
                )
    finally:
        if progress is not None:
            progress.finish_match(match_id)


def _build_jobs(
    mode: str,
    detectives: list[ModelSpec],
    culprits: list[ModelSpec],
    levels: list[str],
    seeds: list[int],
    schedule: str = "balanced",
) -> list[ArenaJob]:
    if mode == "detective":
        active_detectives = detectives
        active_culprits = culprits or []
    elif mode == "culprit":
        active_detectives = detectives
        active_culprits = culprits
        if all(c.kind == "passive" for c in active_culprits):
            raise ValueError("--mode culprit requires at least one non-passive culprit")
    elif mode == "matrix":
        active_detectives = detectives
        active_culprits = culprits
    else:
        raise ValueError(f"unknown mode: {mode}")

    if schedule == "row-major":
        jobs = []
        for detective in active_detectives:
            for culprit in active_culprits:
                for level in levels:
                    for seed in seeds:
                        jobs.append((detective, culprit, level, seed))
        return jobs
    if schedule != "balanced":
        raise ValueError(f"unknown schedule: {schedule}")

    jobs = []
    for level in levels:
        for seed in seeds:
            for offset in range(len(active_culprits)):
                for d_idx, detective in enumerate(active_detectives):
                    culprit = active_culprits[(d_idx + offset) % len(active_culprits)]
                    jobs.append((detective, culprit, level, seed))
    return jobs


class ArenaProgress:
    """Live terminal dashboard for Arena runs, with a plain fallback."""

    def __init__(
        self,
        *,
        total: int,
        run_id: str,
        mode: str,
        out_dir: Path,
        enabled: bool,
        tail: int = 8,
        workers: int = 1,
        schedule: str = "balanced",
        levels: list[str] | None = None,
        seeds: list[int] | None = None,
        gateway_url: str | None = None,
        npc_provider: str = "fallback",
        total_step_budget: int = 0,
        resumed: int = 0,
        initial_matches: list[dict[str, Any]] | None = None,
        initial_step_accounted: int = 0,
        initial_step_observed: int = 0,
        initial_step_skipped: int = 0,
        trueskill_mu: float = 25.0,
        trueskill_sigma: float = 25.0 / 3.0,
        trueskill_beta: float = 25.0 / 6.0,
        trueskill_tau: float = 25.0 / 300.0,
        trueskill_draw_threshold: float = 0.0,
    ) -> None:
        self.total = total
        self.run_id = run_id
        self.mode = mode
        self.out_dir = out_dir
        self.enabled = enabled
        self.tail = tail
        self.workers = workers
        self.schedule = schedule
        self.levels = levels or []
        self.seeds = seeds or []
        self.gateway_url = gateway_url
        self.npc_provider = npc_provider
        self.total_step_budget = total_step_budget
        self.resumed = resumed
        self.trueskill_mu = trueskill_mu
        self.trueskill_sigma = trueskill_sigma
        self.trueskill_beta = trueskill_beta
        self.trueskill_tau = trueskill_tau
        self.trueskill_draw_threshold = trueskill_draw_threshold
        self.started = time.monotonic()
        self.matches: list[dict[str, Any]] = list(initial_matches or [])
        self.active_matches: dict[str, dict[str, Any]] = {}
        self._step_observed = max(0, initial_step_observed)
        self._step_accounted = max(0, initial_step_accounted)
        self._step_skipped = max(0, initial_step_skipped)
        self._last_render = 0.0
        self._rich_enabled = False
        self._console: Any = None
        self._live: Any = None
        self._progress: Any = None
        self._episode_task_id: Any = None
        self._step_task_id: Any = None
        self._closed = False
        self._lock = threading.RLock()
        rich_ready = all(
            obj is not None
            for obj in (
                Console,
                Group,
                Live,
                Panel,
                Progress,
                SpinnerColumn,
                BarColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
                Table,
            )
        )
        if self.enabled and rich_ready:
            self._rich_enabled = True
            self._console = Console()
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}[/bold cyan]"),
                BarColumn(
                    bar_width=None,
                    complete_style="green",
                    finished_style="green",
                    pulse_style="cyan",
                ),
                TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
                TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                expand=True,
            )
            self._episode_task_id = self._progress.add_task("episodes", total=max(0, total))
            self._step_task_id = self._progress.add_task(
                "actor steps",
                total=max(0, total_step_budget),
            )
            self._live = Live(
                self._render_rich(),
                console=self._console,
                refresh_per_second=6,
                transient=False,
                vertical_overflow="visible",
            )
            self._live.start()
        elif self.enabled:
            sys.stdout.write("\x1b[?25l")
            sys.stdout.flush()

    def start_match(
        self,
        *,
        match_id: str,
        detective: str,
        culprit: str,
        level: str,
        seed: int,
        detective_step_budget: int,
        step_budget: int,
    ) -> None:
        with self._lock:
            self.active_matches[match_id] = {
                "match_id": match_id,
                "detective": detective,
                "culprit": culprit,
                "level": level,
                "seed": seed,
                "detective_step_budget": max(0, detective_step_budget),
                "culprit_step_budget": max(
                    0,
                    step_budget - detective_step_budget,
                ),
                "step_budget": max(0, step_budget),
                "accounted": 0,
                "detective_steps": 0,
                "culprit_steps": 0,
                "last_role": "",
                "last_action": "",
                "last_success": True,
                "started": time.monotonic(),
                "updated": time.monotonic(),
            }
        if self.enabled:
            self.render()

    def record_step(self, match_id: str, step: dict[str, Any]) -> None:
        with self._lock:
            active = self.active_matches.get(match_id)
            if active is None:
                return
            role = str(step.get("role", "detective"))
            active["accounted"] += 1
            active["updated"] = time.monotonic()
            active["last_role"] = role
            active["last_action"] = str(step.get("action", ""))
            active["last_success"] = bool(step.get("success", True))
            if role == "culprit":
                active["culprit_steps"] += 1
            else:
                active["detective_steps"] += 1
            self._step_observed += 1
            self._step_accounted += 1
        if self.enabled:
            self.render()

    def finish_match(self, match_id: str) -> None:
        with self._lock:
            active = self.active_matches.pop(match_id, None)
            if active is not None:
                remaining = max(
                    0,
                    int(active.get("step_budget", 0)) - int(active.get("accounted", 0)),
                )
                self._step_accounted += remaining
                self._step_skipped += remaining
        if self.enabled:
            self.render(force=True)

    def skip_match(self, *, step_budget: int) -> None:
        with self._lock:
            skipped = max(0, step_budget)
            self._step_accounted += skipped
            self._step_skipped += skipped
        if self.enabled:
            self.render()

    def record(self, match: dict[str, Any]) -> None:
        with self._lock:
            self.matches.append(match)
        if self.enabled:
            self.render()
        else:
            status = "error" if match.get("error") else "ok"
            print(
                f"[{len(self.matches)}/{self.total}] "
                f"{match.get('detective', {}).get('name')} vs "
                f"{match.get('culprit', {}).get('name')} "
                f"{match.get('level')} seed={match.get('seed')} "
                f"payoff={float(match.get('detective_payoff', 0.0)):.3f} "
                f"{status}"
            )

    def render(self, *, force: bool = False) -> None:
        with self._lock:
            now = time.monotonic()
            if not force and now - self._last_render < 0.1:
                return
            self._last_render = now
            if self._rich_enabled:
                if self._closed:
                    return
                self._progress.update(
                    self._episode_task_id,
                    completed=len(self.matches),
                    total=max(0, self.total),
                )
                self._progress.update(
                    self._step_task_id,
                    completed=min(self._step_accounted, self.total_step_budget),
                    total=max(0, self.total_step_budget),
                )
                self._live.update(self._render_rich(), refresh=True)
                return
            width = shutil.get_terminal_size((100, 30)).columns
            lines = self._lines(width)
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write("\n".join(lines))
            sys.stdout.write("\n")
            sys.stdout.flush()

    def finish(self, outputs: dict[str, Any]) -> None:
        if self.enabled:
            self.render(force=True)
            self.close()
        self._print_final(outputs)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._rich_enabled and self._live is not None:
                self._live.stop()
            elif self.enabled:
                sys.stdout.write("\x1b[?25h\n")
                sys.stdout.flush()
            self._closed = True

    def _summary_stats(self) -> dict[str, Any]:
        done = len(self.matches)
        elapsed = max(0.001, time.monotonic() - self.started)
        rate = done / elapsed
        remaining = max(0, self.total - done)
        eta = remaining / rate if rate > 0 else 0.0
        errors = sum(1 for m in self.matches if m.get("error"))
        solved = sum(1 for m in self.matches if m.get("solved"))
        d_payoff = (
            sum(float(m.get("detective_payoff", 0.0)) for m in self.matches) / done
            if done else 0.0
        )
        c_payoff = (
            sum(float(m.get("culprit_payoff", 0.0)) for m in self.matches) / done
            if done else 0.0
        )
        return {
            "done": done,
            "elapsed": elapsed,
            "rate": rate,
            "eta": eta,
            "errors": errors,
            "solved": solved,
            "detective_payoff": d_payoff,
            "culprit_payoff": c_payoff,
            "solve_rate": solved / max(1, done),
            "guard_blocked": sum(int(m.get("guard_blocked_actions", 0) or 0) for m in self.matches),
            "guard_suppressed": sum(int(m.get("guard_suppressed_events", 0) or 0) for m in self.matches),
            "detective_failed": sum(int(m.get("detective_failed_actions", 0) or 0) for m in self.matches),
            "culprit_failed": sum(int(m.get("culprit_failed_actions", 0) or 0) for m in self.matches),
            "active": len(self.active_matches),
            "step_observed": self._step_observed,
            "step_accounted": self._step_accounted,
            "step_skipped": self._step_skipped,
            "step_budget": self.total_step_budget,
        }

    def _render_rich(self):
        stats = self._summary_stats()
        sections = [
            self._rich_header(stats),
            self._progress,
            self._rich_kpis(stats),
            self._rich_active_panel(),
            self._rich_role_grid(),
            self._rich_recent_panel(),
        ]
        error_panel = self._rich_error_panel()
        if error_panel is not None:
            sections.append(error_panel)
        return Group(*sections)

    def _rich_header(self, stats: dict[str, Any]):
        grid = Table.grid(expand=True, padding=(0, 2), collapse_padding=True)
        grid.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
        grid.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
        grid.add_row(
            "[bold]MysteryArena[/bold]  "
            f"[dim]mode[/dim] {self.mode}  "
            f"[dim]jobs[/dim] {stats['done']}/{self.total}  "
            f"[dim]workers[/dim] {self.workers}  "
            f"[dim]schedule[/dim] {self.schedule}  "
            f"[dim]resumed[/dim] {self.resumed}",
            f"[dim]run[/dim] {self.run_id}  "
            f"[dim]npc[/dim] {self.npc_provider}  "
            f"[dim]levels[/dim] {', '.join(self.levels) or '-'}  "
            f"[dim]seeds[/dim] {self._seed_summary()}",
        )
        grid.add_row(
            f"[dim]out[/dim] {self.out_dir}",
            f"[dim]gateway[/dim] {self.gateway_url or '-'}",
        )
        return Panel(
            grid,
            title="Arena Run",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _rich_kpis(self, stats: dict[str, Any]):
        cells = [
            self._kpi_cell("Completed", f"{stats['done']}/{self.total}", "cyan"),
            self._kpi_cell(
                "Actor Budget",
                f"{min(stats['step_accounted'], stats['step_budget'])}/{stats['step_budget']}",
                "cyan",
            ),
            self._kpi_cell("Actual Steps", str(stats["step_observed"]), "white"),
            self._kpi_cell("Active", str(stats["active"]), "yellow" if stats["active"] else "dim"),
            self._kpi_cell("D Payoff Mean", f"{stats['detective_payoff']:.3f}", "green"),
            self._kpi_cell("C Payoff Mean", f"{stats['culprit_payoff']:.3f}", "red"),
            self._kpi_cell("Solve Rate", f"{stats['solve_rate']:.1%}", "green"),
            self._kpi_cell(
                "Errors",
                str(stats["errors"]),
                "red" if stats["errors"] else "green",
            ),
            self._kpi_cell("D Action Failed", str(stats["detective_failed"]), "yellow"),
            self._kpi_cell("C Action Failed", str(stats["culprit_failed"]), "yellow"),
            self._kpi_cell(
                "Guard",
                f"{stats['guard_blocked']} / {stats['guard_suppressed']}",
                "yellow" if stats["guard_blocked"] or stats["guard_suppressed"] else "dim",
            ),
            self._kpi_cell("ETA", self._duration(stats["eta"]), "white"),
        ]
        grid = Table.grid(expand=True, padding=(0, 1), collapse_padding=True)
        for _ in range(4):
            grid.add_column(ratio=1, min_width=18)
        for idx in range(0, len(cells), 4):
            grid.add_row(*cells[idx:idx + 4])
        return Panel(
            grid,
            title="Live Metrics",
            border_style="white",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _kpi_cell(self, label: str, value: str, style: str) -> str:
        return f"[dim]{label}[/dim]\n[{style}]{value}[/{style}]"

    def _rich_role_grid(self):
        width = (
            self._console.size.width
            if self._console is not None
            else shutil.get_terminal_size((100, 30)).columns
        )
        grid = Table.grid(expand=True, padding=(0, 1), collapse_padding=True)
        if width < 120:
            grid.add_column(ratio=1)
            grid.add_row(self._rich_role_panel("detective"))
            grid.add_row(self._rich_role_panel("culprit"))
            return grid
        grid.add_column(ratio=1, min_width=48)
        grid.add_column(ratio=1, min_width=48)
        grid.add_row(self._rich_role_panel("detective"), self._rich_role_panel("culprit"))
        return grid

    def _rich_active_panel(self):
        if not self.active_matches:
            return Panel(
                "[dim]No active episodes[/dim]",
                title="Active Episodes",
                border_style="magenta",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        table = Table(
            box=box.SIMPLE,
            expand=True,
            show_edge=False,
            pad_edge=False,
        )
        table.add_column("Detective", ratio=2, overflow="ellipsis")
        table.add_column("Culprit", ratio=2, overflow="ellipsis")
        table.add_column("Case", width=12, overflow="ellipsis")
        table.add_column("D Steps", justify="right", width=9)
        table.add_column("C Steps", justify="right", width=9)
        table.add_column("Last", ratio=2, overflow="ellipsis")
        table.add_column("Idle", justify="right", width=7)
        active = sorted(
            self.active_matches.values(),
            key=lambda item: float(item.get("updated", 0.0)),
            reverse=True,
        )
        max_rows = max(4, min(10, self.workers + 1))
        now = time.monotonic()
        for item in active[:max_rows]:
            last_role = str(item.get("last_role") or "-")
            last_action = str(item.get("last_action") or "starting")
            if not item.get("last_success", True):
                role_style = action_style = "bold red"
            elif last_role == "detective":
                role_style = "green"
                action_style = "bold green"
            elif last_role == "culprit":
                role_style = "red"
                action_style = "bold red"
            else:
                role_style = action_style = "white"
            safe_role = rich_escape(last_role) if rich_escape is not None else last_role
            safe_action = rich_escape(last_action) if rich_escape is not None else last_action
            last = f"[{role_style}]{safe_role}[/{role_style}]:[{action_style}]{safe_action}[/{action_style}]"
            table.add_row(
                str(item.get("detective", "unknown")),
                str(item.get("culprit", "unknown")),
                f"{item.get('level')}:{item.get('seed')}",
                f"{item.get('detective_steps', 0)}/{item.get('detective_step_budget', 0)}",
                f"{item.get('culprit_steps', 0)}/{item.get('culprit_step_budget', 0)}",
                last,
                self._duration(now - float(item.get("updated", now))),
            )
        return Panel(
            table,
            title="Active Episodes",
            border_style="magenta",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _rich_role_panel(self, role: str):
        title = "Detective Ability" if role == "detective" else "Culprit Ability"
        value_label = "Solve" if role == "detective" else "Unsolve"
        color = "green" if role == "detective" else "red"
        table = Table(
            box=box.SIMPLE,
            expand=True,
            show_edge=False,
            pad_edge=False,
        )
        table.add_column("#", style="dim", justify="right", width=3)
        table.add_column("Model", ratio=3, overflow="ellipsis")
        table.add_column("Payoff Mean", justify="right", width=11)
        table.add_column("Skill", justify="right", width=7)
        table.add_column("Mu", justify="right", width=7)
        table.add_column("Sigma", justify="right", width=6)
        table.add_column("Ability", justify="left", width=30)
        table.add_column("N", justify="right", width=5)
        table.add_column(value_label, justify="right", width=8)
        rows = self._role_stats(role)
        max_rows = max(6, min(12, len(rows) if rows else 1))
        if not rows:
            table.add_row("-", "[dim]waiting for completed episodes[/dim]", "-", "-", "-", "-", "", "-", "-")
            rendered_rows = 1
        else:
            rendered_rows = 0
        for idx, row in enumerate(rows[:max_rows], 1):
            table.add_row(
                str(idx),
                str(row["name"]),
                f"{row['mean']:.3f}",
                f"{row['skill']:.1f}",
                f"{row['mu']:.1f}",
                f"{row['sigma']:.1f}",
                self._rich_skill_interval(row, color),
                str(row["n"]),
                f"{row['secondary']:.1%}",
            )
            rendered_rows += 1
        for _ in range(rendered_rows, max_rows):
            table.add_row("", "", "", "", "", "", "", "", "")
        return Panel(
            table,
            title=title,
            border_style=color,
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _role_stats(self, role: str) -> list[dict[str, Any]]:
        key = "detective" if role == "detective" else "culprit"
        payoff_key = "detective_payoff" if role == "detective" else "culprit_payoff"
        ratings = compute_role_trueskill(
            self.matches,
            mu=self.trueskill_mu,
            sigma=self.trueskill_sigma,
            beta=self.trueskill_beta,
            tau=self.trueskill_tau,
            draw_threshold=self.trueskill_draw_threshold,
        ).get(key, {})
        grouped: dict[str, list[dict[str, Any]]] = {}
        for match in self.matches:
            name = match.get(key, {}).get("name", "unknown")
            grouped.setdefault(name, []).append(match)
        rows = []
        for name, matches in grouped.items():
            values = [float(m.get(payoff_key, 0.0)) for m in matches]
            if role == "detective":
                secondary = sum(1.0 if m.get("solved") else 0.0 for m in matches) / len(matches)
            else:
                secondary = sum(0.0 if m.get("solved") else 1.0 for m in matches) / len(matches)
            rows.append({
                "name": name,
                "mean": sum(values) / len(values),
                "skill": float(ratings.get(name, {}).get("skill", self.trueskill_mu - 3.0 * self.trueskill_sigma)),
                "mu": float(ratings.get(name, {}).get("mu", self.trueskill_mu)),
                "sigma": float(ratings.get(name, {}).get("sigma", self.trueskill_sigma)),
                "n": len(matches),
                "secondary": secondary,
            })
        rows.sort(key=lambda row: (row["mean"], row["secondary"], row["skill"]), reverse=True)
        return rows

    def _rich_skill_interval(self, row: dict[str, Any], color: str, width: int = 28) -> str:
        """Render TrueSkill mu ± sigma on a fixed TrueSkill scale."""
        scale_min = -10.0
        scale_max = 50.0

        def pos(value: float) -> int:
            clamped = max(scale_min, min(scale_max, value))
            return int(round((clamped - scale_min) / (scale_max - scale_min) * (width - 1)))

        mu = float(row.get("mu", 25.0))
        sigma = max(0.0, float(row.get("sigma", 0.0)))
        center = pos(mu)
        radius = int(round(sigma / (scale_max - scale_min) * (width - 1)))
        lo = max(0, center - radius)
        hi = min(width - 1, center + radius)

        cells: list[str] = []
        for idx in range(width):
            if idx == center:
                cells.append("[bold white]│[/bold white]")
            elif lo <= idx <= hi:
                cells.append(f"[{color}]━[/{color}]")
            else:
                cells.append("[dim]─[/dim]")
        return "".join(cells)

    def _rich_recent_panel(self):
        if not self.matches:
            return Panel(
                "[dim]Waiting for completed episodes[/dim]",
                title="Recent Episodes",
                border_style="blue",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        table = Table(
            box=box.SIMPLE,
            expand=True,
            show_edge=False,
            pad_edge=False,
        )
        table.add_column("Result", ratio=2, min_width=10, overflow="ellipsis", no_wrap=True)
        table.add_column("Detective", ratio=2, overflow="ellipsis")
        table.add_column("Culprit", ratio=2, overflow="ellipsis")
        table.add_column("Case", ratio=1, overflow="ellipsis")
        table.add_column("D", justify="right", width=6)
        table.add_column("C", justify="right", width=6)
        table.add_column("Actions", justify="right", width=9)
        table.add_column("Time", justify="right", width=7)
        for match in self.matches[-self.tail:]:
            table.add_row(
                self._match_result(match, rich=True),
                match.get("detective", {}).get("name", "unknown"),
                match.get("culprit", {}).get("name", "unknown"),
                f"{match.get('level')}:{match.get('seed')}",
                f"{float(match.get('detective_payoff', 0.0)):.3f}",
                f"{float(match.get('culprit_payoff', 0.0)):.3f}",
                f"{match.get('actions_taken', 0)}/{match.get('culprit_actions_taken', 0)}",
                self._duration(float(match.get("elapsed_seconds", 0.0) or 0.0)),
            )
        return Panel(
            table,
            title="Recent Episodes",
            border_style="blue",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _match_result(self, match: dict[str, Any], *, rich: bool = False) -> str:
        if match.get("error"):
            label = f"ERR {self._brief_error(match.get('error'), max_len=28)}"
            return self._styled(label, "red", rich)
        detective_payoff = float(match.get("detective_payoff", 0.0))
        culprit_payoff = float(match.get("culprit_payoff", 0.0))
        if detective_payoff > culprit_payoff:
            return self._styled("D WIN", "green", rich)
        if culprit_payoff > detective_payoff:
            return self._styled("C WIN", "red", rich)
        return self._styled("DRAW", "yellow", rich)

    def _brief_error(self, error: Any, *, max_len: int = 40) -> str:
        text = str(error or "").strip()
        if not text:
            return "-"
        first_line = text.splitlines()[0].strip()
        if len(first_line) <= max_len:
            return first_line
        return first_line[: max(0, max_len - 1)].rstrip() + "…"

    def _styled(self, text: str, style: str, rich: bool) -> str:
        if not rich:
            return text
        safe_text = rich_escape(text) if rich_escape is not None else text.replace("[", r"\[")
        return f"[{style}]{safe_text}[/{style}]"

    def _rich_error_panel(self):
        errors = [m for m in self.matches if m.get("error")]
        if not errors:
            return None
        table = Table(
            box=box.SIMPLE,
            expand=True,
            show_edge=False,
            pad_edge=False,
        )
        table.add_column("Match", ratio=2, overflow="ellipsis")
        table.add_column("Error", ratio=5, overflow="fold")
        for match in errors[-3:]:
            message = str(match.get("error", "")).splitlines()[0]
            table.add_row(match.get("match_id", "unknown"), message)
        return Panel(
            table,
            title="Recent Errors",
            border_style="red",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _seed_summary(self) -> str:
        if not self.seeds:
            return "-"
        values = sorted(set(self.seeds))
        if len(values) <= 6:
            return ",".join(str(v) for v in values)
        contiguous = values == list(range(values[0], values[-1] + 1))
        if contiguous:
            return f"{values[0]}-{values[-1]} ({len(values)})"
        return f"{len(values)} seeds"

    def _lines(self, width: int) -> list[str]:
        stats = self._summary_stats()
        done = stats["done"]
        bar = self._bar(done, self.total, max(20, min(48, width - 40)))
        lines = [
            "MysteryArena Run".ljust(width, "="),
            f"run_id: {self.run_id}",
            f"mode: {self.mode} | out: {self.out_dir}",
            f"progress: {bar} {done}/{self.total} "
            f"({done / max(1, self.total):.1%})",
            f"actor budget: {self._bar(min(stats['step_accounted'], stats['step_budget']), stats['step_budget'], max(20, min(48, width - 40)))} "
            f"{min(stats['step_accounted'], stats['step_budget'])}/{stats['step_budget']} "
            f"| actual steps: {stats['step_observed']} | active: {stats['active']}",
            f"elapsed: {self._duration(stats['elapsed'])} | "
            f"eta: {self._duration(stats['eta'])} "
            f"| rate: {stats['rate']:.2f} eps/s | errors: {stats['errors']}",
            f"solve_rate: {stats['solve_rate']:.1%} | "
            f"mean_detective_payoff: {stats['detective_payoff']:.3f}",
            "",
            "Active episodes",
            *self._active_rows(width),
            "",
            "Detective ability (live mean)",
            *self._role_rows("detective"),
            "",
            "Culprit ability (live mean)",
            *self._role_rows("culprit"),
            "",
            "Recent episodes",
            *self._recent_rows(width),
        ]
        return lines

    def _bar(self, done: int, total: int, width: int) -> str:
        if total <= 0:
            return "[" + " " * width + "]"
        filled = int(width * done / total)
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def _duration(self, seconds: float) -> str:
        seconds = int(seconds)
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:02d}:{sec:02d}"

    def _active_rows(self, width: int) -> list[str]:
        if not self.active_matches:
            return ["  (none)"]
        rows = []
        active = sorted(
            self.active_matches.values(),
            key=lambda item: float(item.get("updated", 0.0)),
            reverse=True,
        )
        for item in active[: max(4, min(10, self.workers + 1))]:
            text = (
                f"  {item.get('detective')} vs {item.get('culprit')} | "
                f"{item.get('level')} seed={item.get('seed')} | "
                f"D={item.get('detective_steps', 0)}/{item.get('detective_step_budget', 0)} "
                f"C={item.get('culprit_steps', 0)}/{item.get('culprit_step_budget', 0)} | "
                f"last={item.get('last_role') or '-'}:{item.get('last_action') or 'starting'}"
            )
            rows.append(text[: max(20, width - 1)])
        return rows

    def _role_rows(self, role: str) -> list[str]:
        rows = self._role_stats(role)
        rendered = []
        for idx, row in enumerate(rows[:12], 1):
            rendered.append(
                f"  {idx:>2}. {row['name']:<24} "
                f"mean={row['mean']:.3f} skill={row['skill']:.1f} "
                f"mu={row['mu']:.1f} sigma={row['sigma']:.1f} n={row['n']}"
            )
        return rendered or ["  (waiting for completed episodes)"]

    def _print_final(self, outputs: dict[str, Any]) -> None:
        if self._console is not None and Table is not None:
            self._console.print(self._final_table("Detective Leaderboard", outputs["detective_leaderboard"], "green"))
            self._console.print(self._final_table("Culprit Leaderboard", outputs["culprit_leaderboard"], "red"))
            return
        print("Detective leaderboard:")
        for row in outputs["detective_leaderboard"]:
            rating = row.get("trueskill", {})
            print(
                f"  #{row['rank']} {row['model']}: "
                f"{row['mean_payoff']:.3f} Skill={rating.get('skill')} "
                f"mu={rating.get('mu')} sigma={rating.get('sigma')}"
            )
        print("\nCulprit leaderboard:")
        for row in outputs["culprit_leaderboard"]:
            rating = row.get("trueskill", {})
            print(
                f"  #{row['rank']} {row['model']}: "
                f"{row['mean_payoff']:.3f} Skill={rating.get('skill')} "
                f"mu={rating.get('mu')} sigma={rating.get('sigma')}"
            )

    def _final_table(self, title: str, rows: list[dict[str, Any]], color: str):
        table = Table(
            title=title,
            box=box.ROUNDED,
            border_style=color,
            show_lines=False,
        )
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("Model", overflow="ellipsis")
        table.add_column("Mean Payoff", justify="right")
        table.add_column("Skill", justify="right")
        table.add_column("Mu", justify="right")
        table.add_column("Sigma", justify="right")
        table.add_column("N", justify="right")
        for row in rows[:12]:
            rating = row.get("trueskill", {})
            table.add_row(
                str(row["rank"]),
                str(row["model"]),
                f"{float(row['mean_payoff']):.3f}",
                str(rating.get("skill", "")),
                str(rating.get("mu", "")),
                str(rating.get("sigma", "")),
                str(row["n"]),
            )
        return table

    def _recent_rows(self, width: int) -> list[str]:
        rows = []
        for match in self.matches[-self.tail:]:
            text = (
                f"  {self._match_result(match)} "
                f"{match.get('detective', {}).get('name')} vs "
                f"{match.get('culprit', {}).get('name')} | "
                f"{match.get('level')} seed={match.get('seed')} | "
                f"D={float(match.get('detective_payoff', 0.0)):.3f} "
                f"C={float(match.get('culprit_payoff', 0.0)):.3f}"
            )
            rows.append(text[: max(20, width - 1)])
        return rows or ["  (none yet)"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MysteryArena Arena matches")
    parser.add_argument("--mode", choices=["detective", "culprit", "matrix"], default="matrix")
    parser.add_argument("--detectives", default="heuristic")
    parser.add_argument("--culprits", default="passive")
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["TRIVIAL", "EASY", "MEDIUM", "HARD", "EXPERT"],
    )
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--out", required=True, help="Arena output directory")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume from completed trajectories already present under --out. "
            "Use --no-resume to force rerunning every scheduled job."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Backward-compatible alias for --resume.",
    )
    parser.add_argument(
        "--schedule",
        choices=["balanced", "row-major"],
        default="balanced",
        help=(
            "Job ordering. balanced diagonalizes detective/culprit pairs so "
            "early workers cover different models; row-major preserves the "
            "old detective-major order."
        ),
    )
    parser.add_argument(
        "--tui",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show a live terminal progress UI. Defaults to on for TTY output.",
    )
    parser.add_argument("--tui-tail", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--trueskill-mu", type=float, default=25.0)
    parser.add_argument("--trueskill-sigma", type=float, default=25.0 / 3.0)
    parser.add_argument("--trueskill-beta", type=float, default=25.0 / 6.0)
    parser.add_argument("--trueskill-tau", type=float, default=25.0 / 300.0)
    parser.add_argument("--trueskill-draw-threshold", type=float, default=0.0)
    parser.add_argument("--elo-initial", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--elo-k", type=float, default=None, help=argparse.SUPPRESS)

    parser.add_argument("--npc-provider", default="fallback", choices=["fallback", "openai", "openrouter", "vllm"])
    parser.add_argument("--npc-model", default="gpt-4o-mini")
    parser.add_argument("--npc-url", default=None)
    parser.add_argument("--npc-seed", type=int, default=42)

    parser.add_argument("--litellm-url", default=None)
    parser.add_argument("--litellm-key-env", default=None)
    parser.add_argument("--litellm-model", default=None)
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Load gateway/API variables from this dotenv file if present.",
    )
    parser.add_argument(
        "--gateway-url-env",
        default="LLM_GATEWAY_URL",
        help="Env var containing the OpenAI-compatible gateway URL.",
    )
    parser.add_argument(
        "--gateway-key-env",
        default="LLM_GATEWAY_API_KEY",
        help="Env var containing the gateway API key.",
    )
    args = parser.parse_args()

    _load_env_file(args.env_file)
    if not args.litellm_url:
        args.litellm_url = os.environ.get(args.gateway_url_env)
    if not args.litellm_key_env and os.environ.get(args.gateway_key_env):
        args.litellm_key_env = args.gateway_key_env

    if args.litellm_url:
        BaseAgent.configure_litellm(
            args.litellm_url,
            api_key_env=args.litellm_key_env,
            model=args.litellm_model,
        )
        print(
            f"LLM gateway: {args.litellm_url} "
            f"(key_env={args.litellm_key_env or 'none'}, "
            f"model={args.litellm_model or 'per-role'})"
        )

    run_id = args.run_id or f"arena_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detectives = parse_model_list(args.detectives, role="detective")
    culprits = parse_model_list(args.culprits, role="culprit")
    levels = _parse_levels(args.levels)
    seeds = _parse_seeds(args.seeds)
    npc = {
        "provider": args.npc_provider,
        "model": args.npc_model,
        "url": args.npc_url,
        "seed": args.npc_seed,
        "prompt_policy": "role_facts_only_no_strategy",
    }
    jobs = _build_jobs(args.mode, detectives, culprits, levels, seeds, args.schedule)
    resume_existing = bool(args.resume or args.skip_existing)
    if resume_existing:
        resumed_entries, run_jobs, ignored_resume = _load_resumable_jobs(
            jobs=jobs,
            out_dir=out_dir,
            run_id=run_id,
            npc=npc,
        )
    else:
        resumed_entries = []
        run_jobs = jobs
        ignored_resume = []

    config = {
        "run_id": run_id,
        "mode": args.mode,
        "levels": levels,
        "seeds": seeds,
        "npc": npc,
        "detectives": [d.to_dict() for d in detectives],
        "culprits": [c.to_dict() for c in culprits],
        "rating": {
            "system": "trueskill",
            "trueskill_mu": args.trueskill_mu,
            "trueskill_sigma": args.trueskill_sigma,
            "trueskill_beta": args.trueskill_beta,
            "trueskill_tau": args.trueskill_tau,
            "trueskill_draw_threshold": args.trueskill_draw_threshold,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "schedule": args.schedule,
        "resume": resume_existing,
        "llm_gateway": {
            "url": args.litellm_url,
            "key_env": args.litellm_key_env,
            "model_override": args.litellm_model,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out_dir / "roster.json").write_text(
        json.dumps(
            {
                "detectives": [d.to_dict() for d in detectives],
                "culprits": [c.to_dict() for c in culprits],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Arena run: {run_id}")
    print(f"Mode: {args.mode} | jobs={len(jobs)} | schedule={args.schedule} | out={out_dir}")
    if resume_existing:
        print(
            f"Resume: loaded={len(resumed_entries)} "
            f"remaining={len(run_jobs)} ignored={len(ignored_resume)}"
        )

    matches: list[dict[str, Any]] = [match for match, _ in resumed_entries]
    use_tui = sys.stdout.isatty() if args.tui is None else args.tui
    total_step_budget = sum(
        _episode_actor_step_budget(level, culprit)
        for _, culprit, level, _ in jobs
    )
    resumed_step_accounted = sum(step_budget for _, step_budget in resumed_entries)
    resumed_step_observed = sum(_match_actor_steps(match) for match, _ in resumed_entries)
    resumed_step_skipped = sum(
        max(0, step_budget - _match_actor_steps(match))
        for match, step_budget in resumed_entries
    )
    progress = ArenaProgress(
        total=len(jobs),
        run_id=run_id,
        mode=args.mode,
        out_dir=out_dir,
        enabled=use_tui,
        tail=args.tui_tail,
        workers=args.workers,
        schedule=args.schedule,
        levels=levels,
        seeds=seeds,
        gateway_url=args.litellm_url,
        npc_provider=args.npc_provider,
        total_step_budget=total_step_budget,
        resumed=len(resumed_entries),
        initial_matches=matches,
        initial_step_accounted=resumed_step_accounted,
        initial_step_observed=resumed_step_observed,
        initial_step_skipped=resumed_step_skipped,
        trueskill_mu=args.trueskill_mu,
        trueskill_sigma=args.trueskill_sigma,
        trueskill_beta=args.trueskill_beta,
        trueskill_tau=args.trueskill_tau,
        trueskill_draw_threshold=args.trueskill_draw_threshold,
    )
    interrupted = False
    try:
        if args.workers <= 1:
            for detective, culprit, level, seed in run_jobs:
                match = _run_one(
                    run_id=run_id,
                    out_dir=out_dir,
                    detective=detective,
                    culprit=culprit,
                    npc=npc,
                    level=level,
                    seed=seed,
                    skip_existing=resume_existing,
                    progress=progress,
                )
                matches.append(match)
                progress.record(match)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                future_to_job = {
                    pool.submit(
                        _run_one,
                        run_id=run_id,
                        out_dir=out_dir,
                        detective=detective,
                        culprit=culprit,
                        npc=npc,
                        level=level,
                        seed=seed,
                        skip_existing=resume_existing,
                        progress=progress,
                    ): (detective, culprit, level, seed)
                    for detective, culprit, level, seed in run_jobs
                }
                for future in as_completed(future_to_job):
                    match = future.result()
                    matches.append(match)
                    progress.record(match)
    except KeyboardInterrupt:
        interrupted = True
        progress.close()
        print("Interrupted. Completed matches will be written before exiting.")

    matches.sort(key=lambda m: (m["detective"]["name"], m["culprit"]["name"], str(m["level"]), int(m["seed"])))
    with (out_dir / "matches.jsonl").open("w", encoding="utf-8") as fh:
        for match in matches:
            fh.write(json.dumps(match, default=str) + "\n")

    outputs = write_outputs(
        out_dir,
        bootstrap_samples=args.bootstrap_samples,
        trueskill_mu=args.trueskill_mu,
        trueskill_sigma=args.trueskill_sigma,
        trueskill_beta=args.trueskill_beta,
        trueskill_tau=args.trueskill_tau,
        trueskill_draw_threshold=args.trueskill_draw_threshold,
    )
    progress.finish(outputs)
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
