"""Background jobs for running and publishing Arena results."""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from arena.hf_publish import HFPublishError, publish_run_to_hf


JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


@dataclass
class ArenaJob:
    job_id: str
    kind: str
    status: JobStatus = "queued"
    run_id: str | None = None
    run_dir: str | None = None
    repo_id: str | None = None
    command: list[str] | None = None
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    updated_at: str = field(default_factory=_now)
    returncode: int | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=500))
    process: subprocess.Popen[str] | None = None
    thread: threading.Thread | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    def append_log(self, line: str) -> None:
        with self.lock:
            self.logs.append(line.rstrip("\n"))
            self.updated_at = _now()

    def snapshot(self, *, tail: int = 80) -> dict[str, Any]:
        with self.lock:
            logs = list(self.logs)
            if tail > 0:
                logs = logs[-tail:]
            return {
                "job_id": self.job_id,
                "kind": self.kind,
                "status": self.status,
                "run_id": self.run_id,
                "run_dir": self.run_dir,
                "repo_id": self.repo_id,
                "command": self.command,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "updated_at": self.updated_at,
                "returncode": self.returncode,
                "error": self.error,
                "result": self.result,
                "logs": logs,
            }


class ArenaJobManager:
    """In-process background job registry for the API server."""

    def __init__(
        self,
        *,
        arena_root: str | Path,
        env_file: str | Path | None,
    ) -> None:
        self.arena_root = Path(arena_root)
        self.env_file = str(env_file) if env_file else None
        self._jobs: dict[str, ArenaJob] = {}
        self._lock = threading.RLock()

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [job.snapshot(tail=20) for job in sorted(jobs, key=lambda j: j.created_at, reverse=True)]

    def get(self, job_id: str) -> ArenaJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def start_run(
        self,
        *,
        mode: str,
        detectives: str,
        culprits: str,
        levels: list[str],
        seeds: str,
        run_id: str,
        out_dir: str | Path | None = None,
        workers: int = 1,
        schedule: str = "balanced",
        resume: bool = True,
        npc: dict[str, Any] | None = None,
        bootstrap_samples: int = 1000,
    ) -> ArenaJob:
        target_dir = Path(out_dir) if out_dir else self.arena_root / run_id
        script = Path(__file__).resolve().parent.parent / "scripts" / "arena_run.py"
        npc = npc or {}
        command = [
            sys.executable,
            str(script),
            "--mode",
            mode,
            "--detectives",
            detectives,
            "--culprits",
            culprits,
            "--levels",
            *levels,
            "--seeds",
            seeds,
            "--workers",
            str(workers),
            "--out",
            str(target_dir),
            "--run-id",
            run_id,
            "--schedule",
            schedule,
            "--bootstrap-samples",
            str(bootstrap_samples),
            "--no-tui",
        ]
        command.append("--resume" if resume else "--no-resume")
        command.extend(["--env-file", self.env_file or ""])
        if npc.get("provider"):
            command.extend(["--npc-provider", str(npc.get("provider"))])
        if npc.get("model"):
            command.extend(["--npc-model", str(npc.get("model"))])
        if npc.get("url"):
            command.extend(["--npc-url", str(npc.get("url"))])
        if npc.get("seed") is not None:
            command.extend(["--npc-seed", str(npc.get("seed"))])

        job = self._register(
            ArenaJob(
                job_id=uuid.uuid4().hex,
                kind="arena_run",
                run_id=run_id,
                run_dir=str(target_dir),
                command=command,
            )
        )
        self._start_thread(job, self._run_subprocess, job)
        return job

    def start_publish(
        self,
        *,
        run_id: str,
        repo_id: str,
        private: bool = False,
        revision: str | None = None,
        create_pr: bool = False,
        include_model_responses: bool = True,
    ) -> ArenaJob:
        run_dir = self.arena_root / run_id
        job = self._register(
            ArenaJob(
                job_id=uuid.uuid4().hex,
                kind="hf_publish",
                run_id=run_id,
                run_dir=str(run_dir),
                repo_id=repo_id,
            )
        )
        self._start_thread(
            job,
            self._publish,
            job,
            repo_id,
            private,
            revision,
            create_pr,
            include_model_responses,
        )
        return job

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        with job.lock:
            if job.status not in {"queued", "running"}:
                return job.snapshot()
            job.status = "cancelled"
            job.finished_at = _now()
            job.updated_at = job.finished_at
            process = job.process
        if process is not None and process.poll() is None:
            process.terminate()
        return job.snapshot()

    def _register(self, job: ArenaJob) -> ArenaJob:
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def _start_thread(self, job: ArenaJob, target, *args: Any) -> None:
        thread = threading.Thread(target=target, args=args, daemon=True)
        with job.lock:
            job.thread = thread
        thread.start()

    def _mark_running(self, job: ArenaJob) -> bool:
        with job.lock:
            if job.status == "cancelled":
                return False
            job.status = "running"
            job.started_at = _now()
            job.updated_at = job.started_at
            return True

    def _mark_finished(
        self,
        job: ArenaJob,
        *,
        status: JobStatus,
        error: str | None = None,
        result: dict[str, Any] | None = None,
        returncode: int | None = None,
    ) -> None:
        with job.lock:
            if job.status == "cancelled" and status != "cancelled":
                return
            job.status = status
            job.error = error
            job.result = result
            job.returncode = returncode
            job.finished_at = _now()
            job.updated_at = job.finished_at

    def _run_subprocess(self, job: ArenaJob) -> None:
        if not self._mark_running(job):
            return
        assert job.command is not None
        job.append_log("$ " + " ".join(job.command))
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            process = subprocess.Popen(
                job.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path(__file__).resolve().parent.parent),
                env=env,
            )
            with job.lock:
                job.process = process
            assert process.stdout is not None
            for line in process.stdout:
                job.append_log(line)
            returncode = process.wait()
            if returncode == 0:
                self._mark_finished(
                    job,
                    status="succeeded",
                    returncode=returncode,
                    result={"run_id": job.run_id, "run_dir": job.run_dir},
                )
            else:
                self._mark_finished(
                    job,
                    status="failed",
                    returncode=returncode,
                    error=f"arena_run exited with {returncode}",
                )
        except Exception as exc:  # noqa: BLE001 - capture background job failures.
            self._mark_finished(job, status="failed", error=f"{type(exc).__name__}: {exc}")

    def _publish(
        self,
        job: ArenaJob,
        repo_id: str,
        private: bool,
        revision: str | None,
        create_pr: bool,
        include_model_responses: bool,
    ) -> None:
        if not self._mark_running(job):
            return
        try:
            job.append_log(f"Publishing {job.run_dir} to dataset {repo_id}")
            result = publish_run_to_hf(
                job.run_dir or "",
                repo_id=repo_id,
                private=private,
                revision=revision,
                create_pr=create_pr,
                include_model_responses=include_model_responses,
            )
            self._mark_finished(job, status="succeeded", result=result)
        except HFPublishError as exc:
            self._mark_finished(job, status="failed", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - capture background job failures.
            self._mark_finished(job, status="failed", error=f"{type(exc).__name__}: {exc}")


__all__ = ["ArenaJob", "ArenaJobManager", "JobStatus"]
