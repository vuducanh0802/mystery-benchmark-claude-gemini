"""Package and publish Arena runs as public Hugging Face Dataset artifacts."""

from __future__ import annotations

import copy
import datetime as dt
import gzip
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arena.aggregate import aggregate_matches, load_matches
from arena.metrics import read_jsonl


SCHEMA_VERSION = 1
ALL_MATCHES_FILE = "matches/all_matches.jsonl.gz"
SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "endpoint",
    "password",
    "secret",
    "token",
    "url",
)
SENSITIVE_KEYS = {"key", "key_env", "url"}
SECRET_PATTERNS = (
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [redacted]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "[redacted-secret]"),
    (re.compile(r"https?://[^\s\"'<>]+"), "[redacted-url]"),
)


class HFPublishError(RuntimeError):
    """Raised when an Arena run cannot be packaged or published."""


@dataclass(frozen=True)
class HFPackageResult:
    run_id: str
    package_dir: Path
    summary: dict[str, Any]
    files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "package_dir": str(self.package_dir),
            "summary": self.summary,
            "files": self.files,
        }


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _json_default(value: Any) -> str:
    return str(value)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HFPublishError(f"invalid JSON in {path}") from exc


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _write_jsonl_gz(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=_json_default))
            fh.write("\n")


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "episode"


def _brief_error(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    first = text.splitlines()[0].strip()
    return _redact_string(first[:500])


def _redact_string(value: str) -> str:
    redacted = value
    for pattern, replacement in SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        redacted = {}
        for key, value in obj.items():
            norm = str(key).strip().lower()
            if norm in SENSITIVE_KEYS or any(part in norm for part in SENSITIVE_KEY_PARTS):
                redacted[key] = "[redacted]"
            else:
                redacted[key] = _redact(value)
        return redacted
    if isinstance(obj, list):
        return [_redact(item) for item in obj]
    if isinstance(obj, str):
        return _redact_string(obj)
    return obj


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    raw_gateway = config.get("llm_gateway")
    gateway_configured = bool(raw_gateway.get("url")) if isinstance(raw_gateway, dict) else False
    public = _redact(copy.deepcopy(config))
    gateway = public.get("llm_gateway")
    if isinstance(gateway, dict):
        public["llm_gateway"] = {
            "configured": gateway_configured,
            "model_override": gateway.get("model_override"),
        }
    npc = public.get("npc")
    if isinstance(npc, dict):
        npc.pop("url", None)
    return public


def _public_match(
    match: dict[str, Any],
    *,
    trajectory_file: str | None,
) -> dict[str, Any]:
    public = _redact(copy.deepcopy(match))
    public.pop("trajectory_path", None)
    if public.get("error"):
        public["error"] = _brief_error(public.get("error"))
    public["trajectory_available"] = bool(trajectory_file)
    if trajectory_file:
        public["trajectory_file"] = trajectory_file
    npc = public.get("npc")
    if isinstance(npc, dict):
        npc.pop("url", None)
    return public


def _public_trajectory_record(
    record: dict[str, Any],
    *,
    include_model_responses: bool,
) -> dict[str, Any]:
    public = _redact(copy.deepcopy(record))
    if not include_model_responses:
        public.pop("model_response", None)
    if public.get("kind") == "footer" and public.get("error"):
        public["error"] = _brief_error(public.get("error"))
    return public


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


def _trajectory_path(run_dir: Path, match: dict[str, Any]) -> Path | None:
    raw = match.get("trajectory_path")
    if not raw:
        return None
    candidates = [Path(raw)]
    if not Path(raw).is_absolute():
        candidates.append(run_dir / raw)
    root = run_dir.resolve()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists() and (resolved == root or root in resolved.parents):
            return resolved
    return None


def _index_entry(summary: dict[str, Any]) -> dict[str, Any]:
    config = summary.get("config", {})
    outputs = summary.get("outputs", {})
    out_summary = outputs.get("summary", {})
    best_detective = (outputs.get("detective_leaderboard") or [{}])[0].get("model")
    best_culprit = (outputs.get("culprit_leaderboard") or [{}])[0].get("model")
    return {
        "run_id": summary["run_id"],
        "published_at": summary["published_at"],
        "mode": config.get("mode"),
        "levels": config.get("levels", []),
        "seeds": config.get("seeds", []),
        "matches": out_summary.get("matches", 0),
        "detectives": out_summary.get("detectives", 0),
        "culprits": out_summary.get("culprits", 0),
        "top_detective": best_detective,
        "top_culprit": best_culprit,
        "summary_file": f"runs/{summary['run_id']}/summary.json",
        "matches_file": f"runs/{summary['run_id']}/matches.jsonl.gz",
    }


def _merge_index(existing_runs: list[dict[str, Any]], entry: dict[str, Any]) -> dict[str, Any]:
    by_id = {
        str(item.get("run_id")): item
        for item in existing_runs
        if item.get("run_id")
    }
    by_id[entry["run_id"]] = entry
    runs = sorted(
        by_id.values(),
        key=lambda item: str(item.get("published_at") or ""),
        reverse=True,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _now(),
        "latest_run_id": runs[0].get("run_id") if runs else None,
        "matches_file": ALL_MATCHES_FILE,
        "split": "matches",
        "total_matches": sum(int(item.get("matches") or 0) for item in runs),
        "runs": runs,
    }


def _safe_split_name(value: Any) -> str:
    split = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip())
    split = re.sub(r"_+", "_", split).strip("_")
    return split or "matches"


def _dataset_readme(index: dict[str, Any]) -> str:
    lines = [
        "---",
        "configs:",
        "- config_name: default",
        "  data_files:",
        "  - split: matches",
        f"    path: {index.get('matches_file') or ALL_MATCHES_FILE}",
        "---",
        "",
        "# MysteryArena Results",
        "",
        "This dataset stores public MysteryArena run summaries, match records, "
        "and compressed episode trajectories for the read-only Streamlit frontend.",
        "",
        "The frontend reads `index/runs.json` first, then loads the unified "
        "`matches/all_matches.jsonl.gz` split. Per-run summaries and compressed "
        "trajectory files are kept for metadata and replay.",
        "",
    ]
    return "\n".join(lines)


def _match_sort_key(match: dict[str, Any]) -> tuple[str, str, str, str, int, str]:
    return (
        str(match.get("run_id") or ""),
        str(match.get("level") or ""),
        str((match.get("detective") or {}).get("name") or ""),
        str((match.get("culprit") or {}).get("name") or ""),
        int(match.get("seed") or 0),
        str(match.get("match_id") or ""),
    )


def _merge_public_matches(
    existing_matches: list[dict[str, Any]],
    current_matches: list[dict[str, Any]],
    *,
    current_run_id: str,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for match in existing_matches:
        run_id = str(match.get("run_id") or "")
        if run_id == current_run_id:
            continue
        match_id = str(match.get("match_id") or "")
        by_key[(run_id, match_id)] = match
    for match in current_matches:
        run_id = str(match.get("run_id") or current_run_id)
        match_id = str(match.get("match_id") or "")
        match["run_id"] = run_id
        by_key[(run_id, match_id)] = match
    return sorted(by_key.values(), key=_match_sort_key)


def package_run_for_hf(
    run_dir: str | Path,
    *,
    package_dir: str | Path | None = None,
    existing_runs: list[dict[str, Any]] | None = None,
    existing_matches: list[dict[str, Any]] | None = None,
    include_model_responses: bool = True,
) -> HFPackageResult:
    """Build a Hugging Face Dataset upload folder for one Arena run."""
    source = Path(run_dir).resolve()
    if not source.exists() or not source.is_dir():
        raise HFPublishError(f"run directory not found: {source}")

    config = _read_json(source / "config.json", {})
    run_id = str(config.get("run_id") or source.name)
    matches = load_matches(source)
    outputs = aggregate_matches(matches, **_rating_kwargs(config)) if matches else {
        "detective_leaderboard": [],
        "culprit_leaderboard": [],
        "ratings": {"system": "trueskill", "detective": {}, "culprit": {}},
        "matrix": {},
        "summary": {"matches": 0, "detectives": 0, "culprits": 0},
    }

    root = (Path(package_dir) if package_dir else Path(tempfile.mkdtemp(prefix="arena_hf_"))).resolve()
    if root == source or source in root.parents:
        raise HFPublishError("package_dir must not be the run directory or inside it")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    run_prefix = Path("runs") / run_id
    trajectories_prefix = run_prefix / "trajectories"
    public_matches = []
    files: list[str] = []

    for idx, match in enumerate(matches):
        match_id = str(match.get("match_id") or f"episode_{idx}")
        trajectory_rel: str | None = None
        path = _trajectory_path(source, match)
        if path is not None:
            trajectory_name = f"{_safe_filename(match_id)}.jsonl.gz"
            trajectory_rel = str(trajectories_prefix / trajectory_name)
            records = [
                _public_trajectory_record(record, include_model_responses=include_model_responses)
                for record in read_jsonl(path)
            ]
            _write_jsonl_gz(root / trajectory_rel, records)
            files.append(trajectory_rel)
        public_matches.append(_public_match(match, trajectory_file=trajectory_rel))

    all_matches = _merge_public_matches(
        existing_matches or [],
        public_matches,
        current_run_id=run_id,
    )
    all_matches_rel = ALL_MATCHES_FILE
    _write_jsonl_gz(root / all_matches_rel, all_matches)
    files.append(all_matches_rel)

    matches_rel = str(run_prefix / "matches.jsonl.gz")
    _write_jsonl_gz(root / matches_rel, public_matches)
    files.append(matches_rel)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "published_at": _now(),
        "source": {
            "type": "mystery-benchmark-arena",
            "source_dir_name": source.name,
        },
        "config": _public_config(config),
        "outputs": _redact(outputs),
        "files": {
            "summary": str(run_prefix / "summary.json"),
            "matches": matches_rel,
            "all_matches": all_matches_rel,
            "trajectories_prefix": str(trajectories_prefix),
        },
    }
    summary_rel = str(run_prefix / "summary.json")
    _write_json(root / summary_rel, summary)
    files.append(summary_rel)

    index = _merge_index(existing_runs or [], _index_entry(summary))
    index_rel = "index/runs.json"
    _write_json(root / index_rel, index)
    files.append(index_rel)

    (root / "README.md").write_text(_dataset_readme(index), encoding="utf-8")
    files.append("README.md")

    return HFPackageResult(
        run_id=run_id,
        package_dir=root,
        summary=summary,
        files=sorted(files),
    )


def _remote_index_runs(
    *,
    repo_id: str,
    token: str | None,
    revision: str | None,
) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise HFPublishError("huggingface_hub is required to publish to Hugging Face") from exc

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="index/runs.json",
            repo_type="dataset",
            revision=revision,
            token=token,
        )
    except Exception:
        return []
    payload = _read_json(Path(path), {})
    runs = payload.get("runs", [])
    return runs if isinstance(runs, list) else []


def _remote_matches_from_path(
    *,
    repo_id: str,
    token: str | None,
    revision: str | None,
    filename: str,
) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise HFPublishError("huggingface_hub is required to publish to Hugging Face") from exc

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            token=token,
        )
    except Exception:
        return []
    return _read_jsonl_gz(Path(path))


def _remote_public_matches(
    *,
    repo_id: str,
    token: str | None,
    revision: str | None,
    existing_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    all_matches = _remote_matches_from_path(
        repo_id=repo_id,
        token=token,
        revision=revision,
        filename=ALL_MATCHES_FILE,
    )
    if all_matches:
        return all_matches

    merged: list[dict[str, Any]] = []
    for item in existing_runs:
        matches_file = item.get("matches_file")
        if not matches_file:
            continue
        merged.extend(
            _remote_matches_from_path(
                repo_id=repo_id,
                token=token,
                revision=revision,
                filename=str(matches_file),
            )
        )
    return merged


def publish_run_to_hf(
    run_dir: str | Path,
    *,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
    revision: str | None = None,
    create_pr: bool = False,
    package_dir: str | Path | None = None,
    include_model_responses: bool = True,
    commit_message: str | None = None,
) -> dict[str, Any]:
    """Package an Arena run and upload it to a Hugging Face Dataset repo."""
    resolved_repo = repo_id or os.environ.get("ARENA_HF_DATASET")
    if not resolved_repo:
        raise HFPublishError("repo_id is required; set ARENA_HF_DATASET or pass repo_id")
    resolved_token = token or os.environ.get("HF_TOKEN")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise HFPublishError("huggingface_hub is required to publish to Hugging Face") from exc

    api = HfApi(token=resolved_token)
    api.create_repo(
        repo_id=resolved_repo,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    existing_runs = _remote_index_runs(
        repo_id=resolved_repo,
        token=resolved_token,
        revision=revision,
    )
    existing_matches = _remote_public_matches(
        repo_id=resolved_repo,
        token=resolved_token,
        revision=revision,
        existing_runs=existing_runs,
    )
    package = package_run_for_hf(
        run_dir,
        package_dir=package_dir,
        existing_runs=existing_runs,
        existing_matches=existing_matches,
        include_model_responses=include_model_responses,
    )
    upload_result = api.upload_folder(
        repo_id=resolved_repo,
        repo_type="dataset",
        folder_path=str(package.package_dir),
        revision=revision,
        create_pr=create_pr,
        commit_message=commit_message or f"Publish Arena run {package.run_id}",
    )
    return {
        "repo_id": resolved_repo,
        "run_id": package.run_id,
        "package_dir": str(package.package_dir),
        "files": package.files,
        "upload": str(upload_result),
    }


__all__ = [
    "HFPackageResult",
    "HFPublishError",
    "ALL_MATCHES_FILE",
    "SCHEMA_VERSION",
    "package_run_for_hf",
    "publish_run_to_hf",
]
