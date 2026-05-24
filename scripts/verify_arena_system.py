"""Verify the Arena API -> result -> Hugging Face package flow."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from arena.api import create_app
from arena.hf_publish import package_run_for_hf


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _wait_for_job(client: TestClient, job_id: str, *, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/arena/jobs/{job_id}")
        response.raise_for_status()
        last = response.json()
        if last["status"] in {"succeeded", "failed", "cancelled"}:
            return last
        time.sleep(0.25)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s; last={last}")


def _assert_no_sensitive_text(payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, default=str).lower()
    forbidden = [
        "https://",
        "http://",
        "llm_gateway_api_key",
        "openai_api_key",
        "openrouter_api_key",
        "hf_token",
        "bearer ",
        "secret.example",
        "sk-",
        "llmbox",
    ]
    leaks = [item for item in forbidden if item in text]
    if leaks:
        raise AssertionError(f"sensitive marker(s) leaked into public package: {leaks}")


def _verify_frontend(package_dir: Path) -> dict[str, Any]:
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError as exc:
        raise AssertionError(
            "streamlit is required for --frontend; run with "
            "`uv run --with streamlit --with pandas --with plotly --with requests ...`"
        ) from exc

    handler = partial(SimpleHTTPRequestHandler, directory=str(package_dir))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    old_env = {
        key: os.environ.get(key)
        for key in ("ARENA_DATASET_BASE_URL", "ARENA_DATASET_REPO", "ARENA_DEFAULT_REVISION")
    }
    try:
        os.environ["ARENA_DATASET_BASE_URL"] = base_url
        os.environ["ARENA_DATASET_REPO"] = "local/mystery-arena-results"
        os.environ["ARENA_DEFAULT_REVISION"] = "main"
        app_path = Path(__file__).resolve().parent.parent / "apps" / "arena_streamlit_space" / "app.py"
        app = AppTest.from_file(str(app_path))
        app.run(timeout=30)
        if app.exception:
            raise AssertionError(f"frontend raised exceptions: {app.exception}")
        tabs = [tab.label for tab in app.tabs]
        expected_tabs = {"Overview", "Leaderboards", "Duel Matrix", "Episode Replay", "API Docs"}
        if not expected_tabs <= set(tabs):
            raise AssertionError(f"missing frontend tabs: expected={expected_tabs} got={tabs}")
        source_inputs = {
            str(getattr(text_input, "label", ""))
            for text_input in app.text_input
        }
        forbidden_source_inputs = {"Dataset repo", "Revision", "Base URL"}
        if source_inputs & forbidden_source_inputs:
            raise AssertionError(f"source controls should not be visible: {source_inputs}")
        button_labels = {str(getattr(button, "label", "")) for button in app.button}
        if "Refresh cache" in button_labels:
            raise AssertionError("refresh cache control should not be visible")
        if len(app.metric) < 7:
            raise AssertionError(f"expected at least 7 metrics, got {len(app.metric)}")
        step_buttons = [
            button
            for button in app.button
            if str(getattr(button, "label", "")).startswith("#")
        ]
        if not step_buttons:
            raise AssertionError("expected clickable replay step buttons")
        step_labels = [str(getattr(button, "label", "")) for button in step_buttons]
        if any(label.endswith(" ok") or label.endswith(" failed") for label in step_labels):
            raise AssertionError(f"replay step labels should not include status text: {step_labels[:3]}")
        if len(step_buttons) > 1:
            step_buttons[1].click().run(timeout=30)
            if app.exception:
                raise AssertionError(f"frontend raised exceptions after step click: {app.exception}")
        return {
            "base_url": base_url,
            "tabs": tabs,
            "metrics": len(app.metric),
            "step_buttons": len(step_buttons),
        }
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def verify(
    *,
    arena_root: Path,
    package_dir: Path,
    timeout: float,
    frontend: bool = False,
) -> dict[str, Any]:
    if arena_root.exists():
        shutil.rmtree(arena_root)
    if package_dir.exists():
        shutil.rmtree(package_dir)

    client = TestClient(create_app(arena_root=arena_root, env_file=None))
    response = client.post(
        "/api/arena/matches",
        json={
            "detective": "heuristic",
            "culprit": "passive",
            "level": "TRIVIAL",
            "seed": 0,
            "run_id": "verify_api_match",
            "bootstrap_samples": 10,
        },
    )
    response.raise_for_status()
    created = response.json()
    job = _wait_for_job(client, created["job_id"], timeout=timeout)
    if job["status"] != "succeeded":
        raise AssertionError(f"arena job failed: {job}")

    run_response = client.get("/api/runs/verify_api_match")
    run_response.raise_for_status()
    run = run_response.json()
    summary = run.get("outputs", {}).get("summary", {})
    if summary.get("matches") != 1:
        raise AssertionError(f"expected 1 match, got {summary}")

    run_dir = arena_root / "verify_api_match"
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["llm_gateway"] = {
        "url": "https://llmbox.secret.example/v1",
        "key_env": "LLM_GATEWAY_API_KEY",
        "model_override": None,
    }
    config["publish_probe"] = {
        "callback_url": "https://secret.example/callback",
        "token": "sk-testsecret123456",
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    package = package_run_for_hf(
        run_dir,
        package_dir=package_dir,
        include_model_responses=False,
    )
    index = json.loads((package.package_dir / "index/runs.json").read_text(encoding="utf-8"))
    public_summary = json.loads(
        (package.package_dir / "runs/verify_api_match/summary.json").read_text(encoding="utf-8")
    )
    matches = _read_jsonl_gz(package.package_dir / "runs/verify_api_match/matches.jsonl.gz")
    all_matches = _read_jsonl_gz(package.package_dir / "matches/all_matches.jsonl.gz")
    if len(index.get("runs", [])) != 1:
        raise AssertionError(f"expected 1 indexed run, got {index}")
    if index.get("latest_run_id") != "verify_api_match":
        raise AssertionError(f"latest_run_id mismatch: {index}")
    if index.get("matches_file") != "matches/all_matches.jsonl.gz":
        raise AssertionError(f"unified matches_file mismatch: {index}")
    readme = (package.package_dir / "README.md").read_text(encoding="utf-8")
    if "split: matches" not in readme or "split: verify_api_match" in readme:
        raise AssertionError("README should expose exactly the unified matches split")
    for rel in package.files:
        if not (package.package_dir / rel).exists():
            raise AssertionError(f"packaged file is missing: {rel}")
    if len(matches) != 1:
        raise AssertionError(f"expected 1 packaged match, got {len(matches)}")
    if len(all_matches) != 1:
        raise AssertionError(f"expected 1 unified packaged match, got {len(all_matches)}")
    if "trajectory_path" in matches[0]:
        raise AssertionError("local trajectory_path leaked into packaged match")
    trajectory_file = matches[0].get("trajectory_file")
    if not trajectory_file:
        raise AssertionError("packaged match is missing trajectory_file")
    trajectory = _read_jsonl_gz(package.package_dir / trajectory_file)
    if len(trajectory) < 3:
        raise AssertionError("packaged trajectory is unexpectedly short")
    if any("model_response" in record for record in trajectory):
        raise AssertionError("model_response was not stripped when include_model_responses=False")

    _assert_no_sensitive_text({
        "summary": public_summary,
        "matches": matches,
        "trajectory_head": trajectory[:3],
    })

    result = {
        "arena_root": str(arena_root),
        "package_dir": str(package.package_dir),
        "job_id": created["job_id"],
        "run_id": package.run_id,
        "matches": len(matches),
        "trajectory_records": len(trajectory),
        "files": package.files,
    }
    if frontend:
        result["frontend"] = _verify_frontend(package.package_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Arena API and HF package flow")
    parser.add_argument("--arena-root", default=None)
    parser.add_argument("--package-dir", default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--frontend",
        action="store_true",
        help="Also run the Streamlit frontend against the packaged dataset.",
    )
    args = parser.parse_args()

    temp_root = Path(tempfile.mkdtemp(prefix="arena_verify_"))
    arena_root = Path(args.arena_root) if args.arena_root else temp_root / "results"
    package_dir = Path(args.package_dir) if args.package_dir else temp_root / "hf_package"
    try:
        result = verify(
            arena_root=arena_root,
            package_dir=package_dir,
            timeout=args.timeout,
            frontend=args.frontend,
        )
    except Exception as exc:  # noqa: BLE001 - this is a verifier entrypoint.
        print(f"verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
