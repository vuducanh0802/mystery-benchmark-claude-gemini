"""Offline regression checks. The HTTP server below never invokes a model."""
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts import run_gpt_oss_kimi_baselines as run
from scripts.prepare_paired_benchmark import prepare
from scripts import run_claude_gemini_baselines as core


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("GPT_OSS", "KIMI"):
        for suffix in ("MODEL", "BASE_URL", "API_KEY", "EXTRA_BODY", "TEMPERATURE", "REVISION"):
            monkeypatch.delenv(f"{name}_{suffix}", raising=False)


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    directory = tmp_path_factory.mktemp("suite")
    prepare(directory)
    return directory


@pytest.fixture
def fake_endpoint():
    requests = []
    mode = {"status": 200, "zero_tokens": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            status = mode["status"]
            if status != 200:
                response = {"error": {"message": "Incorrect API key: fake-secret-test-key", "type": "invalid_request_error", "code": "invalid_api_key"}}
            else:
                response = {
                    "id": "offline-test", "object": "chat.completion", "created": 0,
                    "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {
                        "role": "assistant", "content": json.dumps({
                            "reasoning": "Synthetic test action, not model output.",
                            "beliefs": {}, "action": "ACCUSE", "action_args": {
                                "suspect_name": "Nobody", "weapon_name": "Nothing", "location_name": "Nowhere",
                            },
                        }),
                    }}],
                    "usage": {"prompt_tokens": 0 if mode["zero_tokens"] else 100,
                              "completion_tokens": 0 if mode["zero_tokens"] else 20,
                              "total_tokens": 0 if mode["zero_tokens"] else 120},
                }
            raw = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", requests, mode
    server.shutdown()
    server.server_close()
    thread.join()


def cli(suite, output, url):
    return ["--benchmark-dir", str(suite), "--output-dir", str(output),
            "--levels", "TRIVIAL", "--per-level", "1",
            "--gpt-oss-base-url", url, "--kimi-base-url", url,
            "--max-retries", "0"]


def test_frozen_suite_rejects_mutation(tmp_path):
    prepare(tmp_path)
    manifest = tmp_path / "manifest.json"
    original = manifest.read_bytes()
    manifest.write_text("[]")
    with pytest.raises(ValueError, match="Existing suite differs"):
        prepare(tmp_path)
    assert manifest.read_text() == "[]"
    manifest.write_bytes(original)
    prepare(tmp_path)


def test_explicit_suite_never_falls_back_to_repo_copy(tmp_path):
    # A same-named file in ROOT must not override the explicitly selected suite.
    directory = tmp_path / "level_1"
    directory.mkdir()
    instance = directory / "instance_10042.json"
    instance.write_text('{"source":"explicit"}')
    resolved = core._resolve_instance_path(tmp_path, "data/benchmark_v1/level_1/instance_10042.json", 1)
    assert resolved == instance.resolve()


def test_validate_is_offline(clean_env, suite, tmp_path, fake_endpoint):
    url, calls, _ = fake_endpoint
    assert run.main(cli(suite, tmp_path / "out", url) + ["--validate-only"]) == 0
    assert not calls
    assert not (tmp_path / "out").exists()


def test_paired_http_run_resume_and_reporting(clean_env, monkeypatch, suite, tmp_path, fake_endpoint):
    url, calls, _ = fake_endpoint
    monkeypatch.setenv("GPT_OSS_EXTRA_BODY", '{"reasoning_effort":"medium"}')
    monkeypatch.setenv("KIMI_EXTRA_BODY", '{"chat_template_kwargs":{"enable_thinking":false}}')
    monkeypatch.setenv("KIMI_TEMPERATURE", "0.6")
    output = tmp_path / "out"
    argv = cli(suite, output, url)
    assert run.main(argv) == 0
    config = json.loads((output / "run_config.json").read_text())
    validation = json.loads((output / "validation.json").read_text())
    assert validation["valid"] == validation["expected"] == 4
    assert config["jobs"] == 4
    assert all(c["reasoning_effort"] == "medium" for c in calls if c["model"].startswith("openai/"))
    assert all(c["temperature"] == .6 for c in calls if c["model"].startswith("moonshotai/"))
    assert all(c["chat_template_kwargs"] == {"enable_thinking": False} for c in calls if c["model"].startswith("moonshotai/"))
    before = len(calls)
    assert run.main(argv) == 0
    assert run.main(argv + ["--report-only"]) == 0
    assert len(calls) == before
    for path in output.glob("trajectories/**/*.jsonl"):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if records[0]["detective_policy"] == "guarded":
            assert any(record.get("guard_intervention") for record in records)
    monkeypatch.setenv("GPT_OSS_EXTRA_BODY", '{"reasoning_effort":"high"}')
    with pytest.raises(ValueError, match="different configuration"):
        run.main(argv)
    assert len(calls) == before
    assert json.loads((output / "run_config.json").read_text()) == config


def test_preflight_calls_each_endpoint_once(clean_env, suite, tmp_path, fake_endpoint):
    url, calls, _ = fake_endpoint
    assert run.main(cli(suite, tmp_path / "out", url) + ["--preflight-only"]) == 0
    assert len(calls) == 2
    assert not (tmp_path / "out").exists()


def test_auth_failure_is_not_scored_and_key_is_redacted(clean_env, monkeypatch, suite, tmp_path, fake_endpoint):
    url, calls, mode = fake_endpoint
    mode["status"] = 401
    monkeypatch.setenv("GPT_OSS_API_KEY", "fake-secret-test-key")
    output = tmp_path / "out"
    assert run.main(cli(suite, output, url) + ["--models", "gpt-oss"]) == 1
    assert len(calls) == 1  # stop the other policy after the first fatal error
    assert json.loads((output / "validation.json").read_text())["valid"] == 0
    for path in output.rglob("*"):
        if path.is_file():
            assert "fake-secret-test-key" not in path.read_text()


def test_zero_token_response_is_invalid(clean_env, suite, tmp_path, fake_endpoint):
    url, _, mode = fake_endpoint
    mode["zero_tokens"] = True
    output = tmp_path / "out"
    assert run.main(cli(suite, output, url) + ["--models", "kimi"]) == 1
    assert json.loads((output / "validation.json").read_text())["valid"] == 0


def test_remote_key_is_explicit(clean_env):
    args = run.parse_args(["--models", "kimi", "--kimi-base-url", "https://example.com/v1"])
    with pytest.raises(ValueError, match="KIMI_API_KEY"):
        run.check_credentials(run.endpoints_from_args(args))


def test_body_cannot_override_prompts(clean_env, monkeypatch):
    monkeypatch.setenv("GPT_OSS_EXTRA_BODY", '{"messages":[]}')
    with pytest.raises(ValueError, match="decoding controls"):
        run.endpoints_from_args(run.parse_args(["--models", "gpt-oss"]))


def test_key_not_in_metadata_or_fingerprint(clean_env, monkeypatch):
    args = run.parse_args(["--models", "gpt-oss"])
    monkeypatch.setenv("GPT_OSS_API_KEY", "secret-one")
    first = run.endpoints_from_args(args)["gpt-oss"].metadata()
    monkeypatch.setenv("GPT_OSS_API_KEY", "secret-two")
    second = run.endpoints_from_args(args)["gpt-oss"].metadata()
    assert first == second
    assert "secret" not in json.dumps(second)
