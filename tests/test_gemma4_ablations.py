"""Verify Gemma decoding, optional matched control, routing, and resume offline."""
import csv
import json
from dataclasses import replace

import pytest

from scripts import run_gemma4_ablations as gemma
from scripts import run_claude_gemini_baselines as core
from scripts import run_gpt_oss_kimi_baselines as paired
from test_gpt_oss_ablations import suite, fake_endpoint


def cli(suite, output, url):
    return ["--benchmark-dir", str(suite), "--output-dir", str(output),
            "--base-urls", url, "--levels", "TRIVIAL", "--per-level", "1",
            "--max-retries", "0", "--retry-rounds", "0"]


def test_gemma_profile_and_matched_control_run_resume(suite, tmp_path, fake_endpoint, monkeypatch):
    monkeypatch.delenv("GEMMA4_REVISION", raising=False)
    url, calls = fake_endpoint
    output = tmp_path / "gemma"
    argv = cli(suite, output, url) + ["--include-guarded-reference", "--workers", "4"]
    assert gemma.main(argv) == 0
    assert len(calls) == 8  # full three, minus one, delay three, P+L one
    assert all(r["model"] == gemma.MODEL and r["max_tokens"] == 768 for r in calls)
    assert all(r["temperature"] == 0 and "reasoning_effort" not in r for r in calls)
    assert all(r["response_format"]["json_schema"]["strict"] for r in calls)
    assert all(r["response_format"]["json_schema"]["schema"] == gemma.DETECTIVE_SCHEMA for r in calls)
    config = json.loads((output / "run_config.json").read_text())
    assert config["decoding"]["history_window"] == 4
    assert config["reference_release"]["historical_gemma_guarded_is_paired"] is False
    assert config["variants"]["guarded"]["accusation_mode"] == "evidence_gate"
    assert config["replica_base_urls"] == [url]
    assert json.loads((output / "validation.json").read_text())["valid"] == 4
    metrics = list(csv.DictReader((output / "episode_metrics.csv").open()))
    assert len(metrics) == 4
    assert {r["source_sha256"] for r in metrics} == set(config["case_hashes"].values())
    assert gemma.main(argv) == 0
    assert gemma.main(argv + ["--report-only"]) == 0
    assert len(calls) == 8
    with pytest.raises(ValueError, match="different configuration"):
        gemma.main(argv + ["--history-window", "10"])
    assert len(calls) == 8


def test_gemma_full_matrix_offline_with_or_without_control(suite, tmp_path, capsys):
    argv = ["--benchmark-dir", str(suite), "--output-dir", str(tmp_path / "out"),
            "--validate-only", "--base-urls", "https://example.com/v1"]
    assert gemma.main(argv) == 0
    assert "3000 jobs" in capsys.readouterr().out
    assert gemma.main(argv + ["--include-guarded-reference"]) == 0
    assert "4000 jobs" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_secondary_replica_credentials_checked_before_requests(suite, tmp_path, monkeypatch):
    monkeypatch.delenv("GEMMA4_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMMA4_API_KEY"):
        gemma.main(["--benchmark-dir", str(suite), "--output-dir", str(tmp_path),
                    "--base-urls", "http://127.0.0.1:8332/v1", "https://example.com/v1"])
    assert not (tmp_path / "run_config.json").exists()


def test_replica_factory_preserves_guard_and_decoding(suite, tmp_path):
    args = gemma.parse_args(cli(suite, tmp_path, "http://127.0.0.1:8332/v1") + [
        "--base-urls", "http://127.0.0.1:8332/v1", "http://127.0.0.1:8333/v1",
        "--include-guarded-reference",
    ])
    cases = core.load_cases(suite, {1}, 1)
    jobs = core.build_jobs(cases, [args.endpoints["gemma4"].model], args.policies)
    agents = [gemma.make_agent(job, args) for job in jobs]
    assert isinstance(agents[0], paired.Guarded)
    assert [a.config.base_url for a in agents] == args.base_urls * 2
    assert [a.variant for a in agents[1:]] == list(gemma.VARIANTS)
    assert all(a.config.max_tokens == 768 for a in agents)
    config = gemma.build_config(args, cases)
    assert config["replica_base_urls"] == args.base_urls
    with pytest.raises(ValueError, match="Case differs"):
        gemma.build_config(args, [replace(cases[0], sha256="wrong")])
