import json
from pathlib import Path

import pytest

from agents.base_agent import LLMConfig, LLMUnavailable
from agents.llm_agent import BiasGuardedLLMDetectiveAgent, LLMDetectiveAgent
from mystery_world.world import AgentAction
from scripts.run_claude_gemini_baselines import (
    BenchmarkCase,
    Job,
    ModelSpec,
    build_jobs,
    load_cases,
    validate_trajectory,
)


def test_manifest_matrix_is_paired_and_collision_free(tmp_path: Path):
    manifest = []
    for level in (1, 2):
        level_dir = tmp_path / f"level_{level}"
        level_dir.mkdir()
        for index in range(2):
            instance = level_dir / f"instance_{index}.json"
            instance.write_text("{}", encoding="utf-8")
            manifest.append({
                "level": level,
                "seed": level * 100 + index,
                "instance_file": str(instance.relative_to(tmp_path)),
            })
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    cases = load_cases(tmp_path, {1, 2}, per_level=2)
    models = [
        ModelSpec("claude", "anthropic", "claude-sonnet-4-6"),
        ModelSpec("gemini", "google", "gemini-3.6-flash"),
    ]
    jobs = build_jobs(cases, models, ["vanilla", "guarded"])

    assert len(cases) == 4
    assert len(jobs) == 16
    assert len({job.job_id for job in jobs}) == 16
    for case in cases:
        paired = [job for job in jobs if job.case == case]
        assert {(job.model.name, job.policy) for job in paired} == {
            ("claude", "vanilla"),
            ("claude", "guarded"),
            ("gemini", "vanilla"),
            ("gemini", "guarded"),
        }


def _job(tmp_path: Path) -> Job:
    instance = tmp_path / "instance.json"
    instance.write_text("{}", encoding="utf-8")
    case = BenchmarkCase(
        level=4,
        ordinal=0,
        benchmark_seed=40042,
        path=instance,
        source_label="data/benchmark_v1/level_4/instance.json",
        sha256="abc123",
    )
    return Job(
        model=ModelSpec("claude", "anthropic", "claude-sonnet-4-6"),
        policy="guarded",
        case=case,
    )


def _valid_records(job: Job) -> list[dict]:
    return [
        {
            "kind": "header",
            "schema_version": 2,
            "detective_agent": job.cell_id,
            "detective_model": job.model.model,
            "detective_provider": job.model.provider,
            "detective_policy": job.policy,
            "benchmark_seed": job.case.benchmark_seed,
            "instance_id": job.case.instance_id,
            "source_instance_sha256": job.case.sha256,
        },
        {
            "kind": "step",
            "model_called": True,
            "model_response": '{"action":"EXAMINE_LOCATION"}',
            "input_tokens": 100,
            "output_tokens": 20,
        },
        {
            "kind": "footer",
            "error": None,
            "episode_summary": {},
            "metrics": {"total_tokens": 120},
        },
    ]


def test_resume_requires_successful_nonzero_token_trajectory(tmp_path: Path):
    job = _job(tmp_path)
    path = tmp_path / "trajectory.jsonl"
    records = _valid_records(job)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    assert validate_trajectory(path, job)[0]

    records[1]["input_tokens"] = 0
    records[1]["output_tokens"] = 0
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    valid, reason, _ = validate_trajectory(path, job)
    assert not valid
    assert "zero token" in reason


def test_resume_rejects_error_footer(tmp_path: Path):
    job = _job(tmp_path)
    path = tmp_path / "trajectory.jsonl"
    records = _valid_records(job)
    records[-1]["error"] = "AuthenticationError: 401"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    valid, reason, _ = validate_trajectory(path, job)
    assert not valid
    assert "terminal error" in reason


def test_google_key_accepts_either_environment_name(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    config = LLMConfig(provider="google", model="gemini-3.6-flash")
    assert config.resolved_api_key() == "test-google-key"

    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    assert config.resolved_api_key() == "test-gemini-key"


def test_cloud_provider_missing_key_fails_loud(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable):
        LLMConfig(provider="anthropic").resolved_api_key()


@pytest.mark.parametrize("text", ["", "not json", '{"action":'])
def test_vanilla_malformed_output_is_an_error(text):
    with pytest.raises(ValueError):
        LLMDetectiveAgent()._parse_response(text)


def test_guard_balances_interview_exposure():
    agent = BiasGuardedLLMDetectiveAgent()
    agent._talk_success = {"Alice": 1, "Bob": 0}
    observation = "Available targets: TALK_TO: Alice, Bob | MOVE: Hall"
    action, args = agent._guard_action(
        AgentAction.TALK_TO,
        {"character_name": "Alice", "question": "Again?"},
        observation,
        budget=20,
    )
    assert action == AgentAction.TALK_TO
    assert args["character_name"] == "Bob"


def test_guard_redirects_repeated_object_exposure():
    agent = BiasGuardedLLMDetectiveAgent()
    agent._object_examines = {"Knife": 1, "Glass": 0}
    observation = "Available targets: EXAMINE_OBJECT: Knife, Glass | MOVE: Hall"
    action, args = agent._guard_action(
        AgentAction.EXAMINE_OBJECT,
        {"object_name": "Knife"},
        observation,
        budget=20,
    )
    assert action == AgentAction.EXAMINE_OBJECT
    assert args == {"object_name": "Glass"}


def test_guard_blocks_weak_exposure_based_accusation():
    agent = BiasGuardedLLMDetectiveAgent()
    agent._suspect_names = {"char_1": "Alice", "char_2": "Bob"}
    agent._first_talk_order = ["Alice"]
    agent._talk_success = {"Alice": 2, "Bob": 0}
    observation = "Available targets: TALK_TO: Bob | MOVE: Hall"
    action, args = agent._guard_action(
        AgentAction.ACCUSE,
        {
            "suspect_name": "Alice",
            "weapon_name": "Knife",
            "location_name": "Study",
            "suspect_weapon_evidence": [],
            "weapon_victim_evidence": [],
            "suspect_room_evidence": [],
        },
        observation,
        budget=20,
    )
    assert action == AgentAction.TALK_TO
    assert args["character_name"] == "Bob"
    assert agent._blocked_accusations == 1


def test_guard_accepts_only_observed_triangle_evidence_ids():
    agent = BiasGuardedLLMDetectiveAgent()
    accusation = {
        "suspect_weapon_evidence": ["ev_1"],
        "weapon_victim_evidence": ["ev_2"],
        "suspect_room_evidence": ["ev_3"],
    }
    assert agent._weak_accusation_reasons(accusation)

    agent._seen_evidence_ids = {"ev_1", "ev_2", "ev_3"}
    assert agent._weak_accusation_reasons(accusation) == []
