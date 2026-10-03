"""Offline guard and runner checks; the HTTP fixture is a synthetic endpoint."""
import csv
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from agents.guard_ablations import GuardAblationDetectiveAgent
from agents.llm_agent import BIAS_GUARDED_SYSTEM_PROMPT
from mystery_world.entities import EvidenceState
from mystery_world.events import process_culprit_tampering
from mystery_world.world import AgentAction
from scripts import run_gpt_oss_ablations as run
from scripts import run_claude_gemini_baselines as core
from scripts.prepare_paired_benchmark import prepare


ACCUSE_ARGS = {
    "suspect_name": "Alice", "weapon_name": "Knife", "location_name": "Study",
    "suspect_weapon_evidence": ["ev_1"], "weapon_victim_evidence": ["ev_2"],
    "suspect_room_evidence": ["ev_3"],
}


def agent(variant):
    return GuardAblationDetectiveAgent(variant=variant)


def test_full_minus_accuse_keeps_talk_and_object_redirects():
    a = agent("full_minus_accuse")
    a._talk_success = {"Alice": 1, "Bob": 0}
    result = a._guard_action(AgentAction.TALK_TO, {"character_name": "Alice"},
                             "Available targets: TALK_TO: Alice, Bob", 20)
    assert result[1]["character_name"] == "Bob"
    a._object_examines = {"Knife": 1}
    result = a._guard_action(AgentAction.EXAMINE_OBJECT, {"object_name": "Knife"},
                             "Available targets: EXAMINE_OBJECT: Knife, Glass", 20)
    assert result == (AgentAction.EXAMINE_OBJECT, {"object_name": "Glass"})
    assert a._guard_action(AgentAction.ACCUSE, {}, "", 20) == (AgentAction.ACCUSE, {})
    assert a._blocked_accusations == 0


@pytest.mark.parametrize("action,args", [
    (AgentAction.TALK_TO, {"character_name": "Alice", "question": "Again?"}),
    (AgentAction.EXAMINE_OBJECT, {"object_name": "Knife"}),
    (AgentAction.ACCUSE, ACCUSE_ARGS),
])
def test_prompt_ledger_never_rewrites_actions(action, args):
    a = agent("prompt_ledger")
    a._talk_success = {"Alice": 2, "Bob": 0}
    a._object_examines = {"Knife": 2}
    result = a._guard_action(action, args,
                             "Available targets: TALK_TO: Bob | EXAMINE_OBJECT: Glass", 20)
    assert result == (action, args)
    assert result[1] is args
    assert a._blocked_accusations == 0


def test_delay_ignores_evidence_and_blocks_exactly_two_accusations(monkeypatch):
    a = agent("delay_control")
    a._seen_evidence_ids = {"ev_1", "ev_2", "ev_3"}
    monkeypatch.setattr(a, "_weak_accusation_reasons", lambda args: pytest.fail("evidence was inspected"))
    for count in (1, 2):
        action, _ = a._guard_action(AgentAction.ACCUSE, ACCUSE_ARGS, "", 20)
        assert action == AgentAction.EXAMINE_LOCATION
        assert a._blocked_accusations == count
    assert a._guard_action(AgentAction.ACCUSE, ACCUSE_ARGS, "", 20) == (AgentAction.ACCUSE, ACCUSE_ARGS)


@pytest.mark.parametrize("budget", [0, 5])
def test_delay_respects_end_of_budget_escape(budget):
    a = agent("delay_control")
    assert a._guard_action(AgentAction.ACCUSE, {}, "", budget) == (AgentAction.ACCUSE, {})
    assert a._blocked_accusations == 0


def test_delay_does_not_enable_other_redirects():
    a = agent("delay_control")
    a._talk_success = {"Alice": 2, "Bob": 0}
    a._object_examines = {"Knife": 2}
    obs = "Available targets: TALK_TO: Bob | EXAMINE_OBJECT: Glass"
    for action, args in [(AgentAction.TALK_TO, {"character_name": "Alice"}),
                         (AgentAction.EXAMINE_OBJECT, {"object_name": "Knife"})]:
        assert a._guard_action(action, args, obs, 20) == (action, args)


@pytest.mark.parametrize("variant", list(run.VARIANTS))
def test_all_variants_keep_prompt_ledger_and_proposal_trace(variant, monkeypatch):
    a = agent(variant)
    a._env = SimpleNamespace(budget_remaining=20)
    a._seen_evidence_ids = {"ev_1", "ev_2", "ev_3"}
    prompts = []
    def complete(system, user):
        prompts.append((system, user))
        return json.dumps({"action": "ACCUSE", "action_args": ACCUSE_ARGS}), 20
    monkeypatch.setattr(a, "_complete", complete)
    action, _ = a.decide_action("Available targets: EXAMINE_OBJECT: Glass")
    assert prompts[0][0] == BIAS_GUARDED_SYSTEM_PROMPT
    assert "BIAS CONTROL LEDGER" in prompts[0][1]
    assert "ev_1, ev_2, ev_3" in prompts[0][1]
    assert a.last_proposed_action == "ACCUSE"
    assert a.last_proposed_action_args == ACCUSE_ARGS
    if variant == "delay_control":
        assert action == AgentAction.EXAMINE_OBJECT
        assert a.last_guard_intervention["executed_action"] == "EXAMINE_OBJECT"
    else:
        assert action == AgentAction.ACCUSE
        assert a.last_guard_intervention is None


@pytest.fixture
def clean_env(monkeypatch):
    for suffix in ("MODEL", "BASE_URL", "API_KEY", "EXTRA_BODY", "TEMPERATURE", "REVISION"):
        monkeypatch.delenv(f"GPT_OSS_{suffix}", raising=False)


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ablation-suite")
    prepare(directory)
    return directory


@pytest.fixture
def fake_endpoint():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            content = json.dumps({"action": "ACCUSE", "action_args": ACCUSE_ARGS})
            data = json.dumps({
                "id": "offline", "object": "chat.completion", "created": 0,
                "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", requests
    server.shutdown()
    server.server_close()
    thread.join()


def cli(suite, output, url):
    return ["--benchmark-dir", str(suite), "--output-dir", str(output),
            "--gpt-oss-base-url", url, "--levels", "TRIVIAL", "--per-level", "1",
            "--max-retries", "0", "--retry-rounds", "0"]


def test_full_matrix_validation_is_offline(clean_env, suite, tmp_path, capsys):
    assert run.main(["--benchmark-dir", str(suite), "--output-dir", str(tmp_path / "out"),
                     "--gpt-oss-base-url", "https://example.com/v1", "--validate-only"]) == 0
    assert "3000 jobs" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_three_variant_run_resume_and_config_isolation(clean_env, suite, tmp_path, fake_endpoint):
    url, calls = fake_endpoint
    output = tmp_path / "out"
    argv = cli(suite, output, url)
    assert run.main(argv) == 0
    assert len(calls) == 5  # one + three delayed proposals + one
    assert all(r["model"] == "gpt-oss-120b" and r["max_tokens"] == 8192 for r in calls)
    assert all(r["temperature"] == 0 and r["reasoning_effort"] == "low" for r in calls)
    assert all(r["response_format"] == {"type": "json_object"} for r in calls)
    config = json.loads((output / "run_config.json").read_text())
    assert set(config["variants"]) == set(run.VARIANTS)
    assert config["variants"]["full_minus_accuse"]["redirect_talk"]
    assert config["variants"]["delay_control"]["accusation_mode"] == "fixed_delay"
    assert config["variants"]["prompt_ledger"]["accusation_mode"] == "off"
    assert json.loads((output / "validation.json").read_text())["valid"] == 3
    episodes = list(csv.DictReader((output / "episode_metrics.csv").open()))
    assert len(episodes) == 3 and all(r["source_sha256"] for r in episodes)
    delayed = next(r for r in episodes if r["variant"] == "delay_control")
    assert delayed["guard_interventions"] == "2"
    assert run.main(argv) == 0
    assert run.main(argv + ["--report-only"]) == 0
    assert len(calls) == 5
    for change in (["--max-accuse-blocks", "3"], ["--variants", "prompt_ledger"],
                   ["--reasoning-effort", "high"], ["--temperature", "1"]):
        with pytest.raises(ValueError, match="different configuration"):
            run.main(argv + change)
    assert len(calls) == 5
    assert json.loads((output / "run_config.json").read_text()) == config


def test_reports_count_missing_episode_as_failure(clean_env, suite, tmp_path, fake_endpoint):
    url, calls = fake_endpoint
    output = tmp_path / "out"
    argv = cli(suite, output, url)
    assert run.main(argv) == 0
    path = next(output.glob("trajectories/*/prompt_ledger/**/*.jsonl"))
    path.unlink()
    assert run.main(argv + ["--report-only"]) == 1
    summary = list(csv.DictReader((output / "ablation_summary.csv").open()))
    row = next(r for r in summary if r["variant"] == "prompt_ledger" and r["level"] == "OVERALL")
    assert row["expected_n"] == "1" and row["invalid_n"] == "1"
    assert float(row["solve_rate"]) == 0
    assert len(calls) == 5


def test_mutated_case_is_rejected_before_any_endpoint_call(clean_env, suite, tmp_path, fake_endpoint):
    url, calls = fake_endpoint
    cases = core.load_cases(suite, {1}, 1)
    config_args = run.parse_args(cli(suite, tmp_path / "out", url))
    config_args.endpoints = {"gpt-oss": run.endpoint_from_args(config_args)}
    with pytest.raises(ValueError, match="Case differs"):
        run.build_config(config_args, [replace(cases[0], sha256="wrong")])
    assert not calls


def test_full_suite_matches_published_case_count_and_world_duplicates(suite):
    cases = core.load_cases(suite, {1, 2, 3, 4, 5}, 200)
    assert len(cases) == 1000
    assert len({c.sha256 for c in cases}) == 874
    expected = run.reference_case_hashes()
    assert all(expected[c.instance_id] == c.sha256 for c in cases)


def test_unknown_action_does_not_block_other_episodes():
    assert not core._is_fatal_provider_error("ValueError: unknown model action 'NONE'")
    assert core._is_fatal_provider_error("404: model not found")


class MoveRNG:
    def __init__(self):
        self.draws = []
    def random(self):
        self.draws.append("random")
        return 0
    def choice(self, values):
        self.draws.append(list(values))
        return "move" if "move" in values else values[-1]


def tamper_state(old_location):
    evidence = SimpleNamespace(id="ev_1", name="Trace", linked_character_id="culprit",
                               location_id=old_location, state=EvidenceState.PRISTINE)
    return SimpleNamespace(
        config=SimpleNamespace(free_culprit_actions=False, reactive_events=False, culprit_tamper_prob=1),
        characters={"culprit": SimpleNamespace(id="culprit", is_culprit=True, is_alive=True)},
        evidence={"ev_1": evidence}, current_step=1,
        locations={"hall": SimpleNamespace(name="Hall", objects_here=["ev_1"]),
                   "study": SimpleNamespace(name="Study", objects_here=[])},
    )


def test_inventory_relocation_patch_preserves_rng_and_evidence():
    state = tamper_state("inventory:detective")
    rng = MoveRNG()
    assert process_culprit_tampering(state, rng) == []
    assert len(rng.draws) == 3
    assert state.evidence["ev_1"].location_id == "inventory:detective"
    assert state.evidence["ev_1"].state == EvidenceState.PRISTINE


def test_inventory_patch_keeps_normal_moves_and_other_invalid_rooms_fail():
    state = tamper_state("hall")
    events = process_culprit_tampering(state, MoveRNG())
    assert len(events) == 1 and state.evidence["ev_1"].location_id == "study"
    assert "ev_1" not in state.locations["hall"].objects_here
    assert "ev_1" in state.locations["study"].objects_here
    with pytest.raises(KeyError):
        process_culprit_tampering(tamper_state("invalid-room"), MoveRNG())
