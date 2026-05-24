"""Local TUI client for playing and submitting MysteryArena sessions."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Literal

import requests
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table


Role = Literal["detective", "culprit"]
LEVEL_CHOICES = ["TRIVIAL", "EASY", "MEDIUM", "HARD", "EXPERT"]
DEFAULT_API_URL = os.environ.get("ARENA_API_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_SPACE_URL = os.environ.get("ARENA_SPACE_URL", "https://elfsong-mystery-arena.hf.space")


console = Console()


class ClientError(RuntimeError):
    pass


class ArenaClient:
    def __init__(self, api_url: str) -> None:
        self.api_url = api_url.rstrip("/")

    def request(self, method: str, path: str, *, json_body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.api_url}{path}"
        response = requests.request(method, url, json=json_body, timeout=90)
        try:
            payload = response.json()
        except ValueError:
            payload = {"text": response.text[:500]}
        if response.status_code >= 400:
            raise ClientError(f"{method} {path} failed: {response.status_code} {payload}")
        return payload

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/api/health")

    def models(self) -> dict[str, Any]:
        return self.request("GET", "/api/models")

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/api/sessions", json_body=payload)

    def step_session(
        self,
        session_id: str,
        *,
        action: str,
        action_args: dict[str, Any],
        role: Role | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"action": action, "action_args": action_args}
        if role:
            body["role"] = role
        return self.request("POST", f"/api/sessions/{session_id}/actions", json_body=body)

    def commit_session(
        self,
        session_id: str,
        *,
        run_id: str | None,
        match_id: str | None,
        publish_hf: bool,
        repo_id: str | None,
        include_model_responses: bool,
    ) -> dict[str, Any]:
        return self.request(
            "POST",
            f"/api/sessions/{session_id}/commit",
            json_body={
                "run_id": run_id,
                "match_id": match_id,
                "publish_hf": publish_hf,
                "repo_id": repo_id,
                "include_model_responses": include_model_responses,
            },
        )

    def start_match(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/api/arena/matches", json_body=payload)

    def job(self, job_id: str) -> dict[str, Any]:
        return self.request("GET", f"/api/arena/jobs/{job_id}")

    def publish_run(self, run_id: str, *, repo_id: str | None, include_model_responses: bool) -> dict[str, Any]:
        return self.request(
            "POST",
            f"/api/arena/runs/{run_id}/publish-hf",
            json_body={
                "repo_id": repo_id,
                "include_model_responses": include_model_responses,
            },
        )


@dataclass
class ModelEndpoint:
    name: str
    model: str
    base_url: str
    api_key_env: str

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ClientError(f"Missing local environment variable: {self.api_key_env}")
        return key


def _panel(text: str, title: str, style: str = "cyan") -> Panel:
    return Panel(text.strip(), title=title, border_style=style, box=box.ROUNDED)


def _print_banner(api_url: str) -> None:
    table = Table.grid(expand=True)
    table.add_column(ratio=2)
    table.add_column(justify="right")
    table.add_row("[bold]MysteryArena Local Client[/bold]", f"[dim]{api_url}[/dim]")
    console.print(Panel(table, border_style="green", box=box.ROUNDED))


def _print_session(session: dict[str, Any]) -> None:
    meta = Table(box=box.SIMPLE_HEAVY)
    meta.add_column("Field", style="dim")
    meta.add_column("Value")
    meta.add_row("session", str(session.get("session_id", "-")))
    meta.add_row("role", str(session.get("player_role", "-")))
    meta.add_row("level", str(session.get("level", "-")))
    meta.add_row("seed", str(session.get("seed", "-")))
    meta.add_row("detective", str((session.get("detective") or {}).get("name", "-")))
    meta.add_row("culprit", str((session.get("culprit") or {}).get("name", "-")))
    meta.add_row("detective budget", str(session.get("budget_remaining", "-")))
    meta.add_row("culprit budget", str(session.get("culprit_budget_remaining", "-")))
    console.print(meta)


def _print_events(session: dict[str, Any]) -> None:
    for event in session.get("new_events", []):
        role = event.get("role", "-")
        action = event.get("action", "-")
        success = "ok" if event.get("success", True) else "failed"
        title = f"{role} / {action} / {success}"
        console.print(_panel(str(event.get("result_observation", "")), title, "green" if success == "ok" else "red"))


def _print_result(session: dict[str, Any]) -> None:
    result = session.get("result") or {}
    table = Table(title="Final Result", box=box.ROUNDED)
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    table.add_row("solved", str(result.get("solved")))
    table.add_row("detective_payoff", str(result.get("detective_payoff")))
    table.add_row("culprit_payoff", str(result.get("culprit_payoff")))
    console.print(table)
    summary = (result.get("summary") or {}).get("score_result") or result.get("metrics") or result.get("summary")
    if summary:
        console.print(_panel(json.dumps(summary, indent=2, ensure_ascii=False), "Score", "blue"))


def _help_text() -> str:
    return """
Commands:
  look                         examine current location
  move <location>              move to an adjacent location
  examine <object>             inspect an object
  talk <character>             ask a character one question
  take <object>                pick up a portable object
  inventory                    show collected evidence / inventory
  wait                         pass one step
  accuse                       make final accusation
  json ACTION {"key": "value"}  send raw action JSON
  help                         show this help
  quit                         exit without committing
"""


def _parse_json_command(raw: str) -> tuple[str, dict[str, Any]] | None:
    if not raw.lower().startswith("json "):
        return None
    rest = raw[5:].strip()
    if not rest:
        raise ClientError("Usage: json ACTION {\"key\": \"value\"}")
    if " " not in rest:
        return rest.upper(), {}
    action, args = rest.split(" ", 1)
    return action.strip().upper(), json.loads(args)


def _parse_human_command(raw: str) -> tuple[str, dict[str, Any]] | None:
    raw = raw.strip()
    low = raw.lower()
    parsed_json = _parse_json_command(raw)
    if parsed_json is not None:
        return parsed_json
    if low in {"look", "l", "examine location"}:
        return "EXAMINE_LOCATION", {}
    if low.startswith("move ") or low.startswith("go "):
        target = raw.split(" ", 1)[1].strip()
        return "MOVE", {"target_location": target}
    if low.startswith("examine ") or low.startswith("inspect "):
        obj = raw.split(" ", 1)[1].strip()
        return "EXAMINE_OBJECT", {"object_name": obj}
    if low.startswith("talk ") or low.startswith("ask "):
        name = raw.split(" ", 1)[1].strip()
        question = Prompt.ask(f"Question for {name}", default="Where were you at the time of the murder?")
        return "TALK_TO", {"character_name": name, "question": question}
    if low.startswith("take "):
        obj = raw.split(" ", 1)[1].strip()
        return "TAKE_OBJECT", {"object_name": obj}
    if low in {"inventory", "inv", "evidence"}:
        return "CHECK_INVENTORY", {}
    if low == "wait":
        return "WAIT", {}
    if low == "accuse":
        suspect = Prompt.ask("Suspect name")
        weapon = Prompt.ask("Weapon name")
        location = Prompt.ask("Murder location")
        return "ACCUSE", {
            "suspect_name": suspect,
            "weapon_name": weapon,
            "location_name": location,
        }
    return None


def _commit_finished_session(
    client: ArenaClient,
    session: dict[str, Any],
    *,
    run_id: str | None,
    publish_hf: bool,
    repo_id: str | None,
    include_model_responses: bool,
    space_url: str,
) -> None:
    if not Confirm.ask("Commit trajectory to Arena results database?", default=True):
        return
    result = client.commit_session(
        str(session["session_id"]),
        run_id=run_id,
        match_id=None,
        publish_hf=publish_hf,
        repo_id=repo_id,
        include_model_responses=include_model_responses,
    )
    console.print(_panel(f"run_id: {result['run_id']}\ntrajectory: {result['trajectory_path']}", "Committed", "green"))
    publish_job = result.get("publish_job")
    if publish_job:
        console.print(_panel(f"publish job: {publish_job.get('job_id')}\nstatus: {publish_job.get('status')}", "HF Publish", "cyan"))
        console.print(f"[dim]Viewer Space: {space_url}[/dim]")


def run_human(args: argparse.Namespace) -> None:
    client = ArenaClient(args.api_url)
    _print_banner(args.api_url)
    console.print(_panel("The Hugging Face Space is the public viewer. This client plays through the Arena API and uploads the finished trajectory back to the backend.", "Human Player", "cyan"))

    if args.role == "detective":
        payload = {
            "player_role": "detective",
            "detective": args.name,
            "culprit": args.opponent,
            "level": args.level,
            "seed": args.seed,
        }
    else:
        payload = {
            "player_role": "culprit",
            "detective": args.opponent,
            "culprit": args.name,
            "level": args.level,
            "seed": args.seed,
        }
    session = client.create_session(payload)
    _print_session(session)
    console.print(_panel(str(session.get("briefing", "")), "Briefing", "blue"))
    console.print(_panel(str(session.get("observation", "")), "Observation", "cyan"))

    while not session.get("done"):
        raw = Prompt.ask("[bold]arena[/bold]").strip()
        if raw.lower() in {"quit", "exit", "q"}:
            console.print("[yellow]Exited without committing trajectory.[/yellow]")
            return
        if raw.lower() in {"help", "h", "?"}:
            console.print(_panel(_help_text(), "Help", "cyan"))
            continue
        try:
            parsed = _parse_human_command(raw)
            if not parsed:
                console.print("[red]Unknown command. Type 'help'.[/red]")
                continue
            action, action_args = parsed
            session = client.step_session(
                str(session["session_id"]),
                action=action,
                action_args=action_args,
            )
        except Exception as exc:  # noqa: BLE001 - TUI should keep the user in flow.
            console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
            continue
        _print_events(session)
        if not session.get("done"):
            console.print(_panel(str(session.get("observation", "")), "Observation", "cyan"))

    _print_result(session)
    _commit_finished_session(
        client,
        session,
        run_id=args.run_id,
        publish_hf=args.publish_hf,
        repo_id=args.repo_id,
        include_model_responses=not args.no_model_responses,
        space_url=args.space_url,
    )


def _json_from_model_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _choose_action(endpoint: ModelEndpoint, observation: str, role: Role) -> dict[str, Any]:
    from openai import OpenAI

    system = (
        "You are playing MysteryArena. Return only JSON with this exact shape: "
        "{\"action\": string, \"action_args\": object}. "
        "Valid actions: MOVE, EXAMINE_LOCATION, EXAMINE_OBJECT, TALK_TO, ACCUSE, "
        "WAIT, CHECK_INVENTORY, TAKE_OBJECT. "
        "Common args: MOVE {\"target_location\": \"room\"}; "
        "EXAMINE_OBJECT {\"object_name\": \"object\"}; "
        "TALK_TO {\"character_name\": \"name\", \"question\": \"question\"}; "
        "ACCUSE {\"suspect_name\": \"name\", \"weapon_name\": \"weapon\", "
        "\"location_name\": \"room\"}. "
        f"You are playing the {role} side."
    )
    client = OpenAI(base_url=endpoint.base_url, api_key=endpoint.api_key)
    response = client.chat.completions.create(
        model=endpoint.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": observation},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content or "{}"
    decision = _json_from_model_text(raw)
    action = str(decision.get("action", "")).upper()
    action_args = decision.get("action_args") or {}
    if not action:
        raise ClientError(f"{endpoint.name} returned no action: {raw}")
    if not isinstance(action_args, dict):
        raise ClientError(f"{endpoint.name} returned non-object action_args: {raw}")
    return {"action": action, "action_args": action_args, "raw": raw}


def _model_endpoint_from_args(args: argparse.Namespace, prefix: str, default_env: str) -> ModelEndpoint:
    return ModelEndpoint(
        name=getattr(args, f"{prefix}_name") or getattr(args, f"{prefix}_model"),
        model=getattr(args, f"{prefix}_model"),
        base_url=getattr(args, f"{prefix}_base_url"),
        api_key_env=getattr(args, f"{prefix}_api_key_env") or default_env,
    )


def _print_secret_notice(endpoints: list[ModelEndpoint]) -> None:
    lines = [
        "Model API keys stay on this machine.",
        "This client reads keys from local environment variables and never sends key values to the Arena API.",
        "Only observations, selected actions, and final trajectory records are sent to Arena.",
        "",
        "Local key env vars:",
    ]
    lines.extend(f"  - {endpoint.name}: {endpoint.api_key_env}" for endpoint in endpoints)
    console.print(_panel("\n".join(lines), "API Key Safety", "yellow"))


def run_model_session(args: argparse.Namespace) -> None:
    client = ArenaClient(args.api_url)
    _print_banner(args.api_url)
    detective_endpoint = _model_endpoint_from_args(args, "detective", "MODEL_API_KEY")
    controlled_endpoint = detective_endpoint
    if args.role == "culprit":
        controlled_endpoint = _model_endpoint_from_args(args, "culprit", "CULPRIT_MODEL_API_KEY")
    endpoints = [controlled_endpoint]
    culprit_endpoint: ModelEndpoint | None = None
    if args.role == "both":
        culprit_endpoint = _model_endpoint_from_args(args, "culprit", "CULPRIT_MODEL_API_KEY")
        endpoints = [detective_endpoint, culprit_endpoint]
    _print_secret_notice(endpoints)

    if args.role == "detective":
        payload = {
            "player_role": "detective",
            "detective": detective_endpoint.name,
            "culprit": args.opponent,
            "level": args.level,
            "seed": args.seed,
        }
    elif args.role == "culprit":
        payload = {
            "player_role": "culprit",
            "detective": args.opponent,
            "culprit": controlled_endpoint.name,
            "level": args.level,
            "seed": args.seed,
        }
    else:
        assert culprit_endpoint is not None
        payload = {
            "player_role": "both",
            "detective": detective_endpoint.name,
            "culprit": culprit_endpoint.name,
            "level": args.level,
            "seed": args.seed,
        }

    session = client.create_session(payload)
    _print_session(session)
    console.print(_panel(str(session.get("briefing", ""))[:6000], "Briefing", "blue"))

    turn = 0
    while not session.get("done") and turn < args.max_turns:
        turn += 1
        if args.role == "both":
            observations = session.get("observations") or {}
            for role, endpoint in (("detective", detective_endpoint), ("culprit", culprit_endpoint)):
                if session.get("done"):
                    break
                assert endpoint is not None
                obs = observations.get(role) or session.get("observation") or ""
                decision = _choose_action(endpoint, obs, role)  # type: ignore[arg-type]
                console.print(_panel(
                    json.dumps(
                        {"model": endpoint.name, "role": role, "action": decision["action"], "action_args": decision["action_args"]},
                        indent=2,
                        ensure_ascii=False,
                    ),
                    f"Turn {turn}",
                    "magenta",
                ))
                session = client.step_session(
                    str(session["session_id"]),
                    action=decision["action"],
                    action_args=decision["action_args"],
                    role=role,  # type: ignore[arg-type]
                )
                _print_events(session)
                observations = session.get("observations") or {}
        else:
            role = args.role
            obs = session.get("observation") or ""
            decision = _choose_action(controlled_endpoint, obs, role)
            console.print(_panel(
                json.dumps(
                    {"model": controlled_endpoint.name, "role": role, "action": decision["action"], "action_args": decision["action_args"]},
                    indent=2,
                    ensure_ascii=False,
                ),
                f"Turn {turn}",
                "magenta",
            ))
            session = client.step_session(
                str(session["session_id"]),
                action=decision["action"],
                action_args=decision["action_args"],
            )
            _print_events(session)

    if not session.get("done"):
        raise ClientError(f"Session did not finish within --max-turns={args.max_turns}")

    _print_result(session)
    _commit_finished_session(
        client,
        session,
        run_id=args.run_id,
        publish_hf=args.publish_hf,
        repo_id=args.repo_id,
        include_model_responses=not args.no_model_responses,
        space_url=args.space_url,
    )


def run_registered_match(args: argparse.Namespace) -> None:
    client = ArenaClient(args.api_url)
    _print_banner(args.api_url)
    payload = {
        "detective": args.detective,
        "culprit": args.culprit,
        "level": args.level,
        "seed": args.seed,
        "run_id": args.run_id,
        "resume": not args.no_resume,
        "bootstrap_samples": args.bootstrap_samples,
    }
    job = client.start_match(payload)
    console.print(_panel(json.dumps(job, indent=2, ensure_ascii=False), "Started Match Job", "cyan"))
    job_id = job["job_id"]
    last_status = ""
    while True:
        current = client.job(job_id)
        status = str(current.get("status", "unknown"))
        if status != last_status:
            console.print(f"[bold]job[/bold] {job_id}: {status}")
            last_status = status
        if status in {"succeeded", "failed", "cancelled"}:
            if status != "succeeded":
                raise ClientError(json.dumps(current, indent=2, ensure_ascii=False))
            break
        time.sleep(args.poll_interval)

    console.print(_panel(f"run_id: {current.get('run_id')}\nrun_dir: {current.get('run_dir')}", "Completed", "green"))
    if args.publish_hf:
        publish = client.publish_run(
            str(current["run_id"]),
            repo_id=args.repo_id,
            include_model_responses=not args.no_model_responses,
        )
        console.print(_panel(json.dumps(publish, indent=2, ensure_ascii=False), "Publish Job", "cyan"))
        console.print(f"[dim]Viewer Space: {args.space_url}[/dim]")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Arena backend API URL.")
    parser.add_argument("--space-url", default=DEFAULT_SPACE_URL, help="Viewer Space URL shown after publish.")
    parser.add_argument(
        "--level",
        default="TRIVIAL",
        choices=LEVEL_CHOICES,
        help="Difficulty bucket for the generated mystery.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic case id within the selected level.")
    parser.add_argument("--run-id", default=None, help="Result group name used when committing or publishing.")
    parser.add_argument("--repo-id", default=os.environ.get("ARENA_HF_DATASET"), help="Target Hugging Face Dataset repo.")
    parser.add_argument("--publish-hf", action="store_true", help="Publish committed run to Hugging Face Dataset.")
    parser.add_argument("--no-model-responses", action="store_true", help="Strip raw model responses when publishing.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MysteryArena local TUI client")
    sub = parser.add_subparsers(dest="command")

    human = sub.add_parser("human", help="Play as a human through the Arena API.")
    _add_common(human)
    human.add_argument("--role", choices=["detective", "culprit"], default="detective", help="Side controlled by the human.")
    human.add_argument("--name", default="human", help="Display name recorded for the human player.")
    human.add_argument("--opponent", default="passive", help="Backend registered opponent model.")
    human.set_defaults(func=run_human)

    match = sub.add_parser("match", help="Run backend registered Model A vs Model B.")
    _add_common(match)
    match.add_argument("--detective", required=True, help="Solver-side backend registered model or provider:model reference.")
    match.add_argument("--culprit", required=True, help="Culprit-side backend registered model or provider:model reference.")
    match.add_argument("--poll-interval", type=float, default=2.0)
    match.add_argument("--bootstrap-samples", type=int, default=1000)
    match.add_argument("--no-resume", action="store_true")
    match.set_defaults(func=run_registered_match)

    model = sub.add_parser("model", help="Run local OpenAI-compatible model client for one or both roles.")
    _add_common(model)
    model.add_argument("--role", choices=["detective", "culprit", "both"], default="detective", help="Side controlled by local model client.")
    model.add_argument("--opponent", default="passive", help="Backend registered opponent when --role is detective/culprit.")
    model.add_argument("--detective-name", default=None, help="Display name for the local detective model.")
    model.add_argument("--detective-model", default=os.environ.get("MODEL_NAME", "my-model"), help="Provider model id for the local detective.")
    model.add_argument("--detective-base-url", default=os.environ.get("MODEL_BASE_URL", "http://127.0.0.1:9000/v1"), help="OpenAI-compatible base URL for the local detective.")
    model.add_argument("--detective-api-key-env", default=os.environ.get("MODEL_API_KEY_ENV", "MODEL_API_KEY"), help="Local env var that contains the detective API key.")
    model.add_argument("--culprit-name", default=None, help="Display name for the local culprit model.")
    model.add_argument("--culprit-model", default=os.environ.get("CULPRIT_MODEL_NAME", "my-culprit-model"), help="Provider model id for the local culprit.")
    model.add_argument("--culprit-base-url", default=os.environ.get("CULPRIT_MODEL_BASE_URL", "http://127.0.0.1:9001/v1"), help="OpenAI-compatible base URL for the local culprit.")
    model.add_argument("--culprit-api-key-env", default=os.environ.get("CULPRIT_MODEL_API_KEY_ENV", "CULPRIT_MODEL_API_KEY"), help="Local env var that contains the culprit API key.")
    model.add_argument("--max-turns", type=int, default=80, help="Stop local model play after this many client turns.")
    model.set_defaults(func=run_model_session)

    return parser


def interactive_menu(parser: argparse.ArgumentParser) -> argparse.Namespace:
    console.print(Panel("[bold]MysteryArena[/bold]\nChoose a local client mode.", border_style="green", box=box.ROUNDED))
    mode = Prompt.ask(
        "Mode",
        choices=["human", "match", "model"],
        default="human",
    )
    argv = [mode]
    api_url = Prompt.ask("Arena API URL", default=DEFAULT_API_URL)
    argv.extend(["--api-url", api_url])
    if mode == "human":
        role = Prompt.ask("Your role", choices=["detective", "culprit"], default="detective")
        opponent = Prompt.ask("Opponent model", default="passive" if role == "detective" else "heuristic")
        level = Prompt.ask("Level", choices=LEVEL_CHOICES, default="TRIVIAL")
        seed = str(IntPrompt.ask("Seed", default=0))
        argv.extend(["--role", role, "--opponent", opponent, "--level", level, "--seed", seed])
    elif mode == "match":
        detective = Prompt.ask("Detective model", default="gpt-5.5")
        culprit = Prompt.ask("Culprit model", default=detective)
        level = Prompt.ask("Level", choices=LEVEL_CHOICES, default="TRIVIAL")
        seed = str(IntPrompt.ask("Seed", default=0))
        argv.extend(["--detective", detective, "--culprit", culprit, "--level", level, "--seed", seed])
    else:
        role = Prompt.ask("Model role", choices=["detective", "culprit", "both"], default="detective")
        level = Prompt.ask("Level", choices=LEVEL_CHOICES, default="TRIVIAL")
        seed = str(IntPrompt.ask("Seed", default=0))
        argv.extend(["--role", role, "--level", level, "--seed", seed])
        if role != "both":
            opponent = Prompt.ask("Backend opponent model", default="passive" if role == "detective" else "heuristic")
            argv.extend(["--opponent", opponent])
    if Confirm.ask("Publish to Hugging Face Dataset after completion?", default=False):
        argv.append("--publish-hf")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = interactive_menu(parser)
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI entrypoint.
        console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
