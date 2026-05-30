"""Gradio dashboard for MysteryArena Arena results."""

from __future__ import annotations

import argparse
import html
import json
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from arena.aggregate import aggregate_matches, load_matches
from arena.metrics import read_jsonl


DETECTIVE_HEADERS = [
    "rank",
    "model",
    "n",
    "mean",
    "CI low",
    "CI high",
    "Skill",
    "Mu",
    "Sigma",
    "solve",
    "accuracy",
    "triangle",
    "alibi",
    "elim",
    "actions",
    "fail rate",
    "guard",
]

CULPRIT_HEADERS = [
    "rank",
    "model",
    "n",
    "mean",
    "CI low",
    "CI high",
    "Skill",
    "Mu",
    "Sigma",
    "detective fail",
    "drop vs passive",
    "det actions",
    "culprit actions",
    "culprit fail",
    "guard",
]


def _load_outputs(arena_dir: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = Path(arena_dir)
    config = {}
    if (root / "config.json").exists():
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    matches = load_matches(root)
    rating = config.get("rating", {})
    outputs = aggregate_matches(
        matches,
        bootstrap_samples=int(rating.get("bootstrap_samples", 1000)),
        trueskill_mu=float(rating.get("trueskill_mu", 25.0)),
        trueskill_sigma=float(rating.get("trueskill_sigma", 25.0 / 3.0)),
        trueskill_beta=float(rating.get("trueskill_beta", 25.0 / 6.0)),
        trueskill_tau=float(rating.get("trueskill_tau", 25.0 / 300.0)),
        trueskill_draw_threshold=float(rating.get("trueskill_draw_threshold", 0.0)),
    ) if matches else {
        "detective_leaderboard": [],
        "culprit_leaderboard": [],
        "ratings": {"system": "trueskill", "detective": {}, "culprit": {}},
        "matrix": {},
        "summary": {"matches": 0, "detectives": 0, "culprits": 0},
    }
    return config, matches, outputs


def _api_json(api_url: str, path: str) -> dict[str, Any]:
    url = api_url.rstrip("/") + path
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API request failed: {exc.reason}") from exc


def _load_outputs_api(api_url: str, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    encoded = urllib.parse.quote(run_id.strip(), safe="")
    payload = _api_json(api_url, f"/api/runs/{encoded}")
    return payload.get("config", {}), payload.get("matches", []), payload.get("outputs", {})


def _default_api_run(api_url: str, fallback: str) -> str:
    try:
        payload = _api_json(api_url, "/api/runs")
    except RuntimeError:
        return fallback
    runs = payload.get("runs", [])
    if not runs:
        return fallback
    return str(runs[0].get("run_id") or fallback)


def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 4)
    return value


def _detective_rows(outputs: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for r in outputs.get("detective_leaderboard", []):
        rating = r.get("trueskill", {})
        rows.append([
            r.get("rank"),
            r.get("model"),
            r.get("n"),
            _fmt(r.get("mean_payoff")),
            _fmt(r.get("ci_low")),
            _fmt(r.get("ci_high")),
            _fmt(rating.get("skill")),
            _fmt(rating.get("mu")),
            _fmt(rating.get("sigma")),
            _fmt(r.get("solve_rate")),
            _fmt(r.get("accusation_accuracy")),
            _fmt(r.get("triangle")),
            _fmt(r.get("alibi")),
            _fmt(r.get("elimination")),
            _fmt(r.get("avg_actions")),
            _fmt(r.get("failed_action_rate")),
            _fmt(r.get("guard_blocked")),
        ])
    return rows


def _culprit_rows(outputs: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for r in outputs.get("culprit_leaderboard", []):
        rating = r.get("trueskill", {})
        rows.append([
            r.get("rank"),
            r.get("model"),
            r.get("n"),
            _fmt(r.get("mean_payoff")),
            _fmt(r.get("ci_low")),
            _fmt(r.get("ci_high")),
            _fmt(rating.get("skill")),
            _fmt(rating.get("mu")),
            _fmt(rating.get("sigma")),
            _fmt(r.get("detective_failure_rate")),
            _fmt(r.get("score_drop_vs_passive")),
            _fmt(r.get("avg_detective_actions")),
            _fmt(r.get("avg_culprit_actions")),
            _fmt(r.get("culprit_failed_action_rate")),
            _fmt(r.get("guard_blocked")),
        ])
    return rows


def _overview_html(config: dict[str, Any], matches: list[dict[str, Any]], outputs: dict[str, Any]) -> str:
    summary = outputs.get("summary", {})
    best_d = (outputs.get("detective_leaderboard") or [{}])[0]
    best_c = (outputs.get("culprit_leaderboard") or [{}])[0]
    npc = config.get("npc", {})
    cards = [
        ("Matches", summary.get("matches", len(matches))),
        ("Detectives", summary.get("detectives", 0)),
        ("Culprits", summary.get("culprits", 0)),
        ("Mode", config.get("mode", "unknown")),
        ("Top Detective", best_d.get("model", "")),
        ("Top Culprit", best_c.get("model", "")),
        ("NPC", npc.get("provider", "fallback")),
    ]
    card_html = "".join(
        f"<div style='padding:12px;border:1px solid #ddd;border-radius:6px;background:#fff;'>"
        f"<div style='font-size:12px;color:#666'>{html.escape(str(label))}</div>"
        f"<div style='font-size:22px;font-weight:650'>{html.escape(str(value))}</div></div>"
        for label, value in cards
    )
    levels = ", ".join(config.get("levels", []))
    seeds = config.get("seeds", [])
    seed_text = f"{len(seeds)} seeds" if isinstance(seeds, list) else str(seeds)
    return (
        "<div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px'>"
        f"{card_html}</div>"
        f"<p><b>Run:</b> {html.escape(str(config.get('run_id', 'unknown')))} "
        f"&nbsp; <b>Levels:</b> {html.escape(levels)} "
        f"&nbsp; <b>Seeds:</b> {html.escape(seed_text)}</p>"
    )


def _matrix_html(outputs: dict[str, Any]) -> str:
    matrix = outputs.get("matrix", {})
    if not matrix:
        return "<p>No duel matrix yet.</p>"
    detectives = sorted(matrix)
    culprits = sorted({c for row in matrix.values() for c in row})
    header = "".join(f"<th>{html.escape(c)}</th>" for c in culprits)
    rows = []
    for d in detectives:
        cells = []
        for c in culprits:
            cell = matrix.get(d, {}).get(c)
            if not cell:
                cells.append("<td style='background:#f5f5f5;color:#999'>-</td>")
                continue
            value = float(cell.get("detective_payoff", 0.0))
            red = int(255 * (1.0 - value))
            green = int(255 * value)
            bg = f"rgb({red},{green},90)"
            color = "#111" if value > 0.55 else "#fff"
            cells.append(
                f"<td title='n={cell.get('n', 0)}' "
                f"style='background:{bg};color:{color};font-weight:650;text-align:center'>"
                f"{value:.3f}</td>"
            )
        rows.append(f"<tr><th>{html.escape(d)}</th>{''.join(cells)}</tr>")
    return (
        "<p>Cell value = detective payoff. Lower values mean stronger culprit pressure.</p>"
        "<table style='border-collapse:collapse'>"
        "<tr><th>Detective \\ Culprit</th>"
        f"{header}</tr>{''.join(rows)}</table>"
    )


def _role_gap_html(outputs: dict[str, Any]) -> str:
    d_rows = {r["model"]: r for r in outputs.get("detective_leaderboard", [])}
    c_rows = {r["model"]: r for r in outputs.get("culprit_leaderboard", [])}
    names = sorted(set(d_rows) & set(c_rows))
    if not names:
        return "<p>No model appears in both roles yet.</p>"
    d_vals = [float(d_rows[n].get("mean_payoff") or 0.0) for n in names]
    c_vals = [float(c_rows[n].get("mean_payoff") or 0.0) for n in names]
    width, height, pad = 620, 420, 50

    def scale_x(x: float) -> float:
        return pad + x * (width - 2 * pad)

    def scale_y(y: float) -> float:
        return height - pad - y * (height - 2 * pad)

    points = []
    labels = []
    for name, x, y in zip(names, d_vals, c_vals, strict=True):
        cx = scale_x(x)
        cy = scale_y(y)
        points.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='6' fill='#2563eb' />")
        labels.append(
            f"<text x='{cx + 8:.1f}' y='{cy - 8:.1f}' font-size='12'>"
            f"{html.escape(name)}</text>"
        )
    grid = []
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = scale_x(tick)
        y = scale_y(tick)
        grid.append(f"<line x1='{x}' x2='{x}' y1='{pad}' y2='{height-pad}' stroke='#eee' />")
        grid.append(f"<line x1='{pad}' x2='{width-pad}' y1='{y}' y2='{y}' stroke='#eee' />")
        grid.append(f"<text x='{x-8}' y='{height-pad+22}' font-size='11'>{tick:.2f}</text>")
        grid.append(f"<text x='8' y='{y+4}' font-size='11'>{tick:.2f}</text>")
    diag = f"<line x1='{scale_x(0)}' y1='{scale_y(0)}' x2='{scale_x(1)}' y2='{scale_y(1)}' stroke='#999' stroke-dasharray='4 4' />"
    return (
        "<p>x = detective mean payoff, y = culprit mean payoff.</p>"
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' "
        "style='background:white;border:1px solid #ddd;border-radius:6px'>"
        f"{''.join(grid)}{diag}"
        f"<text x='{width/2-60}' y='{height-10}' font-size='13'>Detective ability</text>"
        "<text x='10' y='20' font-size='13'>Culprit ability</text>"
        f"{''.join(points)}{''.join(labels)}</svg>"
    )


def _episode_choices(matches: list[dict[str, Any]]) -> list[tuple[str, str]]:
    choices = []
    for match in matches:
        path = match.get("trajectory_path")
        if not path or not Path(path).exists():
            continue
        label = (
            f"{match.get('detective', {}).get('name')} vs "
            f"{match.get('culprit', {}).get('name')} | "
            f"{match.get('level')} seed={match.get('seed')} | "
            f"payoff={float(match.get('detective_payoff', 0.0)):.3f}"
        )
        choices.append((label, path))
    return choices


def _episode_choices_api(matches: list[dict[str, Any]]) -> list[tuple[str, str]]:
    choices = []
    for match in matches:
        episode_id = match.get("episode_id") or match.get("match_id")
        if not episode_id or not match.get("trajectory_available"):
            continue
        label = (
            f"{match.get('detective', {}).get('name')} vs "
            f"{match.get('culprit', {}).get('name')} | "
            f"{match.get('level')} seed={match.get('seed')} | "
            f"payoff={float(match.get('detective_payoff', 0.0)):.3f}"
        )
        choices.append((label, str(episode_id)))
    return choices


def refresh(arena_dir: str):
    config, matches, outputs = _load_outputs(arena_dir)
    choices = _episode_choices(matches)
    first = choices[0][1] if choices else None
    return (
        _overview_html(config, matches, outputs),
        _detective_rows(outputs),
        _culprit_rows(outputs),
        _role_gap_html(outputs),
        _matrix_html(outputs),
        gr.update(choices=choices, value=first),
    )


def _records(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    return read_jsonl(p)


def _records_api(api_url: str, run_id: str, episode_id: str | None) -> list[dict[str, Any]]:
    if not episode_id:
        return []
    run = urllib.parse.quote(run_id.strip(), safe="")
    episode = urllib.parse.quote(str(episode_id), safe="")
    payload = _api_json(api_url, f"/api/runs/{run}/episodes/{episode}/trajectory")
    return payload.get("records", [])


def _steps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("kind") == "step"]


def _timeline_html(records: list[dict[str, Any]], active_idx: int = 0) -> str:
    steps = _steps(records)
    if not steps:
        return "<p>No steps.</p>"
    rows = []
    for idx, rec in enumerate(steps):
        role = rec.get("role", "detective")
        bg = "#eff6ff" if role == "detective" else "#fff7ed"
        border = "2px solid #111" if idx == active_idx else "1px solid #ddd"
        mark = "ok" if rec.get("success", True) else "failed"
        rows.append(
            f"<div style='border:{border};background:{bg};padding:6px;margin:4px 0;border-radius:4px'>"
            f"<b>{idx}</b> {html.escape(role)} "
            f"<code>{html.escape(rec.get('action', ''))}</code> "
            f"<span style='color:#666'>{html.escape(mark)}</span></div>"
        )
    return "".join(rows)


def _footer(records: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in reversed(records) if r.get("kind") == "footer"), {})


def _render_step_records(records: list[dict[str, Any]], step: int | float):
    steps = _steps(records)
    if not steps:
        return "<p>No trajectory selected.</p>", "", "", "", ""
    idx = max(0, min(int(step), len(steps) - 1))
    rec = steps[idx]
    kwargs = json.dumps(rec.get("action_kwargs", {}), indent=2, ensure_ascii=False)
    obs = rec.get("observation", "")
    result = rec.get("result_observation", "")
    raw = rec.get("model_response") or ""
    details = (
        f"### Step {idx}\n"
        f"- Actor: `{rec.get('role')}` / `{rec.get('actor_id')}`\n"
        f"- Action: `{rec.get('action')}`\n"
        f"- Success: `{rec.get('success')}`\n"
        f"- World hash: `{rec.get('world_state_hash', '')[:12]}`\n\n"
        f"```json\n{kwargs}\n```"
    )
    footer = _footer(records)
    score = (footer.get("episode_summary") or {}).get("score_result") or {}
    final = (
        f"### Final\n"
        f"- Error: `{footer.get('error')}`\n"
        f"- Composite: `{score.get('composite_score', '')}`\n"
        f"- Triangle: `{score.get('triangle_score', '')}`\n"
        f"- Alibi: `{score.get('alibi_score', '')}`\n"
        f"- Elimination: `{score.get('elimination_score', '')}`"
    )
    return _timeline_html(records, idx), details, obs, raw, result + "\n\n" + final


def _render_step(path: str | None, step: int | float):
    return _render_step_records(_records(path), step)


def episode_selected(path: str | None):
    records = _records(path)
    steps = _steps(records)
    max_idx = max(1, len(steps) - 1)
    timeline, details, obs, raw, result = _render_step_records(records, 0)
    return gr.update(minimum=0, maximum=max_idx, value=0, step=1), timeline, details, obs, raw, result


def build_app(default_dir: str, api_url: str | None = None) -> gr.Blocks:
    def refresh_view(target: str):
        config, matches, outputs = (
            _load_outputs_api(api_url, target)
            if api_url
            else _load_outputs(target)
        )
        choices = _episode_choices_api(matches) if api_url else _episode_choices(matches)
        first = choices[0][1] if choices else None
        return (
            _overview_html(config, matches, outputs),
            _detective_rows(outputs),
            _culprit_rows(outputs),
            _role_gap_html(outputs),
            _matrix_html(outputs),
            gr.update(choices=choices, value=first),
        )

    def episode_selected_view(target: str, episode_value: str | None):
        records = (
            _records_api(api_url, target, episode_value)
            if api_url
            else _records(episode_value)
        )
        steps = _steps(records)
        max_idx = max(1, len(steps) - 1)
        timeline, details, obs, raw, result = _render_step_records(records, 0)
        return gr.update(minimum=0, maximum=max_idx, value=0, step=1), timeline, details, obs, raw, result

    def render_step_view(target: str, episode_value: str | None, step_value: int | float):
        records = (
            _records_api(api_url, target, episode_value)
            if api_url
            else _records(episode_value)
        )
        return _render_step_records(records, step_value)

    with gr.Blocks(title="MysteryArena Dashboard") as demo:
        gr.Markdown("# MysteryArena Dashboard")
        with gr.Row():
            arena_dir = gr.Textbox(
                label="Arena run id" if api_url else "Arena result directory",
                value=default_dir,
                scale=4,
            )
            refresh_btn = gr.Button("Refresh", variant="primary")

        overview = gr.HTML()
        with gr.Tabs():
            with gr.Tab("Detective Leaderboard"):
                detective_table = gr.Dataframe(
                    headers=DETECTIVE_HEADERS,
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("Culprit Leaderboard"):
                culprit_table = gr.Dataframe(
                    headers=CULPRIT_HEADERS,
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("Role Gap"):
                role_gap = gr.HTML()
            with gr.Tab("Duel Matrix"):
                matrix = gr.HTML()
            with gr.Tab("Episode Replay"):
                episode = gr.Dropdown(label="Episode")
                step = gr.Slider(label="Step", minimum=0, maximum=1, step=1, value=0)
                with gr.Row():
                    timeline = gr.HTML(label="Timeline")
                    with gr.Column():
                        details = gr.Markdown()
                        obs = gr.Textbox(label="Observation into actor", lines=10)
                        raw = gr.Textbox(label="Raw model response", lines=8)
                        result = gr.Textbox(label="Result / final score", lines=10)

        refresh_btn.click(
            refresh_view,
            inputs=[arena_dir],
            outputs=[overview, detective_table, culprit_table, role_gap, matrix, episode],
        )
        demo.load(
            refresh_view,
            inputs=[arena_dir],
            outputs=[overview, detective_table, culprit_table, role_gap, matrix, episode],
        )
        episode.change(
            episode_selected_view,
            inputs=[arena_dir, episode],
            outputs=[step, timeline, details, obs, raw, result],
        )
        step.change(
            render_step_view,
            inputs=[arena_dir, episode, step],
            outputs=[timeline, details, obs, raw, result],
        )
    return demo


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _pick_port(host: str, requested: int, scan: int, strict: bool) -> int:
    if strict or _port_available(host, requested):
        return requested
    for port in range(requested + 1, requested + scan + 1):
        if _port_available(host, port):
            print(f"Port {requested} is busy; using {port} instead.")
            return port
    raise OSError(
        f"Cannot find empty port in range: {requested}-{requested + scan}. "
        "Pass --port with a free port, or close the existing dashboard."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch MysteryArena dashboard")
    parser.add_argument("--arena-dir", default="arena/results")
    parser.add_argument(
        "--api-url",
        default=None,
        help="MysteryArena backend API URL. When set, the textbox contains a run id.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Initial run id to load when --api-url is set.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--port-scan",
        type=int,
        default=50,
        help="If --port is busy, scan this many subsequent ports.",
    )
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="Fail instead of falling back when --port is busy.",
    )
    args = parser.parse_args()
    if args.api_url:
        default_target = args.run_id or _default_api_run(args.api_url, args.arena_dir)
    else:
        default_target = args.arena_dir
    app = build_app(default_target, api_url=args.api_url)
    port = _pick_port(args.host, args.port, args.port_scan, args.strict_port)
    app.launch(server_name=args.host, server_port=port)


if __name__ == "__main__":
    main()
