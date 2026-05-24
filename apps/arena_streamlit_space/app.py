from __future__ import annotations

import gzip
import hashlib
import html
import json
import os
import re
from typing import Any
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


DEFAULT_REPO = os.environ.get("ARENA_DATASET_REPO", "org/mystery-arena-results")
DEFAULT_REVISION = os.environ.get("ARENA_DEFAULT_REVISION", "main")
DEFAULT_BASE_URL = os.environ.get("ARENA_DATASET_BASE_URL", "").rstrip("/")
DEFAULT_MATCHES_FILE = "matches/all_matches.jsonl.gz"


st.set_page_config(
    page_title="MysteryArena Arena",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #141817;
          --muted: #64706b;
          --line: #dbe3df;
          --panel: #ffffff;
          --soft: #f5f7f6;
          --green: #16836f;
          --amber: #b67616;
          --red: #b84a45;
          --cyan: #0d9488;
        }
        html, body, [data-testid="stAppViewContainer"] {
          background: var(--soft);
          color: var(--ink);
        }
        .block-container {
          padding-top: 3rem;
          padding-bottom: 2rem;
          max-width: 1540px;
        }
        [data-testid="stSidebar"] {
          background: #111614;
          border-right: 1px solid #27312d;
        }
        [data-testid="stSidebar"] * {
          color: #e7eee9;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] * {
          color: #111614 !important;
        }
        [data-testid="stSidebar"] .stButton button {
          background: #1f7a6a;
          color: #ffffff;
          border: 0;
          border-radius: 6px;
        }
        .stButton button {
          justify-content: flex-start;
          text-align: left;
          padding-left: 14px;
          padding-right: 14px;
        }
        .stButton button p {
          width: 100%;
          text-align: left;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          white-space: normal;
          overflow-wrap: anywhere;
        }
        [data-testid="stSidebar"] .stButton button {
          justify-content: center;
        }
        [data-testid="stSidebar"] .stButton button p {
          text-align: center;
          font-family: inherit;
        }
        div[data-testid="stMetric"] {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 11px 13px;
          box-shadow: 0 1px 2px rgba(20, 24, 23, 0.05);
        }
        div[data-testid="stMetric"] label {
          color: var(--muted);
          font-weight: 700;
        }
        div[data-testid="stMetricValue"] {
          color: var(--ink);
          font-size: 1.35rem;
        }
        .arena-console {
          background: #101412;
          color: #ecf4ef;
          border: 1px solid #27312d;
          border-radius: 8px;
          padding: 24px 18px 14px 18px;
          margin-top: 12px;
          margin-bottom: 14px;
          box-shadow: 0 12px 32px rgba(16, 20, 18, 0.18);
          overflow: visible;
        }
        .console-top {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 16px;
          flex-wrap: wrap;
        }
        .console-title {
          font-size: 1.6rem;
          line-height: 1.35;
          font-weight: 780;
          margin: 0;
          letter-spacing: 0;
          padding-top: 1px;
        }
        .console-grid {
          display: grid;
          grid-template-columns: repeat(5, minmax(120px, 1fr));
          gap: 8px;
          margin-top: 14px;
        }
        .console-kv {
          border: 1px solid #2e3a35;
          background: #171d1a;
          border-radius: 7px;
          padding: 8px 10px;
          min-height: 54px;
        }
        .console-kv small {
          display: block;
          color: #8da299;
          font-size: 0.72rem;
          text-transform: uppercase;
          font-weight: 760;
        }
        .console-kv strong {
          display: block;
          margin-top: 2px;
          color: #f4faf7;
          font-size: 0.98rem;
          word-break: break-word;
        }
        .panel {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 14px;
          margin-bottom: 12px;
          box-shadow: 0 1px 2px rgba(20, 24, 23, 0.04);
        }
        .panel-title {
          font-size: 0.82rem;
          color: var(--muted);
          text-transform: uppercase;
          font-weight: 800;
          margin-bottom: 9px;
        }
        .run-row,
        .rank-card,
        .feed-row {
          border: 1px solid var(--line);
          background: #fbfcfb;
          border-radius: 7px;
          padding: 9px 10px;
          margin-bottom: 7px;
        }
        .run-row {
          display: grid;
          grid-template-columns: 1.6fr 0.7fr 0.7fr 1fr;
          gap: 10px;
          align-items: center;
          font-size: 0.86rem;
        }
        .rank-card {
          display: grid;
          grid-template-columns: 42px 1fr 82px;
          gap: 10px;
          align-items: center;
        }
        .rank {
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-weight: 800;
          color: var(--green);
        }
        .model-name {
          font-weight: 780;
          color: var(--ink);
          overflow-wrap: anywhere;
        }
        .muted {
          color: var(--muted);
          font-size: 0.82rem;
        }
        .score {
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-weight: 800;
          text-align: right;
          color: var(--ink);
        }
        .bar {
          height: 7px;
          background: #e4ebe7;
          border-radius: 999px;
          overflow: hidden;
          margin-top: 6px;
        }
        .bar span {
          display: block;
          height: 100%;
          background: linear-gradient(90deg, var(--cyan), #68b88f);
          border-radius: 999px;
        }
        .replay-summary {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 9px;
          margin-bottom: 12px;
        }
        .replay-card {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 11px 12px;
          min-height: 76px;
          box-shadow: 0 1px 2px rgba(20, 24, 23, 0.05);
        }
        .replay-card small {
          display: block;
          color: var(--muted);
          font-weight: 760;
          margin-bottom: 6px;
        }
        .replay-card strong {
          display: block;
          color: var(--ink);
          font-size: 1rem;
          line-height: 1.25;
          overflow-wrap: anywhere;
          word-break: break-word;
          white-space: normal;
        }
        .replay-card strong.number {
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: 1.2rem;
        }
        .feed-row {
          display: grid;
          grid-template-columns: 44px 1fr auto;
          gap: 10px;
          align-items: start;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: 0.78rem;
        }
        .feed-row.active {
          border-color: #1f8f7a;
          background: #eefaf6;
        }
        .tag {
          display: inline-block;
          border-radius: 999px;
          padding: 2px 7px;
          border: 1px solid var(--line);
          background: #f2f6f4;
          color: #35423d;
          font-size: 0.72rem;
          font-weight: 760;
          margin-right: 4px;
        }
        .tag.ok {
          border-color: #acd8ca;
          background: #ecfaf5;
          color: #116b59;
        }
        .tag.fail {
          border-color: #efb2ad;
          background: #fff0ee;
          color: #9f352f;
        }
        .terminal-box {
          background: #101412;
          color: #dce9e3;
          border: 1px solid #27312d;
          border-radius: 8px;
          padding: 12px;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: 0.82rem;
          white-space: pre-wrap;
          min-height: 140px;
          max-height: 360px;
          overflow: auto;
        }
        .stTabs [data-baseweb="tab-list"] {
          gap: 6px;
        }
        .stTabs [data-baseweb="tab"] {
          background: #ffffff;
          border: 1px solid var(--line);
          border-radius: 7px 7px 0 0;
          padding: 8px 12px;
        }
        @media (max-width: 900px) {
          .console-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
          .run-row { grid-template-columns: 1fr; }
          .rank-card { grid-template-columns: 36px 1fr; }
          .replay-summary { grid-template-columns: 1fr; }
          .score { text-align: left; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _pct(value: Any) -> str:
    return f"{_safe_float(value):.1%}"


def _resolve_url(repo_id: str, revision: str, path: str, base_url: str = "") -> str:
    if base_url:
        clean_path = quote(path.lstrip("/"), safe="/")
        return f"{base_url.rstrip('/')}/{clean_path}"
    repo = quote(repo_id.strip(), safe="/")
    rev = quote(revision.strip() or "main", safe="")
    clean_path = quote(path.lstrip("/"), safe="/")
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{clean_path}"


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_bytes(repo_id: str, revision: str, path: str, base_url: str = "") -> bytes:
    url = _resolve_url(repo_id, revision, path, base_url)
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    return response.content


@st.cache_data(ttl=300, show_spinner=False)
def load_json(repo_id: str, revision: str, path: str, base_url: str = "") -> dict[str, Any]:
    return json.loads(_fetch_bytes(repo_id, revision, path, base_url).decode("utf-8"))


@st.cache_data(ttl=300, show_spinner=False)
def load_jsonl_gz(repo_id: str, revision: str, path: str, base_url: str = "") -> list[dict[str, Any]]:
    raw = gzip.decompress(_fetch_bytes(repo_id, revision, path, base_url)).decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _matches_df(matches: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "run_id",
        "match_id",
        "detective",
        "culprit",
        "level",
        "seed",
        "detective_payoff",
        "culprit_payoff",
        "solved",
        "actions",
        "culprit_actions",
        "error",
        "trajectory_file",
    ]
    rows = []
    for match in matches:
        detective = match.get("detective") or {}
        culprit = match.get("culprit") or {}
        rows.append({
            "run_id": match.get("run_id"),
            "match_id": match.get("match_id"),
            "detective": detective.get("name"),
            "culprit": culprit.get("name"),
            "level": match.get("level"),
            "seed": match.get("seed"),
            "detective_payoff": match.get("detective_payoff"),
            "culprit_payoff": match.get("culprit_payoff"),
            "solved": match.get("solved"),
            "actions": match.get("actions_taken"),
            "culprit_actions": match.get("culprit_actions_taken"),
            "error": match.get("error"),
            "trajectory_file": match.get("trajectory_file"),
        })
    return pd.DataFrame(rows, columns=columns)


def _load_all_matches(repo_id: str, revision: str, index: dict[str, Any], base_url: str) -> list[dict[str, Any]]:
    matches_file = str(index.get("matches_file") or DEFAULT_MATCHES_FILE)
    try:
        return load_jsonl_gz(repo_id, revision, matches_file, base_url)
    except Exception:
        merged: list[dict[str, Any]] = []
        for item in index.get("runs", []):
            run_matches_file = item.get("matches_file")
            if not run_matches_file:
                continue
            merged.extend(load_jsonl_gz(repo_id, revision, str(run_matches_file), base_url))
        return merged


def _leaderboard_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for row in rows:
        rating = row.get("trueskill") or {}
        flat.append({
            "rank": row.get("rank"),
            "model": row.get("model"),
            "n": row.get("n"),
            "mean_payoff": row.get("mean_payoff"),
            "ci_low": row.get("ci_low"),
            "ci_high": row.get("ci_high"),
            "skill": rating.get("skill", row.get("skill")),
            "mu": rating.get("mu", row.get("mu")),
            "sigma": rating.get("sigma", row.get("sigma")),
            "solve_rate": row.get("solve_rate"),
            "detective_failure_rate": row.get("detective_failure_rate"),
            "avg_actions": row.get("avg_actions", row.get("avg_detective_actions")),
            "guard_blocked": row.get("guard_blocked"),
        })
    return pd.DataFrame(flat)


def _filter_matches(df: pd.DataFrame, models: list[str], levels: list[str]) -> pd.DataFrame:
    filtered = df
    if models:
        filtered = filtered[
            filtered["detective"].isin(models) | filtered["culprit"].isin(models)
        ]
    if levels:
        filtered = filtered[filtered["level"].isin(levels)]
    return filtered


def _global_leaderboard(matches: pd.DataFrame, *, role: str) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()
    if role == "detective":
        model_col = "detective"
        payoff_col = "detective_payoff"
    else:
        model_col = "culprit"
        payoff_col = "culprit_payoff"
    rows = []
    for model, group in matches.dropna(subset=[model_col]).groupby(model_col):
        payoff = pd.to_numeric(group[payoff_col], errors="coerce")
        solved = pd.to_numeric(group["solved"], errors="coerce") if "solved" in group else pd.Series(dtype=float)
        actions = pd.to_numeric(group["actions"], errors="coerce") if "actions" in group else pd.Series(dtype=float)
        rows.append({
            "model": model,
            "n": int(len(group)),
            "mean_payoff": float(payoff.mean()) if not payoff.dropna().empty else 0.0,
            "skill": float(payoff.mean()) if not payoff.dropna().empty else 0.0,
            "solve_rate": float(solved.mean()) if role == "detective" and not solved.dropna().empty else None,
            "detective_failure_rate": (
                1.0 - float(solved.mean())
                if role == "culprit" and not solved.dropna().empty
                else None
            ),
            "avg_actions": float(actions.mean()) if not actions.dropna().empty else None,
        })
    rows.sort(key=lambda item: item["mean_payoff"], reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return pd.DataFrame(rows)


def _global_outputs(matches: pd.DataFrame) -> dict[str, Any]:
    detectives = _global_leaderboard(matches, role="detective")
    culprits = _global_leaderboard(matches, role="culprit")
    return {
        "detective_leaderboard": detectives.to_dict("records"),
        "culprit_leaderboard": culprits.to_dict("records"),
        "summary": {
            "matches": int(len(matches)),
            "detectives": int(matches["detective"].nunique()) if not matches.empty else 0,
            "culprits": int(matches["culprit"].nunique()) if not matches.empty else 0,
        },
    }


def _runs_df(index: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in index.get("runs", []):
        rows.append({
            "run_id": item.get("run_id"),
            "matches": item.get("matches"),
            "levels": ", ".join(str(level) for level in item.get("levels", [])),
            "top_detective": item.get("top_detective"),
            "top_culprit": item.get("top_culprit"),
            "published_at": item.get("published_at"),
        })
    return pd.DataFrame(rows)


def render_header(index: dict[str, Any], matches: pd.DataFrame) -> None:
    outputs = _global_outputs(matches)
    best_d = (outputs.get("detective_leaderboard") or [{}])[0].get("model")
    best_c = (outputs.get("culprit_leaderboard") or [{}])[0].get("model")
    total_matches = sum(
        int(item.get("matches") or 0)
        for item in index.get("runs", [])
    )
    st.markdown(
        f"""
        <div class="arena-console">
          <div class="console-top">
            <div>
              <div class="console-title">MysteryArena</div>
            </div>
          </div>
          <div class="console-grid">
            <div class="console-kv"><small>Run</small><strong>All Runs</strong></div>
            <div class="console-kv"><small>Matches</small><strong>{len(matches)}</strong></div>
            <div class="console-kv"><small>Total Matches</small><strong>{total_matches}</strong></div>
            <div class="console-kv"><small>Top Detective</small><strong>{_escape(best_d or '-')}</strong></div>
            <div class="console-kv"><small>Top Culprit</small><strong>{_escape(best_c or '-')}</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_rank_cards(df: pd.DataFrame, *, value_col: str = "mean_payoff", limit: int = 6) -> None:
    if df.empty:
        st.info("No leaderboard rows.")
        return
    cards = []
    for _, row in df.head(limit).iterrows():
        score = _safe_float(row.get(value_col))
        width = max(4, min(100, score * 100))
        cards.append(
            f"""
            <div class="rank-card">
              <div class="rank">#{_escape(row.get('rank'))}</div>
              <div>
                <div class="model-name">{_escape(row.get('model'))}</div>
                <div class="muted">n={_escape(row.get('n'))} skill={_metric_value(row.get('skill'))}</div>
                <div class="bar"><span style="width:{width:.1f}%"></span></div>
              </div>
              <div class="score">{score:.3f}</div>
            </div>
            """
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


def render_overview(index: dict[str, Any], matches: pd.DataFrame) -> None:
    outputs = _global_outputs(matches)
    counts = outputs.get("summary", {})
    solve_rate = float(matches["solved"].mean()) if not matches.empty and "solved" in matches else 0.0
    avg_actions = float(matches["actions"].dropna().mean()) if not matches.empty else 0.0

    cols = st.columns(7)
    cols[0].metric("Matches", _metric_value(counts.get("matches", len(matches))))
    cols[1].metric("Filtered", _metric_value(len(matches)))
    cols[2].metric("Detectives", _metric_value(counts.get("detectives")))
    cols[3].metric("Culprits", _metric_value(counts.get("culprits")))
    cols[4].metric("Solve Rate", f"{solve_rate:.1%}")
    cols[5].metric("Avg Actions", f"{avg_actions:.1f}")
    cols[6].metric("Runs", _metric_value(len(index.get("runs", []))))

    left, mid = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Detective Signal</div>', unsafe_allow_html=True)
            render_rank_cards(_leaderboard_df(outputs.get("detective_leaderboard", [])))
    with mid:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Culprit Signal</div>', unsafe_allow_html=True)
            render_rank_cards(_leaderboard_df(outputs.get("culprit_leaderboard", [])))

    with st.container(border=True):
        st.markdown('<div class="panel-title">Match Ledger</div>', unsafe_allow_html=True)
        visible = matches.drop(columns=["trajectory_file"], errors="ignore")
        st.dataframe(
            visible,
            hide_index=True,
            width="stretch",
            height=420,
            column_config={
                "detective_payoff": st.column_config.NumberColumn("detective_payoff", format="%.3f"),
                "culprit_payoff": st.column_config.NumberColumn("culprit_payoff", format="%.3f"),
            },
        )


def render_leaderboards(matches: pd.DataFrame) -> None:
    outputs = _global_outputs(matches)
    d_df = _leaderboard_df(outputs.get("detective_leaderboard", []))
    c_df = _leaderboard_df(outputs.get("culprit_leaderboard", []))
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Detective Leaderboard</div>', unsafe_allow_html=True)
            render_rank_cards(d_df, limit=8)
            st.dataframe(d_df, hide_index=True, width="stretch", height=360)
    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Culprit Leaderboard</div>', unsafe_allow_html=True)
            render_rank_cards(c_df, limit=8)
            st.dataframe(c_df, hide_index=True, width="stretch", height=360)


def render_matrix(matches: pd.DataFrame) -> None:
    if matches.empty:
        st.info("No duel matrix for the current filters.")
        return
    df = (
        matches.groupby(["detective", "culprit"], dropna=True)
        .agg(detective_payoff=("detective_payoff", "mean"), n=("match_id", "count"))
        .reset_index()
    )
    pivot = df.pivot(index="detective", columns="culprit", values="detective_payoff").astype(float)
    count_pivot = df.pivot(index="detective", columns="culprit", values="n").fillna(0).astype(int)
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    col_order = pivot.mean(axis=0).sort_values(ascending=True).index
    pivot = pivot.reindex(index=row_order, columns=col_order)
    count_pivot = count_pivot.reindex(index=row_order, columns=col_order).fillna(0).astype(int)

    text = []
    for detective in pivot.index:
        row = []
        for culprit in pivot.columns:
            value = pivot.loc[detective, culprit]
            count = int(count_pivot.loc[detective, culprit])
            row.append("" if pd.isna(value) else f"{value:.2f}<br><span style='font-size:10px'>n={count}</span>")
        text.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            text=text,
            texttemplate="%{text}",
            textfont={"size": 12, "color": "#111614"},
            customdata=count_pivot.values,
            zmin=0,
            zmax=1,
            colorscale=[
                [0.0, "#8e2430"],
                [0.2, "#d95f4d"],
                [0.5, "#f4e7ad"],
                [0.8, "#38a982"],
                [1.0, "#075f55"],
            ],
            xgap=3,
            ygap=3,
            hovertemplate=(
                "Detective=%{y}<br>"
                "Culprit=%{x}<br>"
                "Payoff=%{z:.3f}<br>"
                "Matches=%{customdata}<extra></extra>"
            ),
            colorbar=dict(
                title="Payoff",
                thickness=14,
                len=0.84,
                tickmode="array",
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=["0", "0.25", "0.50", "0.75", "1"],
            ),
        )
    )
    fig.update_layout(
        height=max(520, 46 * len(pivot.index) + 120),
        margin=dict(l=10, r=10, t=24, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis=dict(
            title="Culprit",
            side="top",
            tickangle=-35,
            showgrid=False,
            zeroline=False,
            categoryorder="array",
            categoryarray=list(pivot.columns),
        ),
        yaxis=dict(
            title="Detective",
            showgrid=False,
            zeroline=False,
            autorange="reversed",
            categoryorder="array",
            categoryarray=list(pivot.index),
        ),
        font=dict(color="#141817", size=12),
    )
    with st.container(border=True):
        st.markdown('<div class="panel-title">Payoff Matrix</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, width="stretch")


def _step_label(record: dict[str, Any], idx: int) -> str:
    role = record.get("role", "detective")
    action = record.get("action", "")
    return f"#{idx:02d}  {role}  {action or '-'}"


def _component_key(value: Any) -> str:
    raw = "" if value is None else str(value)
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", raw)[:42].strip("_")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{normalized}_{digest}" if normalized else digest


def _step_feed(steps: list[dict[str, Any]], selected_idx: int) -> str:
    start = max(0, selected_idx - 5)
    end = min(len(steps), selected_idx + 6)
    rows = []
    for idx in range(start, end):
        record = steps[idx]
        active = " active" if idx == selected_idx else ""
        rows.append(
            f"""
            <div class="feed-row{active}">
              <div>#{idx:02d}</div>
              <div>
                <span class="tag">{_escape(record.get('role', 'detective'))}</span>
                <div class="model-name">{_escape(record.get('action') or '-')}</div>
              </div>
              <div class="muted">{_escape(str(record.get('world_state_hash', ''))[:8])}</div>
            </div>
            """
        )
    return "".join(rows)


def _render_step_buttons(steps: list[dict[str, Any]], *, episode_key: str) -> int:
    selected_episode_key = f"replay_episode_{episode_key}"
    selected_step_key = f"replay_step_{episode_key}"
    if st.session_state.get(selected_episode_key) != episode_key:
        st.session_state[selected_episode_key] = episode_key
        st.session_state[selected_step_key] = 0

    selected_idx = int(st.session_state.get(selected_step_key, 0))
    selected_idx = max(0, min(selected_idx, len(steps) - 1))

    for idx, record in enumerate(steps):
        label = _step_label(record, idx)
        button_type = "primary" if idx == selected_idx else "secondary"
        if st.button(
            label,
            key=f"step_button_{episode_key}_{idx}",
            width="stretch",
            type=button_type,
        ):
            selected_idx = idx
            st.session_state[selected_step_key] = idx

    return selected_idx


def _terminal_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return _escape(text)


def _replay_summary(selected: pd.Series) -> str:
    detective_payoff = _safe_float(selected.get("detective_payoff"))
    culprit_payoff = _safe_float(selected.get("culprit_payoff"), 1.0 - detective_payoff)
    return f"""
    <div class="replay-summary">
      <div class="replay-card">
        <small>Detective</small>
        <strong>{_escape(selected.get("detective"))}</strong>
      </div>
      <div class="replay-card">
        <small>Culprit</small>
        <strong>{_escape(selected.get("culprit"))}</strong>
      </div>
      <div class="replay-card">
        <small>Detective Payoff</small>
        <strong class="number">{detective_payoff:.3f}</strong>
      </div>
      <div class="replay-card">
        <small>Culprit Payoff</small>
        <strong class="number">{culprit_payoff:.3f}</strong>
      </div>
    </div>
    """


def render_replay(repo_id: str, revision: str, base_url: str, matches: pd.DataFrame) -> None:
    replayable = matches[matches["trajectory_file"].notna()] if not matches.empty else matches
    if replayable.empty:
        st.info("No replayable trajectories in this run.")
        return

    labels = []
    for _, row in replayable.iterrows():
        labels.append(
            f"{row['detective']} vs {row['culprit']} | {row['level']} seed={row['seed']} | "
            f"{row['run_id']} | "
            f"D={_safe_float(row['detective_payoff']):.3f}"
        )
    selected_label = st.selectbox("Episode", labels, index=0)
    selected = replayable.iloc[labels.index(selected_label)]
    records = load_jsonl_gz(repo_id, revision, str(selected["trajectory_file"]), base_url)
    steps = [record for record in records if record.get("kind") == "step"]
    footer = next((record for record in reversed(records) if record.get("kind") == "footer"), {})

    if not steps:
        st.info("No steps in selected trajectory.")
        return

    left, right = st.columns([0.8, 1.55])
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Episode Replay</div>', unsafe_allow_html=True)
            st.markdown(_replay_summary(selected), unsafe_allow_html=True)
            episode_key = _component_key(selected["trajectory_file"])
            step_idx = _render_step_buttons(steps, episode_key=episode_key)
            st.progress((step_idx + 1) / len(steps))
            st.markdown(_step_feed(steps, step_idx), unsafe_allow_html=True)

    record = steps[step_idx]
    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Step State</div>', unsafe_allow_html=True)
            top = st.columns(4)
            top[0].metric("Role", _metric_value(record.get("role")))
            top[1].metric("Action", _metric_value(record.get("action")))
            top[2].metric("Success", _metric_value(record.get("success")))
            top[3].metric("Hash", str(record.get("world_state_hash", ""))[:12])

            obs_col, result_col = st.columns(2)
            with obs_col:
                st.markdown("**Observation**")
                st.markdown(
                    f'<div class="terminal-box">{_terminal_text(record.get("observation"))}</div>',
                    unsafe_allow_html=True,
                )
            with result_col:
                st.markdown("**Result**")
                st.markdown(
                    f'<div class="terminal-box">{_terminal_text(record.get("result_observation"))}</div>',
                    unsafe_allow_html=True,
                )

            details_col, final_col = st.columns(2)
            with details_col:
                st.markdown("**Action Args**")
                st.json(record.get("action_kwargs", record.get("action_args", {})), expanded=False)
            with final_col:
                st.markdown("**Final**")
                final = footer.get("episode_summary", {}).get("score_result") or footer.get("metrics") or {}
                st.json(final, expanded=False)


def render_api_docs() -> None:
    base_url = os.environ.get("ARENA_API_PUBLIC_URL", "https://elfsong-mystery-arena.hf.space")
    st.markdown("### API Document")
    st.caption("How to start, manage, commit, and publish MysteryArena duels.")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Where The Backend Runs</div>', unsafe_allow_html=True)
            st.markdown(
                """
                This Docker Space runs both services. Streamlit serves the viewer, FastAPI serves
                `/api/*`, and nginx exposes both through the same Hugging Face Space URL.

                Use the Space URL as `ARENA_API_URL`; do not point users at a separate local backend
                unless they are developing locally.
                """
            )
            st.code(
                f"""export ARENA_API_URL="{base_url}"
export ARENA_SPACE_URL="https://elfsong-mystery-arena.hf.space"

curl "$ARENA_API_URL/api/health"
git clone https://huggingface.co/spaces/Elfsong/Mystery_Arena
cd Mystery_Arena
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/arena_client.py --help""",
                language="bash",
            )
    with c2:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Endpoint Map</div>', unsafe_allow_html=True)
            st.code(
                """GET  /api/health
GET  /api/models

POST /api/arena/matches
GET  /api/arena/jobs/{job_id}
POST /api/arena/runs/{run_id}/publish-hf

POST /api/sessions
GET  /api/sessions/{session_id}
POST /api/sessions/{session_id}/actions
POST /api/sessions/{session_id}/commit""",
                language="text",
            )

    st.markdown("### Required Parameters")
    st.caption("These fields define the match. `culprit` is the opponent/culprit side; it is not `corporate`.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "field": "level",
                    "where": "match, session, client CLI",
                    "example": "TRIVIAL",
                    "meaning": "Difficulty bucket. Allowed values: TRIVIAL, EASY, MEDIUM, HARD, EXPERT.",
                },
                {
                    "field": "seed",
                    "where": "match, session, client CLI",
                    "example": "0",
                    "meaning": "Deterministic case id. Same level + seed creates the same mystery instance.",
                },
                {
                    "field": "detective",
                    "where": "/api/arena/matches, /api/sessions",
                    "example": "gpt-5.5 or human",
                    "meaning": "Solver side. The detective tries to identify suspect, weapon, and location.",
                },
                {
                    "field": "culprit",
                    "where": "/api/arena/matches, /api/sessions",
                    "example": "kimi-k2.5 or passive",
                    "meaning": "Culprit side. The culprit tries to obstruct the detective or preserve uncertainty.",
                },
                {
                    "field": "player_role",
                    "where": "/api/sessions only",
                    "example": "detective",
                    "meaning": "Which side the local client controls: detective, culprit, or both.",
                },
                {
                    "field": "run_id",
                    "where": "match, commit, client CLI",
                    "example": "gpt55_vs_kimi_trivial_0",
                    "meaning": "Result group written by the backend and later shown by the viewer.",
                },
                {
                    "field": "publish_hf",
                    "where": "commit, client CLI",
                    "example": "true",
                    "meaning": "If true, publish the completed trajectory to the configured Hugging Face Dataset.",
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    st.markdown("### Which Flow To Use")
    st.markdown(
        """
        - Use `POST /api/arena/matches` when both players are backend-registered models.
          The Space backend needs server-side model gateway secrets for this flow.
        - Use `POST /api/sessions` when a human or local model client controls one or both sides.
          Local model API keys stay on the user's machine; the client sends only selected actions
          and trajectory data to Arena.
        """
    )

    st.markdown("### 1. Model A vs Model B")
    st.caption("Use this when both models are already registered on the Arena backend.")
    st.code(
        """python scripts/arena_client.py match \\
  --api-url "$ARENA_API_URL" \\
  --detective gpt-5.5 \\
  --culprit kimi-k2.5 \\
  --level TRIVIAL \\
  --seed 0 \\
  --run-id gpt55_vs_kimi_trivial_0 \\
  --publish-hf""",
        language="bash",
    )
    st.code(
        """POST /api/arena/matches
{
  "detective": "gpt-5.5",
  "culprit": "kimi-k2.5",
  "level": "TRIVIAL",
  "seed": 0,
  "run_id": "gpt55_vs_kimi_trivial_0"
}

GET /api/arena/jobs/{job_id}""",
        language="text",
    )

    st.markdown("### 2. Human vs Model")
    st.caption("Run the local TUI player. The backend records every step and commits the trajectory at the end.")
    human_left, human_right = st.columns(2)
    with human_left:
        st.markdown("**Human as detective**")
        st.code(
            """python scripts/arena_client.py human \\
  --api-url "$ARENA_API_URL" \\
  --role detective \\
  --opponent gpt-5.5 \\
  --level TRIVIAL \\
  --seed 0 \\
  --run-id human_vs_gpt55_trivial_0 \\
  --publish-hf""",
            language="bash",
        )
    with human_right:
        st.markdown("**Human as culprit**")
        st.code(
            """python scripts/arena_client.py human \\
  --api-url "$ARENA_API_URL" \\
  --role culprit \\
  --opponent gpt-5.5 \\
  --level TRIVIAL \\
  --seed 0 \\
  --run-id gpt55_vs_human_culprit_0 \\
  --publish-hf""",
            language="bash",
        )
    st.code(
        """POST /api/sessions
{
  "player_role": "detective",
  "detective": "human",
  "culprit": "gpt-5.5",
  "level": "TRIVIAL",
  "seed": 0
}

POST /api/sessions/{session_id}/actions
{
  "action": "TALK_TO",
  "action_args": {
    "character_name": "Avery Stone",
    "question": "Where were you at the time of the murder?"
  }
}

POST /api/sessions/{session_id}/commit
{
  "run_id": "human_vs_gpt55_trivial_0",
  "publish_hf": true,
  "include_model_responses": false
}""",
        language="json",
    )

    st.markdown("### 3. Two External Models")
    st.caption("Both model keys remain local. The Arena backend stores observations/actions, not provider keys.")
    st.code(
        """export MODEL_BASE_URL="http://127.0.0.1:9000/v1"
export MODEL_API_KEY="..."
export MODEL_NAME="model-a"

export CULPRIT_MODEL_BASE_URL="http://127.0.0.1:9001/v1"
export CULPRIT_MODEL_API_KEY="..."
export CULPRIT_MODEL_NAME="model-b"

python scripts/arena_client.py model \\
  --api-url "$ARENA_API_URL" \\
  --role both \\
  --detective-model "$MODEL_NAME" \\
  --detective-base-url "$MODEL_BASE_URL" \\
  --detective-api-key-env MODEL_API_KEY \\
  --culprit-model "$CULPRIT_MODEL_NAME" \\
  --culprit-base-url "$CULPRIT_MODEL_BASE_URL" \\
  --culprit-api-key-env CULPRIT_MODEL_API_KEY \\
  --level TRIVIAL \\
  --seed 0 \\
  --publish-hf""",
        language="bash",
    )


def render_filters(matches: pd.DataFrame) -> tuple[list[str], list[str]]:
    all_models = sorted(set(matches.get("detective", [])) | set(matches.get("culprit", [])))
    all_levels = sorted(str(level) for level in matches.get("level", pd.Series(dtype=str)).dropna().unique())
    model_col, level_col = st.columns([1.4, 1.0])
    with model_col:
        models = st.multiselect("Models", all_models, placeholder="All models")
    with level_col:
        levels = st.multiselect("Levels", all_levels, placeholder="All levels")
    return models, levels


def main() -> None:
    _css()
    repo_id = DEFAULT_REPO
    revision = DEFAULT_REVISION
    base_url = DEFAULT_BASE_URL

    try:
        index = load_json(repo_id, revision, "index/runs.json", base_url)
    except Exception as exc:  # noqa: BLE001 - show data loading failure in the app.
        st.error(f"Could not load index/runs.json: {exc}")
        st.stop()

    if not index.get("runs"):
        st.warning("No published Arena runs found.")
        st.stop()

    try:
        matches = _matches_df(_load_all_matches(repo_id, revision, index, base_url))
    except Exception as exc:  # noqa: BLE001 - show data loading failure in the app.
        st.error(f"Could not load unified matches: {exc}")
        st.stop()

    models, levels = render_filters(matches)
    filtered_matches = _filter_matches(matches, models, levels)
    render_header(index, filtered_matches)

    overview, leaderboards, matrix, replay, api_docs = st.tabs([
        "Overview",
        "Leaderboards",
        "Duel Matrix",
        "Episode Replay",
        "API Docs",
    ])
    with overview:
        render_overview(index, filtered_matches)
    with leaderboards:
        render_leaderboards(filtered_matches)
    with matrix:
        render_matrix(filtered_matches)
    with replay:
        render_replay(repo_id, revision, base_url, filtered_matches)
    with api_docs:
        render_api_docs()


if __name__ == "__main__":
    main()
