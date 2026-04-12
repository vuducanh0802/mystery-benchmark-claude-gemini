"""
Statistical analysis and figure generation for the paper.

Produces:
  - Figure 1: Solve rate × complexity level curves (per agent type)
  - Figure 2: Belief accuracy trajectory over steps
  - Figure 3: Clue efficiency bar chart
  - Figure 4: Token cost scaling
  - Table 2: Full numeric results
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from evaluation.metrics import AggregateMetrics, EpisodeMetrics, aggregate_metrics


def load_results(results_dir: str | Path) -> pd.DataFrame:
    """Load all episode metric files from a results directory into a DataFrame."""
    results_dir = Path(results_dir)
    records = []
    metrics_file = results_dir / "all_metrics.json"
    if metrics_file.exists():
        data = json.loads(metrics_file.read_text())
        records.extend(data)
    else:
        for f in results_dir.glob("episode_*.json"):
            ep = json.loads(f.read_text())
            if ep.get("metrics"):
                records.append(ep["metrics"])
    return pd.DataFrame(records)


def compute_aggregate_table(
    dfs: dict[str, pd.DataFrame],
    levels: list[int] = [1, 2, 3, 4, 5],
) -> pd.DataFrame:
    """
    Compute Table 2 (main results) across agent types and complexity levels.

    Parameters
    ----------
    dfs : dict[str, DataFrame]
        Agent name → DataFrame of episode metrics.
    levels : list[int]
        Complexity levels.

    Returns
    -------
    DataFrame with rows = (agent, level) and columns = metric names.
    """
    rows = []
    for agent_name, df in dfs.items():
        for level in levels:
            level_df = df[df["complexity_level"] == level]
            if level_df.empty:
                continue
            n = len(level_df)
            sr = level_df["solved"].mean()
            rows.append({
                "agent": agent_name,
                "level": level,
                "n": n,
                "solve_rate": sr,
                "solve_rate_ci95": 1.96 * np.sqrt(sr * (1 - sr) / max(n, 1)),
                "partial_score": level_df["partial_score"].mean(),
                "belief_accuracy": level_df["final_belief_accuracy"].mean(),
                "clue_efficiency": level_df["clue_efficiency"].mean(),
                "mean_tokens": level_df["total_tokens"].mean(),
                "action_efficiency": level_df["action_efficiency"].mean(),
            })
    return pd.DataFrame(rows)


def significance_test(
    df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str = "solved"
) -> dict[str, float]:
    """Two-sample proportion test (or t-test) between two agent result sets."""
    a = df_a[metric].values.astype(float)
    b = df_b[metric].values.astype(float)
    if metric == "solved":
        # Two-proportion z-test
        p1, p2 = a.mean(), b.mean()
        n1, n2 = len(a), len(b)
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) if p_pool > 0 else 1e-10
        z = (p1 - p2) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        return {"z_statistic": float(z), "p_value": float(p_value), "effect_size": float(p1 - p2)}
    else:
        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        cohens_d = (a.mean() - b.mean()) / np.sqrt((a.std() ** 2 + b.std() ** 2) / 2) if (a.std() + b.std()) > 0 else 0
        return {"t_statistic": float(t_stat), "p_value": float(p_value), "cohens_d": float(cohens_d)}


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def set_paper_style() -> None:
    """Set matplotlib style suitable for ACL/EMNLP/ICLR papers."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        "figure.figsize": (7, 4.5),
        "figure.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })


def plot_solve_rate_vs_complexity(
    dfs: dict[str, pd.DataFrame],
    output_path: str | Path,
    levels: list[int] = [1, 2, 3, 4, 5],
) -> None:
    """
    Figure 1: Solve rate × complexity level curves per agent type.
    """
    set_paper_style()
    fig, ax = plt.subplots()

    markers = ["o", "s", "^", "D", "v"]
    colors = sns.color_palette("colorblind", len(dfs))

    for i, (agent_name, df) in enumerate(dfs.items()):
        means, cis, xs = [], [], []
        for level in levels:
            level_df = df[df["complexity_level"] == level]
            if level_df.empty:
                continue
            sr = level_df["solved"].mean()
            n = len(level_df)
            ci = 1.96 * np.sqrt(sr * (1 - sr) / max(n, 1))
            means.append(sr)
            cis.append(ci)
            xs.append(level)
        ax.errorbar(xs, means, yerr=cis, marker=markers[i % len(markers)],
                     color=colors[i], label=agent_name, linewidth=2, capsize=4)

    ax.set_xlabel("Complexity Level")
    ax.set_ylabel("Solve Rate")
    ax.set_title("Solve Rate vs. Complexity Level")
    ax.set_xticks(levels)
    ax.set_xticklabels(["Trivial", "Easy", "Medium", "Hard", "Expert"])
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_belief_accuracy_trajectory(
    results: dict[str, list[list[float]]],  # agent -> list of belief traces
    output_path: str | Path,
) -> None:
    """
    Figure 2: Mean belief accuracy over steps (how quickly agents converge).
    """
    set_paper_style()
    fig, ax = plt.subplots()

    colors = sns.color_palette("colorblind", len(results))
    for i, (agent_name, traces) in enumerate(results.items()):
        # Pad traces to same length
        max_len = max(len(t) for t in traces) if traces else 0
        padded = np.full((len(traces), max_len), np.nan)
        for j, t in enumerate(traces):
            padded[j, :len(t)] = t
        mean_trace = np.nanmean(padded, axis=0)
        std_trace = np.nanstd(padded, axis=0)
        steps = np.arange(max_len)
        ax.plot(steps, mean_trace, color=colors[i], label=agent_name, linewidth=2)
        ax.fill_between(steps, mean_trace - std_trace, mean_trace + std_trace,
                         alpha=0.2, color=colors[i])

    ax.set_xlabel("Step")
    ax.set_ylabel("Belief Accuracy (P(true culprit))")
    ax.set_title("Belief Accuracy Trajectory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_clue_efficiency_bar(
    dfs: dict[str, pd.DataFrame],
    output_path: str | Path,
    levels: list[int] = [1, 2, 3, 4, 5],
) -> None:
    """
    Figure 3: Clue efficiency grouped bar chart.
    """
    set_paper_style()
    fig, ax = plt.subplots()

    n_agents = len(dfs)
    bar_width = 0.8 / n_agents
    level_labels = ["Trivial", "Easy", "Medium", "Hard", "Expert"]
    colors = sns.color_palette("colorblind", n_agents)

    for i, (agent_name, df) in enumerate(dfs.items()):
        means = []
        for level in levels:
            level_df = df[df["complexity_level"] == level]
            means.append(level_df["clue_efficiency"].mean() if not level_df.empty else 0)
        x = np.arange(len(levels)) + i * bar_width
        ax.bar(x, means, bar_width, label=agent_name, color=colors[i])

    ax.set_xlabel("Complexity Level")
    ax.set_ylabel("Clue Efficiency")
    ax.set_title("Evidence Discovery Efficiency")
    ax.set_xticks(np.arange(len(levels)) + bar_width * (n_agents - 1) / 2)
    ax.set_xticklabels(level_labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_token_cost(
    dfs: dict[str, pd.DataFrame],
    output_path: str | Path,
    levels: list[int] = [1, 2, 3, 4, 5],
) -> None:
    """
    Figure 4: Token cost scaling across complexity.
    """
    set_paper_style()
    fig, ax = plt.subplots()

    markers = ["o", "s", "^"]
    colors = sns.color_palette("colorblind", len(dfs))
    level_labels = ["Trivial", "Easy", "Medium", "Hard", "Expert"]

    for i, (agent_name, df) in enumerate(dfs.items()):
        means, stds = [], []
        for level in levels:
            level_df = df[df["complexity_level"] == level]
            means.append(level_df["total_tokens"].mean() if not level_df.empty else 0)
            stds.append(level_df["total_tokens"].std() if not level_df.empty else 0)
        ax.errorbar(levels, means, yerr=stds, marker=markers[i % len(markers)],
                     color=colors[i], label=agent_name, linewidth=2, capsize=4)

    ax.set_xlabel("Complexity Level")
    ax.set_ylabel("Total Tokens")
    ax.set_title("Token Cost Scaling")
    ax.set_xticks(levels)
    ax.set_xticklabels(level_labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(
    results_dirs: dict[str, str | Path],
    output_dir: str | Path,
) -> None:
    """
    Generate all paper figures.

    Parameters
    ----------
    results_dirs : dict[str, Path]
        Agent name → results directory.
    output_dir : Path
        Where to save figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dfs = {}
    for name, rdir in results_dirs.items():
        dfs[name] = load_results(rdir)

    plot_solve_rate_vs_complexity(dfs, output_dir / "fig1_solve_rate.pdf")
    plot_clue_efficiency_bar(dfs, output_dir / "fig3_clue_efficiency.pdf")
    plot_token_cost(dfs, output_dir / "fig4_token_cost.pdf")

    # Belief trajectory requires loading full episode results
    belief_traces: dict[str, list[list[float]]] = {}
    for name, rdir in results_dirs.items():
        rdir = Path(rdir)
        traces = []
        for f in rdir.glob("episode_*.json"):
            ep = json.loads(f.read_text())
            bt = ep.get("belief_trace", [])
            trace = [s.get("suspect_probs", {}).get(list(s.get("suspect_probs", {}).keys())[0], 0) if s.get("suspect_probs") else 0 for s in bt]
            traces.append(trace)
        belief_traces[name] = traces
    plot_belief_accuracy_trajectory(belief_traces, output_dir / "fig2_belief_accuracy.pdf")

    # Table
    table_df = compute_aggregate_table(dfs)
    table_df.to_csv(output_dir / "table2_results.csv", index=False)
    print(f"All figures saved to {output_dir}")
