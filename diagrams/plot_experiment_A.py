"""
Experiment A – Grouped Bar Diagrams
A1: Solution Quality  (coverage, distance, energy, max route distance, energy fairness)
A2: Computational Cost (runtime + objective gap vs MILP optimal)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import LogLocator, LogFormatter

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "results/Archive", "experiment_A_main_comparison.csv")
OUT_DIR = os.path.dirname(__file__)

# ── palette ───────────────────────────────────────────────────────────────────
METHODS = ["MILP", "Greedy NN", "Static Partitioning + Greedy NN", "Genetic Algorithm", "ALNS"]
METHOD_LABELS = ["MILP", "Greedy NN", "SP + Greedy NN", "Genetic Alg.", "ALNS"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
METHOD_COLOR = dict(zip(METHODS, COLORS))
METHOD_LABEL = dict(zip(METHODS, METHOD_LABELS))

BAR_W = 0.15
N_METHODS = len(METHODS)
OFFSETS = np.linspace(-(N_METHODS - 1) / 2 * BAR_W, (N_METHODS - 1) / 2 * BAR_W, N_METHODS)

SPINE_STYLE = dict(top=False, right=False)


def load_agg(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    agg = (
        df.groupby(["node_count", "method_name"])
        .agg(
            coverage_ratio=("coverage_ratio", "mean"),
            total_distance=("total_distance", "mean"),
            total_energy=("total_energy", "mean"),
            max_route_distance=("max_route_distance", "mean"),
            energy_fairness=("jain_fairness_energy", "mean"),
            runtime_sec=("runtime_sec", "mean"),
            objective_value=("objective_value", "mean"),
        )
        .reset_index()
    )
    return agg


def bar_values(agg: pd.DataFrame, node_counts, method: str, metric: str) -> list:
    rows = agg[agg["method_name"] == method].set_index("node_count")
    return [rows.loc[nc, metric] if nc in rows.index else np.nan for nc in node_counts]


def style_ax(ax, xlabel="Number of Nodes", ylabel="", title=""):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.35, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    for spine, vis in SPINE_STYLE.items():
        ax.spines[spine].set_visible(vis)
    ax.tick_params(labelsize=8)


def add_legend(fig, y=-0.03):
    handles = [
        mpatches.Patch(color=METHOD_COLOR[m], label=METHOD_LABEL[m], alpha=0.88)
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=N_METHODS,
        fontsize=8.5,
        bbox_to_anchor=(0.5, y),
        frameon=True,
        title="Methods",
        title_fontsize=9,
    )


# ── Plot A1: Solution Quality ─────────────────────────────────────────────────
def plot_A1(agg: pd.DataFrame, out_path: str):
    node_counts = sorted(agg["node_count"].unique())
    x = np.arange(len(node_counts))
    xlabels = [str(nc) for nc in node_counts]

    metrics = [
        ("coverage_ratio",     "Coverage Ratio",         "",       None, None),
        ("total_distance",     "Total Distance",         "(km)",   None, None),
        ("total_energy",       "Total Energy",           "(Wh)",   None, None),
        ("max_route_distance", "Max Route Distance",     "(km)",   None, None),
        ("energy_fairness",    "Energy Fairness\n(Jain's Index)", "", 0.85, 1.01),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle(
        "Experiment A  –  Solution Quality Comparison\n"
        "(averaged over seeds, UAV counts, coverage time profiles)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    axes_flat = axes.flatten()

    for idx, (col, title, unit, ymin, ymax) in enumerate(metrics):
        ax = axes_flat[idx]
        for j, method in enumerate(METHODS):
            vals = bar_values(agg, node_counts, method, col)
            ax.bar(
                x + OFFSETS[j], vals, BAR_W,
                color=METHOD_COLOR[method], alpha=0.88,
                edgecolor="white", linewidth=0.4, zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ylabel = f"{title} {unit}".strip()
        style_ax(ax, ylabel=ylabel, title=title + (f" {unit}" if unit else ""))
        if ymin is not None:
            ax.set_ylim(ymin, ymax)

    axes_flat[5].set_visible(False)
    add_legend(fig, y=-0.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Plot A2: Computational Cost ───────────────────────────────────────────────
def plot_A2(agg: pd.DataFrame, out_path: str):
    node_counts = sorted(agg["node_count"].unique())
    x = np.arange(len(node_counts))
    xlabels = [str(nc) for nc in node_counts]

    # Compute objective gap (%) vs MILP optimal for each method/node_count
    milp_obj = agg[agg["method_name"] == "MILP"].set_index("node_count")["objective_value"]

    def obj_gap(method: str) -> list:
        vals = []
        for nc in node_counts:
            row = agg[(agg["node_count"] == nc) & (agg["method_name"] == method)]
            if method == "MILP":
                vals.append(0.0)
            else:
                m_obj = milp_obj.get(nc, np.nan)
                h_obj = row["objective_value"].values[0] if len(row) else np.nan
                gap = (m_obj - h_obj) / m_obj * 100 if (not np.isnan(m_obj) and m_obj > 0) else 0.0
                vals.append(gap)
        return vals

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(
        "Experiment A  –  Computational Cost\n"
        "(averaged over seeds, UAV counts, coverage time profiles)",
        fontsize=13, fontweight="bold",
    )

    # ── subplot 1: Runtime ────────────────────────────────────────────────────
    ax = axes[0]
    for j, method in enumerate(METHODS):
        vals = bar_values(agg, node_counts, method, "runtime_sec")
        ax.bar(
            x + OFFSETS[j], vals, BAR_W,
            color=METHOD_COLOR[method], alpha=0.88,
            edgecolor="white", linewidth=0.4, zorder=3,
        )
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    style_ax(ax, ylabel="Runtime (seconds, log scale)", title="Runtime (log scale)")
    ax.grid(axis="y", which="both", alpha=0.25, linestyle="--", zorder=0)

    # ── subplot 2: Objective Gap vs MILP ─────────────────────────────────────
    ax = axes[1]
    for j, method in enumerate(METHODS):
        vals = obj_gap(method)
        ax.bar(
            x + OFFSETS[j], vals, BAR_W,
            color=METHOD_COLOR[method], alpha=0.88,
            edgecolor="white", linewidth=0.4, zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    style_ax(
        ax,
        ylabel="Objective Gap vs MILP (%)",
        title="Solution Quality Gap vs MILP Optimal\n(MILP gap = 0; all solutions proven optimal)",
    )
    # annotate MILP bar at 0 with a star
    for nc_idx in range(len(node_counts)):
        ax.annotate(
            "★", xy=(x[nc_idx] + OFFSETS[0], 0.003),
            fontsize=7, ha="center", color=METHOD_COLOR["MILP"],
        )

    add_legend(fig, y=-0.06)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agg = load_agg(DATA_PATH)

    a1_path = os.path.join(OUT_DIR, "experiment_A1_solution_quality.png")
    a2_path = os.path.join(OUT_DIR, "experiment_A2_computational_cost.png")

    print("Generating A1 …")
    plot_A1(agg, a1_path)
    print("Generating A2 …")
    plot_A2(agg, a2_path)
    print("Done.")
