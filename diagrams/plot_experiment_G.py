"""
Experiment G – Line plots
G1: Battery Factor vs Number of Clusters / Sorties
G2: Battery Factor vs Single-Charge Energy Used (%)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "results",
                         "experiment_G_extreme_constraint_stress.csv")
OUT_DIR = os.path.dirname(__file__)

# ── palette / style ────────────────────────────────────────────────────────────
PROFILES = ["low", "medium", "high"]
PROFILE_COLORS = {"low": "#1f77b4", "medium": "#ff7f0e", "high": "#d62728"}
PROFILE_LABELS = {"low": "Low", "medium": "Medium", "high": "High"}
MARKERS = {"low": "o", "medium": "s", "high": "^"}

BATTERY_LABELS = {0.50: "0.50×\n(50%)", 0.75: "0.75×\n(75%)", 1.00: "1.00×\n(100%)"}


def spine_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.35, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def load_agg(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    agg = (
        df.groupby(["usable_capacity_factor", "coverage_time_profile"])
        .agg(
            num_clusters=("num_clusters", "mean"),
            sortie_count=("sortie_count", "mean"),
            single_charge_energy_used_percent=("single_charge_energy_used_percent", "mean"),
        )
        .reset_index()
    )
    return agg


# ── Plot G1: Clusters / Sorties ────────────────────────────────────────────────
def plot_G1(agg: pd.DataFrame, out_path: str):
    battery_factors = sorted(agg["usable_capacity_factor"].unique())
    x = np.arange(len(battery_factors))
    xlabels = [BATTERY_LABELS[b] for b in battery_factors]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(
        "Battery Factor vs Number of Clusters / Sorties",
        fontsize=13, fontweight="bold",
    )

    sub_titles = [
        ("num_clusters",  "Avg Number of Clusters"),
        ("sortie_count",  "Avg Number of Sorties"),
    ]

    for ax, (metric, subtitle) in zip(axes, sub_titles):
        for profile in PROFILES:
            sub = agg[agg["coverage_time_profile"] == profile].set_index("usable_capacity_factor")
            vals = [sub.loc[b, metric] for b in battery_factors]
            ax.plot(
                x, vals,
                color=PROFILE_COLORS[profile],
                marker=MARKERS[profile],
                linewidth=2.2,
                markersize=8,
                label=f"{PROFILE_LABELS[profile]} coverage",
                zorder=3,
            )
            # annotate each point
            for xi, v in zip(x, vals):
                ax.annotate(
                    f"{v:.1f}",
                    xy=(xi, v),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center", fontsize=8,
                    color=PROFILE_COLORS[profile],
                )

        ax.set_title(subtitle, fontsize=10, fontweight="bold", pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_xlabel("Battery Capacity Factor", fontsize=10)
        ax.set_ylabel(subtitle, fontsize=10)
        spine_style(ax)

        # arrow annotation to reinforce narrative (only on clusters subplot)
        if metric == "num_clusters":
            ax.annotate(
                "← Lower battery\n   more clusters\n   (smaller subproblems)",
                xy=(0, ax.get_ylim()[0]),
                xytext=(0.03, 0.18),
                textcoords="axes fraction",
                fontsize=8, color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#cccccc", lw=0.8),
            )

    # shared legend
    handles = [
        mlines.Line2D([], [], color=PROFILE_COLORS[p], marker=MARKERS[p],
                      linewidth=2, markersize=8, label=f"{PROFILE_LABELS[p]} coverage")
        for p in PROFILES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        title="Coverage Time Profile",
        title_fontsize=9,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Plot G2: Single-charge Energy % ────────────────────────────────────────────
def plot_G2(agg: pd.DataFrame, out_path: str):
    battery_factors = sorted(agg["usable_capacity_factor"].unique())
    x = np.arange(len(battery_factors))
    xlabels = [BATTERY_LABELS[b] for b in battery_factors]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.suptitle(
        "Battery Factor vs Single-Charge Energy Used (%)",
        fontsize=13, fontweight="bold",
    )

    for profile in PROFILES:
        sub = agg[agg["coverage_time_profile"] == profile].set_index("usable_capacity_factor")
        vals = [sub.loc[b, "single_charge_energy_used_percent"] for b in battery_factors]
        ax.plot(
            x, vals,
            color=PROFILE_COLORS[profile],
            marker=MARKERS[profile],
            linewidth=2.2,
            markersize=8,
            label=f"{PROFILE_LABELS[profile]} coverage",
            zorder=3,
        )
        for xi, v in zip(x, vals):
            offset = 10 if v < 200 else -16
            ax.annotate(
                f"{v:.1f}%",
                xy=(xi, v),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center", fontsize=8.5,
                color=PROFILE_COLORS[profile],
            )

    # 100% reference line
    ax.axhline(100, color="#888888", linewidth=1.2, linestyle="--", zorder=2)
    ax.annotate(
        "100% = one full charge",
        xy=(len(x) - 1, 100),
        xytext=(-6, 5),
        textcoords="offset points",
        ha="right", fontsize=8, color="#666666",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_xlabel("Battery Capacity Factor", fontsize=10)
    ax.set_ylabel("Single-Charge Energy Used (%)", fontsize=10)
    spine_style(ax)

    ax.legend(
        loc="upper right",
        fontsize=9,
        title="Coverage Time Profile",
        title_fontsize=9,
        frameon=True,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agg = load_agg(DATA_PATH)

    g1_path = os.path.join(OUT_DIR, "experiment_G1_battery_vs_clusters.png")
    g2_path = os.path.join(OUT_DIR, "experiment_G2_battery_vs_energy_percent.png")

    print("Generating G1 …")
    plot_G1(agg, g1_path)
    print("Generating G2 …")
    plot_G2(agg, g2_path)
    print("Done.")
