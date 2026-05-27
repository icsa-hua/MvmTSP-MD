from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate experiment plots from result CSV files.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing experiment CSV files.")
    parser.add_argument("--output_dir", type=str, default="results/plots", help="Directory where plots will be saved.")
    return parser.parse_args()


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_line_plot(df: pd.DataFrame, x: str, y: str, hue: str, title: str, output_path: Path) -> None:
    if x not in df.columns or y not in df.columns or hue not in df.columns:
        return
    plt.figure(figsize=(10, 6))
    for label, group in df.groupby(hue):
        grouped = group.groupby(x, dropna=False)[y].mean().reset_index()
        plt.plot(grouped[x], grouped[y], marker="o", label=str(label))
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def _save_bar_plot(df: pd.DataFrame, x: str, y: str, title: str, output_path: Path) -> None:
    if x not in df.columns or y not in df.columns:
        return
    grouped = df.groupby(x, dropna=False)[y].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.bar(grouped[x].astype(str), grouped[y], color="#2563EB")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def _save_distance_per_uav_plot(df: pd.DataFrame, output_path: Path) -> None:
    if "distance_per_uav" not in df.columns or "method_name" not in df.columns:
        return
    rows = []
    for _, row in df.iterrows():
        try:
            payload = json.loads(row["distance_per_uav"])
        except Exception:
            continue
        for agent_id, distance in payload.items():
            rows.append({"method_name": row["method_name"], "agent_id": str(agent_id), "distance": float(distance)})
    if not rows:
        return
    distance_df = pd.DataFrame(rows)
    pivot = distance_df.groupby(["method_name", "agent_id"], dropna=False)["distance"].mean().unstack(fill_value=0.0)
    pivot.plot(kind="bar", figsize=(12, 6))
    plt.title("Distance per UAV")
    plt.xlabel("Method")
    plt.ylabel("Average distance")
    plt.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    _ensure_output_dir(output_dir)

    experiment_a = _load_csv(results_dir / "experiment_A_main_comparison.csv")
    if experiment_a is not None and not experiment_a.empty:
        _save_line_plot(
            experiment_a,
            x="node_count",
            y="runtime_sec",
            hue="method_name",
            title="Runtime vs Node Count",
            output_path=output_dir / "runtime_vs_node_count.png",
        )
        _save_line_plot(
            experiment_a,
            x="node_count",
            y="total_distance",
            hue="method_name",
            title="Total Distance Comparison",
            output_path=output_dir / "total_distance_comparison.png",
        )
        _save_line_plot(
            experiment_a,
            x="node_count",
            y="energy_std_across_uavs",
            hue="method_name",
            title="Energy Variance Comparison",
            output_path=output_dir / "energy_variance_comparison.png",
        )
        _save_line_plot(
            experiment_a,
            x="node_count",
            y="coverage_ratio",
            hue="method_name",
            title="Coverage Percentage Comparison",
            output_path=output_dir / "coverage_percentage_comparison.png",
        )
        _save_distance_per_uav_plot(experiment_a, output_dir / "distance_per_uav_bar_plot.png")

    experiment_b = _load_csv(results_dir / "experiment_B_solver_comparison.csv")
    if experiment_b is not None and not experiment_b.empty:
        _save_line_plot(
            experiment_b,
            x="node_count",
            y="optimality_gap_percent",
            hue="solver_name",
            title="Optimality Gap Comparison",
            output_path=output_dir / "optimality_gap_comparison.png",
        )

    experiment_c = _load_csv(results_dir / "experiment_C_warm_start.csv")
    if experiment_c is not None and not experiment_c.empty:
        _save_line_plot(
            experiment_c,
            x="node_count",
            y="gap_reduction_vs_no_warm_start",
            hue="warm_start_method",
            title="Warm-start Gap Reduction",
            output_path=output_dir / "warm_start_gap_reduction.png",
        )


if __name__ == "__main__":
    main()
