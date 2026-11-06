from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import pvariance
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate solver run artifacts into comparison CSVs and figures.")
    parser.add_argument("--manifest", required=True, help="CSV manifest created by compare_solver_baselines.sh")
    parser.add_argument("--output_dir", required=True, help="Directory where CSVs and figures will be written.")
    parser.add_argument("--num_agents", type=int, default=0, help="Optional override for the number of UAVs.")
    return parser.parse_args()


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_playback_frames(path: Path) -> Dict[str, List[Dict[str, float]]]:
    grouped: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped[str(row["agent"])].append(
                {
                    "time_step": int(float(row["time_step"])),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                }
            )

    for frames in grouped.values():
        frames.sort(key=lambda item: item["time_step"])
    return grouped


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_variance(values: Iterable[float]) -> float:
    series = [float(value) for value in values]
    if not series:
        return 0.0
    return float(pvariance(series))


def build_agent_ids(instance_payload: Dict[str, Any], override_num_agents: int) -> List[str]:
    number_of_agents = int(override_num_agents or instance_payload.get("number_of_agents", 0))
    return [str(agent_id) for agent_id in range(1, number_of_agents + 1)]


def aggregate_run_metrics(
    run_result: Dict[str, Any],
    instance_payload: Dict[str, Any],
    label: str,
    scenario: str,
    model_name: str,
    num_agents_override: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    summary = dict(run_result.get("summary", {}))
    depots = {int(node_id) for node_id in instance_payload.get("depots", [])}
    agent_ids = build_agent_ids(instance_payload, num_agents_override)

    per_agent: Dict[str, Dict[str, Any]] = {
        agent_id: {"distance": 0.0, "energy": 0.0, "nodes": set()}
        for agent_id in agent_ids
    }

    for cluster_result in run_result.get("cluster_results", []):
        for agent_id, metrics in cluster_result.get("agent_metrics", {}).items():
            per_agent.setdefault(str(agent_id), {"distance": 0.0, "energy": 0.0, "nodes": set()})
            per_agent[str(agent_id)]["distance"] += float(metrics.get("distance", 0.0))
            per_agent[str(agent_id)]["energy"] += float(metrics.get("energy", 0.0))
            for node_id in metrics.get("visited_nodes", []):
                node_int = int(node_id)
                if node_int not in depots:
                    per_agent[str(agent_id)]["nodes"].add(node_int)

    for agent_id in per_agent:
        per_agent[agent_id]["node_count"] = len(per_agent[agent_id]["nodes"])

    distance_values = [float(per_agent[agent_id]["distance"]) for agent_id in agent_ids]
    energy_values = [float(per_agent[agent_id]["energy"]) for agent_id in agent_ids]
    node_counts = [int(per_agent[agent_id]["node_count"]) for agent_id in agent_ids]
    max_route_distance = max(distance_values, default=0.0)
    coverage_ratio = float(summary.get("coverage_ratio", summary.get("node_coverage_ratio", 0.0)))
    runtime_seconds = float(run_result.get("elapsed_time_seconds", summary.get("solve_time_seconds", 0.0)) or 0.0)

    summary_row = {
        "label": label,
        "scenario": scenario,
        "model_name": model_name,
        "total_distance": float(summary.get("distance", 0.0)),
        "max_route_distance": max_route_distance,
        "route_variance": safe_variance(distance_values),
        "energy_variance": safe_variance(energy_values),
        "coverage_percent": coverage_ratio * 100.0,
        "runtime_seconds": runtime_seconds,
        "num_agents": len(agent_ids),
        "node_distribution_per_uav": json.dumps({agent_id: per_agent[agent_id]["node_count"] for agent_id in agent_ids}, sort_keys=True),
        "distance_distribution_per_uav": json.dumps({agent_id: per_agent[agent_id]["distance"] for agent_id in agent_ids}, sort_keys=True),
    }

    per_uav_rows: List[Dict[str, Any]] = []
    for agent_id in agent_ids:
        per_uav_rows.append(
            {
                "label": label,
                "scenario": scenario,
                "model_name": model_name,
                "agent_id": agent_id,
                "distance": float(per_agent[agent_id]["distance"]),
                "energy": float(per_agent[agent_id]["energy"]),
                "node_count": int(per_agent[agent_id]["node_count"]),
            }
        )

    return summary_row, per_uav_rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def create_table_figure(summary_rows: List[Dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    metric_columns = [
        ("label", "Solver"),
        ("total_distance", "Total distance"),
        ("max_route_distance", "Max route"),
        ("route_variance", "Route variance"),
        ("energy_variance", "Energy variance"),
        ("coverage_percent", "Coverage %"),
        ("runtime_seconds", "Runtime (s)"),
    ]
    table_data = [
        [format_metric(row[column]) for column, _ in metric_columns]
        for row in summary_rows
    ]

    fig_height = max(2.8, 1.0 + 0.5 * len(table_data))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=[header for _, header in metric_columns],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    ax.set_title("Solver Comparison Table", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_scalar_bar_figure(summary_rows: List[Dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["label"] for row in summary_rows]
    metric_specs = [
        ("total_distance", "Total distance"),
        ("max_route_distance", "Max route distance"),
        ("route_variance", "Route variance"),
        ("energy_variance", "Energy variance"),
        ("coverage_percent", "Coverage %"),
        ("runtime_seconds", "Runtime (s)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for axis, (column, title) in zip(axes.flat, metric_specs):
        values = [float(row[column]) for row in summary_rows]
        axis.bar(labels, values, color="#3B82F6")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("Solver Metric Comparison", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_workload_balance_figure(per_uav_rows: List[Dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    solver_labels = sorted({row["label"] for row in per_uav_rows})
    agent_ids = sorted({row["agent_id"] for row in per_uav_rows}, key=lambda item: int(item))
    color_map = {
        label: color
        for label, color in zip(solver_labels, ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED", "#0F766E"])
    }

    distance_lookup = {(row["label"], row["agent_id"]): float(row["distance"]) for row in per_uav_rows}
    node_lookup = {(row["label"], row["agent_id"]): int(row["node_count"]) for row in per_uav_rows}

    base_positions = np.arange(len(agent_ids))
    width = 0.18 if solver_labels else 0.18
    offset_origin = (len(solver_labels) - 1) / 2.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for index, label in enumerate(solver_labels):
        offset = (index - offset_origin) * width
        axes[0].bar(
            base_positions + offset,
            [node_lookup.get((label, agent_id), 0) for agent_id in agent_ids],
            width=width,
            label=label,
            color=color_map[label],
        )
        axes[1].bar(
            base_positions + offset,
            [distance_lookup.get((label, agent_id), 0.0) for agent_id in agent_ids],
            width=width,
            label=label,
            color=color_map[label],
        )

    axes[0].set_ylabel("Nodes served")
    axes[0].set_title("Node Distribution per UAV")
    axes[0].grid(axis="y", linestyle=":", alpha=0.4)
    axes[1].set_ylabel("Distance")
    axes[1].set_title("Route Distance per UAV")
    axes[1].grid(axis="y", linestyle=":", alpha=0.4)
    axes[1].set_xticks(base_positions)
    axes[1].set_xticklabels([f"UAV {agent_id}" for agent_id in agent_ids])
    axes[0].legend(ncols=min(4, max(1, len(solver_labels))), loc="upper right")

    fig.suptitle("Workload Balance by Solver", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_paths_figure(playback_by_solver: List[Dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not playback_by_solver:
        return

    total_plots = len(playback_by_solver)
    ncols = 2 if total_plots > 1 else 1
    nrows = (total_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), squeeze=False)

    all_x = []
    all_y = []
    for solver_payload in playback_by_solver:
        for agent_frames in solver_payload["frames"].values():
            all_x.extend(frame["x"] for frame in agent_frames)
            all_y.extend(frame["y"] for frame in agent_frames)

    if not all_x or not all_y:
        return

    x_pad = max((max(all_x) - min(all_x)) * 0.05, 1e-6)
    y_pad = max((max(all_y) - min(all_y)) * 0.05, 1e-6)

    color_cycle = ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED", "#0F766E", "#DB2777"]
    for index, solver_payload in enumerate(playback_by_solver):
        axis = axes[index // ncols][index % ncols]
        axis.set_title(solver_payload["label"])
        for color_index, (agent_id, frames) in enumerate(sorted(solver_payload["frames"].items(), key=lambda item: int(item[0]))):
            xs = [frame["x"] for frame in frames]
            ys = [frame["y"] for frame in frames]
            if not xs or not ys:
                continue
            axis.plot(xs, ys, color=color_cycle[color_index % len(color_cycle)], linewidth=1.8, label=f"UAV {agent_id}")
            axis.scatter(xs[0], ys[0], color=color_cycle[color_index % len(color_cycle)], marker="o", s=18)
            axis.scatter(xs[-1], ys[-1], color=color_cycle[color_index % len(color_cycle)], marker="x", s=24)

        axis.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
        axis.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.grid(True, linestyle=":", alpha=0.35)
        axis.legend(loc="best", fontsize=8)

    for empty_index in range(total_plots, nrows * ncols):
        axes[empty_index // ncols][empty_index % ncols].axis("off")

    fig.suptitle("Distinct Solver Paths", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    manifest_rows = load_manifest(manifest_path)
    successful_rows = [row for row in manifest_rows if row.get("status") == "success" and row.get("artifact_dir")]
    if not successful_rows:
        raise SystemExit("No successful runs found in the manifest.")

    summary_rows: List[Dict[str, Any]] = []
    per_uav_rows: List[Dict[str, Any]] = []
    playback_by_solver: List[Dict[str, Any]] = []

    for manifest_row in successful_rows:
        artifact_dir = Path(manifest_row["artifact_dir"])
        run_result = load_json(artifact_dir / "run_result.json")
        instance_payload = load_json(artifact_dir / "instance.json")
        summary_row, uav_rows = aggregate_run_metrics(
            run_result=run_result,
            instance_payload=instance_payload,
            label=manifest_row["label"],
            scenario=manifest_row["scenario"],
            model_name=manifest_row["model_name"],
            num_agents_override=args.num_agents,
        )
        summary_rows.append(summary_row)
        per_uav_rows.extend(uav_rows)
        playback_by_solver.append(
            {
                "label": manifest_row["label"],
                "frames": read_playback_frames(artifact_dir / "playback_frames.csv"),
            }
        )

    summary_fieldnames = [
        "label",
        "scenario",
        "model_name",
        "total_distance",
        "max_route_distance",
        "route_variance",
        "energy_variance",
        "coverage_percent",
        "runtime_seconds",
        "num_agents",
        "node_distribution_per_uav",
        "distance_distribution_per_uav",
    ]
    per_uav_fieldnames = [
        "label",
        "scenario",
        "model_name",
        "agent_id",
        "distance",
        "energy",
        "node_count",
    ]

    write_csv(output_dir / "solver_summary.csv", summary_rows, summary_fieldnames)
    write_csv(output_dir / "per_uav_metrics.csv", per_uav_rows, per_uav_fieldnames)
    create_table_figure(summary_rows, output_dir / "solver_comparison_table.png")
    create_scalar_bar_figure(summary_rows, output_dir / "solver_metric_bars.png")
    create_workload_balance_figure(per_uav_rows, output_dir / "workload_balance.png")
    create_paths_figure(playback_by_solver, output_dir / "solver_paths.png")


if __name__ == "__main__":
    main()
