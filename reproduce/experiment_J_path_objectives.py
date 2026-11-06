"""
Experiment J: Objective Path Comparison

Single configuration, three objectives: energy, coverage, two-stage hybrid.
Produces a side-by-side visualisation of UAV paths on the Voronoi map for each
objective so the routing trade-offs are immediately visible.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from dummy_app.designs.agents import generate_agent_colormap
from dummy_app.experiments.metrics import aggregate_agent_routes
from dummy_app.experiments.runner_utils import create_scenario, run_method
from dummy_app.program_config import (
    EXPERIMENT_DEFAULT_ENV,
    EXPERIMENT_DEFAULT_MEMORY_LIMIT,
    EXPERIMENT_DEFAULT_SCENARIO,
    EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
    EXPERIMENT_RESULTS_DIR,
)


FIXED_CONFIG = {
    "node_count": 100,
    "uav_count": 5,
    "coverage_time_profile": "medium",
    "seed": 42,
}

OBJECTIVES = [
    {
        "label": "Energy",
        "objective_function": "energy",
        "stage_solution": 1,
        "color": "#e74c3c",
    },
    {
        "label": "Coverage",
        "objective_function": "coverage",
        "stage_solution": 1,
        "color": "#2ecc71",
    },
    {
        "label": "Two-Stage\nHybrid",
        "objective_function": "coverage",
        "stage_solution": 2,
        "color": "#3498db",
    },
]


def _ordered_node_sequence(route: List[Tuple[int, int, int]]) -> List[int]:
    """Return the ordered list of nodes visited (excluding hover self-loops)."""
    nodes: List[int] = []
    for source, target, _ in route:
        if not nodes:
            nodes.append(int(source))
        if int(target) != int(source):
            nodes.append(int(target))
    return nodes


def _draw_route_on_ax(
    ax: Axes,
    routes: Dict[int, List[Tuple[int, int, int]]],
    centroids: List[Tuple[float, float]],
    depot_ids: List[int],
    agent_colors: List[Any],
    path_colors: List[Any],
) -> None:
    depot_set = set(int(d) for d in depot_ids)

    for uav_idx, (_agent_id, route) in enumerate(sorted(routes.items())):
        color = agent_colors[uav_idx % len(agent_colors)]
        path_col = path_colors[uav_idx % len(path_colors)]
        nodes = _ordered_node_sequence(route)
        if len(nodes) < 2:
            continue

        xs = [centroids[n][0] for n in nodes]
        ys = [centroids[n][1] for n in nodes]

        ax.plot(xs, ys, color=path_col, linewidth=1.8, zorder=3, alpha=0.85)

        for i in range(len(nodes) - 1):
            dx = centroids[nodes[i + 1]][0] - centroids[nodes[i]][0]
            dy = centroids[nodes[i + 1]][1] - centroids[nodes[i]][1]
            mx = (centroids[nodes[i]][0] + centroids[nodes[i + 1]][0]) / 2.0
            my = (centroids[nodes[i]][1] + centroids[nodes[i + 1]][1]) / 2.0
            ax.annotate(
                "",
                xy=(mx + dx * 0.01, my + dy * 0.01),
                xytext=(mx - dx * 0.01, my - dy * 0.01),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2),
                zorder=4,
            )

        ax.scatter(
            centroids[nodes[0]][0],
            centroids[nodes[0]][1],
            color=color,
            s=90,
            zorder=6,
            edgecolors="white",
            linewidths=1.0,
        )

    all_visited: set[int] = set()
    for route in routes.values():
        for _, target, _ in route:
            all_visited.add(int(target))

    for node_id, (cx, cy) in enumerate(centroids):
        is_depot = node_id in depot_set
        if is_depot:
            ax.scatter(cx, cy, marker="*", s=260, color="#f39c12", zorder=7, edgecolors="black", linewidths=0.8)
        else:
            visited = node_id in all_visited
            ax.scatter(
                cx,
                cy,
                marker="o",
                s=35 if visited else 28,
                color="#2c3e50" if visited else "#bdc3c7",
                zorder=5,
                edgecolors="white" if visited else "grey",
                linewidths=0.5,
            )
        ax.text(cx, cy + 0.0005, str(node_id), fontsize=5.5, ha="center", va="bottom", color="#555", zorder=8)


def _draw_regions(ax: Axes, regions: list, alpha: float = 0.12) -> None:
    palette = plt.cm.Pastel1.colors  # type: ignore[attr-defined]
    for idx, poly in enumerate(regions):
        try:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=alpha, color=palette[idx % len(palette)], zorder=1)
            ax.plot(x, y, color="#aaa", linewidth=0.5, zorder=2)
        except Exception:
            pass


def _metrics_text(metrics: Dict[str, Any]) -> str:
    cov = metrics.get("coverage_ratio")
    energy = metrics.get("total_energy")
    dist = metrics.get("total_distance")
    gap = metrics.get("optimality_gap_percent")
    parts = []
    if cov is not None:
        parts.append(f"Cov {cov:.0%}")
    if energy is not None:
        parts.append(f"E {energy:.1f} Wh")
    if dist is not None:
        parts.append(f"D {dist:.2f} km")
    if gap is not None:
        parts.append(f"Gap {gap:.1f}%")
    return "  |  ".join(parts)


def run_experiment(output_dir: Path) -> None:
    print("Creating scenario …")
    scenario_payload = create_scenario(
        node_count=FIXED_CONFIG["node_count"],
        uav_count=FIXED_CONFIG["uav_count"],
        coverage_time_profile=FIXED_CONFIG["coverage_time_profile"],
        seed=FIXED_CONFIG["seed"],
        scenario_name=EXPERIMENT_DEFAULT_SCENARIO,
        objective_function="coverage",
        env_type=EXPERIMENT_DEFAULT_ENV,
    )

    n_agents = FIXED_CONFIG["uav_count"]
    agent_colors, path_colors = generate_agent_colormap(n_agents)

    results: list[dict] = []
    for obj_cfg in OBJECTIVES:
        label = obj_cfg["label"].replace("\n", " ")
        print(f"  Solving: {label} …")
        scenario_payload["objective_function"] = obj_cfg["objective_function"]
        result = run_method(
            scenario_payload,
            "MILP",
            model_name="milp",
            stage_solution=obj_cfg["stage_solution"],
            time_limit_seconds=EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
            memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
        )
        results.append(result)
        status_msg = result["status"]
        metrics = result.get("metrics", {})
        cov = metrics.get("coverage_ratio", 0.0)
        print(f"    status={status_msg}  coverage={cov:.0%}")

    _plot_results(scenario_payload, results, agent_colors, path_colors, output_dir)


def _plot_results(
    scenario_payload: Dict[str, Any],
    results: list[dict],
    agent_colors: list,
    path_colors: list,
    output_dir: Path,
) -> None:
    centroids = scenario_payload["centroids"]
    regions = scenario_payload["regions"]
    depot_ids = list(scenario_payload["depot_ids"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor("#f8f9fa")

    all_x = [c[0] for c in centroids]
    all_y = [c[1] for c in centroids]
    margin_x = (max(all_x) - min(all_x)) * 0.08
    margin_y = (max(all_y) - min(all_y)) * 0.08
    xlim = (min(all_x) - margin_x, max(all_x) + margin_x)
    ylim = (min(all_y) - margin_y, max(all_y) + margin_y)

    for ax, obj_cfg, result in zip(axes, OBJECTIVES, results):
        ax.set_facecolor("#fdfdfd")

        _draw_regions(ax, regions)
        run_result = result.get("run_result")
        metrics = result.get("metrics", {})

        if run_result is not None:
            routes = aggregate_agent_routes(run_result.cluster_results)
            _draw_route_on_ax(ax, routes, centroids, depot_ids, agent_colors, path_colors)
        else:
            ax.text(
                0.5, 0.5, "No solution",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="red",
            )

        label = obj_cfg["label"]
        ax.set_title(
            label,
            fontsize=15, fontweight="bold", pad=10,
            color=obj_cfg["color"],
        )

        metrics_str = _metrics_text(metrics) if result["status"] == "ok" else f"Failed: {result.get('error_message','')[:60]}"
        ax.set_xlabel(metrics_str, fontsize=8.5, labelpad=6, color="#333")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_edgecolor("#ccc")

    legend_elements = [
        Line2D([0], [0], color=path_colors[i], linewidth=2.5, label=f"UAV {i + 1}")
        for i in range(FIXED_CONFIG["uav_count"])
    ] + [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#f39c12",
               markersize=11, label="Depot"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2c3e50",
               markersize=7, label="Visited node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bdc3c7",
               markersize=7, label="Unvisited node"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(legend_elements),
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    seed = FIXED_CONFIG["seed"]
    n = FIXED_CONFIG["node_count"]
    k = FIXED_CONFIG["uav_count"]
    ct = FIXED_CONFIG["coverage_time_profile"]
    fig.suptitle(
        f"UAV Route Comparison by Objective  —  {n} areas · {k} UAVs · seed={seed} · coverage={ct}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "experiment_J_path_objectives.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nFigure saved to: {out_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment J: objective path comparison visualization.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(EXPERIMENT_RESULTS_DIR) / "experiment_J"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(Path(args.output_dir))


if __name__ == "__main__":
    main()
