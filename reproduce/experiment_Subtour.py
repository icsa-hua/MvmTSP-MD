from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

import matplotlib.pyplot as plt
from tqdm import tqdm

from dummy_app.experiments.runner_utils import create_scenario, format_result_row, run_method, save_results
from dummy_app.program_config import (
    EXPERIMENT_DEFAULT_ENV,
    EXPERIMENT_DEFAULT_MEMORY_LIMIT,
    EXPERIMENT_DEFAULT_OBJECTIVE,
    EXPERIMENT_DEFAULT_SCENARIO,
    EXPERIMENT_DEFAULT_SEEDS,
    EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
    EXPERIMENT_RESULTS_DIR,
    EXPERIMENT_SUBTOUR_AREAS,
    EXPERIMENT_SUBTOUR_MODES,
    EXPERIMENT_SUBTOUR_SOLVER,
    EXPERIMENT_SUBTOUR_UAVS,
)


EMPTY_FIELDS = {
    "solver_backend": EXPERIMENT_SUBTOUR_SOLVER,
    "subtour_mode": "",
    "raw_status": "",
    "normalized_status": "",
    "termination_reason": "",
    "num_variables": None,
    "num_binary_variables": None,
    "num_continuous_variables": None,
    "num_constraints": None,
    "model_build_time_sec": None,
    "solver_runtime_sec": None,
    "objective_value": None,
    "best_bound": None,
    "optimality_gap_percent": None,
    "coverage_ratio": None,
    "total_distance": None,
    "total_energy": None,
    "total_travel_time": None,
    "subtour_cuts_added": None,
    "dfj_iterations": None,
    "dfj_solve_passes": None,
    "violated_subtours_detected": None,
    "dfj_loop_executed": False,
    "dfj_cuts_triggered": False,
    "optimality_proven": False,
    "time_limit_reached": False,
    "time_limit_feasible": False,
    "time_limit_no_solution": False,
    "artifact_dir": "",
    "error_message": "",
    "total_data_rate_mbps": None,
    "data_rate_per_hour_mbps": None,
    "data_rate_per_kwh_mbps": None,
    "avg_data_rate_per_cluster_mbps": None,
    "mean_sinr_db": None,
    "coverage_prob_at_0db": None,
    "coverage_prob_at_10db": None,
    "coverage_prob_at_20db": None,
}


def _derive_solver_status_fields(result: Mapping[str, Any], metrics: Mapping[str, Any]) -> Dict[str, Any]:
    run_result = result.get("run_result")
    raw_status = getattr(run_result, "raw_status", "") if run_result is not None else ""
    normalized_status = getattr(run_result, "normalized_status", "") if run_result is not None else ""
    termination_reason = getattr(run_result, "termination_reason", "") if run_result is not None else ""

    if not raw_status:
        raw_status = str(metrics.get("raw_status", "") or "")
    if not normalized_status:
        normalized_status = str(metrics.get("normalized_status", "") or "")
    if not termination_reason:
        termination_reason = str(metrics.get("termination_reason", "") or "")
    if not termination_reason and result.get("error_message"):
        termination_reason = str(result.get("error_message", ""))

    gap_percent = metrics.get("optimality_gap_percent")
    has_positive_gap = gap_percent is not None and float(gap_percent) > 1e-9
    if bool(metrics.get("time_limit_feasible")) and has_positive_gap:
        raw_status = "Not Solved"
        normalized_status = "feasible_time_limit"
        if not termination_reason:
            termination_reason = "time_limit_or_undefined_solver_stop"
    elif bool(metrics.get("optimality_proven")) and not normalized_status:
        raw_status = "Optimal"
        normalized_status = "optimal"

    return {
        "raw_status": raw_status,
        "normalized_status": normalized_status,
        "termination_reason": termination_reason,
    }


def _derive_dfj_validation_fields(metrics: Mapping[str, Any], subtour_mode: str) -> Dict[str, Any]:
    dfj_solve_passes = metrics.get("dfj_solve_passes")
    violated_subtours_detected = metrics.get("violated_subtours_detected")
    subtour_cuts_added = metrics.get("subtour_cuts_added")
    dfj_iterations = metrics.get("dfj_iterations")

    if str(subtour_mode) != "dfj_iter":
        return {
            "dfj_loop_executed": False,
            "dfj_cuts_triggered": False,
        }

    return {
        "dfj_loop_executed": bool(dfj_solve_passes is not None and int(dfj_solve_passes) > 0),
        "dfj_cuts_triggered": bool(
            (violated_subtours_detected is not None and int(violated_subtours_detected) > 0)
            or (subtour_cuts_added is not None and int(subtour_cuts_added) > 0)
            or (dfj_iterations is not None and int(dfj_iterations) > 0)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment Subtour: compare MILP subtour formulations.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(EXPERIMENT_RESULTS_DIR) / "experiment_Subtour.csv"),
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default=str(Path(EXPERIMENT_RESULTS_DIR) / "experiment_Subtour_figures"),
    )
    parser.add_argument("--append", action="store_true", help="Append to the output CSV instead of replacing it.")
    return parser.parse_args()


def _scenario_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("seed", 0) or 0),
        int(row.get("node_count", 0) or 0),
        int(row.get("uav_count", 0) or 0),
    )


def _read_playback_frames(path: Path) -> Dict[str, List[Dict[str, float]]]:
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


def _create_paths_figure(
    scenario_rows: List[Dict[str, Any]],
    *,
    subtour_modes: List[str],
    output_path: Path,
) -> None:
    successful = {
        str(row["subtour_mode"]): row
        for row in scenario_rows
        if str(row.get("status", "")) == "ok" and str(row.get("artifact_dir", "")).strip()
    }
    if not successful:
        return

    playback_by_mode: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    all_x: List[float] = []
    all_y: List[float] = []
    for subtour_mode, row in successful.items():
        playback_frames = _read_playback_frames(Path(row["artifact_dir"]) / "playback_frames.csv")
        playback_by_mode[subtour_mode] = playback_frames
        for agent_frames in playback_frames.values():
            all_x.extend(frame["x"] for frame in agent_frames)
            all_y.extend(frame["y"] for frame in agent_frames)

    if not all_x or not all_y:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_pad = max((max(all_x) - min(all_x)) * 0.05, 1e-6)
    y_pad = max((max(all_y) - min(all_y)) * 0.05, 1e-6)
    color_cycle = ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED", "#0F766E", "#DB2777"]

    fig, axes = plt.subplots(1, len(subtour_modes), figsize=(7 * len(subtour_modes), 6), squeeze=False)
    seed = int(scenario_rows[0].get("seed", 0) or 0)
    node_count = int(scenario_rows[0].get("node_count", 0) or 0)
    uav_count = int(scenario_rows[0].get("uav_count", 0) or 0)

    for index, subtour_mode in enumerate(subtour_modes):
        axis = axes[0][index]
        row = successful.get(subtour_mode)
        axis.set_title(subtour_mode)
        if row is None:
            axis.text(0.5, 0.5, "No successful run", ha="center", va="center", transform=axis.transAxes)
            axis.axis("off")
            continue

        for color_index, (agent_id, frames) in enumerate(
            sorted(playback_by_mode[subtour_mode].items(), key=lambda item: int(item[0]))
        ):
            xs = [frame["x"] for frame in frames]
            ys = [frame["y"] for frame in frames]
            if not xs or not ys:
                continue
            color = color_cycle[color_index % len(color_cycle)]
            axis.plot(xs, ys, color=color, linewidth=1.8, label=f"UAV {agent_id}")
            axis.scatter(xs[0], ys[0], color=color, marker="o", s=18)
            axis.scatter(xs[-1], ys[-1], color=color, marker="x", s=24)

        axis.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
        axis.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.grid(True, linestyle=":", alpha=0.35)
        axis.legend(loc="best", fontsize=8)

    fig.suptitle(f"Subtour Path Comparison | seed={seed} n={node_count} k={uav_count}", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    figure_dir = Path(args.figure_dir)
    if output_path.exists() and not args.append:
        output_path.unlink()

    uav_values = [int(value) for value in EXPERIMENT_SUBTOUR_UAVS]
    subtour_modes = [str(mode) for mode in EXPERIMENT_SUBTOUR_MODES]
    total_runs = len(EXPERIMENT_DEFAULT_SEEDS) * len(EXPERIMENT_SUBTOUR_AREAS) * len(uav_values) * len(subtour_modes)

    scenario_rows_for_figures: List[Dict[str, Any]] = []
    with tqdm(total=total_runs, desc="Experiment Subtour", unit="run", dynamic_ncols=True) as progress:
        for seed in EXPERIMENT_DEFAULT_SEEDS:
            for node_count in EXPERIMENT_SUBTOUR_AREAS:
                for uav_count in uav_values:
                    scenario_payload = create_scenario(
                        node_count=int(node_count),
                        uav_count=int(uav_count),
                        coverage_time_profile="medium",
                        seed=int(seed),
                        scenario_name=EXPERIMENT_DEFAULT_SCENARIO,
                        objective_function=EXPERIMENT_DEFAULT_OBJECTIVE,
                        env_type=EXPERIMENT_DEFAULT_ENV,
                    )
                    for subtour_mode in subtour_modes:
                        progress.set_postfix_str(
                            f"seed={seed} n={node_count} k={uav_count} | {subtour_mode}",
                            refresh=False,
                        )
                        result = run_method(
                            scenario_payload,
                            "MILP",
                            model_name="milp",
                            solver_backend=EXPERIMENT_SUBTOUR_SOLVER,
                            subtour_mode=subtour_mode,
                            time_limit_seconds=EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
                            memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                            stage_solution=1,
                        )
                        metrics = dict(EMPTY_FIELDS)
                        metrics.update(result.get("metrics", {}))
                        metrics["solver_backend"] = EXPERIMENT_SUBTOUR_SOLVER
                        metrics["subtour_mode"] = subtour_mode
                        metrics["artifact_dir"] = result.get("artifact_dir", "")
                        metrics["error_message"] = result.get("error_message", "")
                        metrics.update(_derive_solver_status_fields(result, metrics))
                        metrics.update(_derive_dfj_validation_fields(metrics, subtour_mode))
                        row = format_result_row(
                            scenario_payload,
                            "MILP",
                            status=result["status"],
                            extra_fields=metrics,
                        )
                        save_results(output_path, [row])
                        scenario_rows_for_figures.append(dict(row))
                        progress.update(1)

    grouped_rows: Dict[tuple[int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in scenario_rows_for_figures:
        grouped_rows[_scenario_key(row)].append(row)

    for scenario_key, scenario_rows in grouped_rows.items():
        seed, node_count, uav_count = scenario_key
        output_name = f"subtour_paths_seed{seed}_n{node_count}_k{uav_count}.png"
        _create_paths_figure(
            scenario_rows,
            subtour_modes=subtour_modes,
            output_path=figure_dir / output_name,
        )


if __name__ == "__main__":
    main()
