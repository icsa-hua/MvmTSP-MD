from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from experiments.runner_utils import create_scenario, format_result_row, run_method, save_results
from program_config import (
    EXPERIMENT_C_GRID,
    EXPERIMENT_C_WARM_STARTS,
    EXPERIMENT_DEFAULT_ENV,
    EXPERIMENT_DEFAULT_MEMORY_LIMIT,
    EXPERIMENT_DEFAULT_OBJECTIVE,
    EXPERIMENT_DEFAULT_SCENARIO,
    EXPERIMENT_DEFAULT_SEEDS,
    EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
    EXPERIMENT_RESULTS_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment C: MILP warm-start comparison.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(EXPERIMENT_RESULTS_DIR) / "experiment_C_warm_start.csv"),
    )
    parser.add_argument("--append", action="store_true", help="Append to the output CSV instead of replacing it.")
    return parser.parse_args()


def _warm_start_objective(result: dict) -> float | None:
    run_result = result.get("run_result")
    if run_result is None:
        return None
    objective_values = []
    for cluster_result in run_result.cluster_results:
        warm_start_summary = dict(cluster_result.diagnostics).get("warm_start_summary", {})
        objective_value = warm_start_summary.get("objective_value")
        if objective_value is not None:
            objective_values.append(float(objective_value))
    if not objective_values:
        return None
    return float(sum(objective_values))


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.append:
        output_path.unlink()

    for seed in EXPERIMENT_DEFAULT_SEEDS:
        for node_count, uav_count, battery_level in product(
            EXPERIMENT_C_GRID["areas"],
            EXPERIMENT_C_GRID["uavs"],
            EXPERIMENT_C_GRID["battery_level"],
        ):
            scenario_payload = create_scenario(
                node_count=node_count,
                uav_count=uav_count,
                battery_level=battery_level,
                coverage_time_profile="medium",
                seed=seed,
                scenario_name=EXPERIMENT_DEFAULT_SCENARIO,
                objective_function=EXPERIMENT_DEFAULT_OBJECTIVE,
                env_type=EXPERIMENT_DEFAULT_ENV,
            )
            scenario_results = {}
            for warm_start_method in EXPERIMENT_C_WARM_STARTS:
                label = f"MILP + {warm_start_method.upper()}" if warm_start_method != "none" else "MILP"
                result = run_method(
                    scenario_payload,
                    "MILP",
                    model_name="milp",
                    warm_start_mode=warm_start_method,
                    time_limit_seconds=EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
                    memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                    stage_solution=1,
                )
                scenario_results[warm_start_method] = result
                baseline_metrics = scenario_results.get("none", {}).get("metrics", {})
                current_metrics = result.get("metrics", {})
                baseline_gap = baseline_metrics.get("optimality_gap_percent")
                baseline_objective = baseline_metrics.get("objective_value")
                current_gap = current_metrics.get("optimality_gap_percent")
                current_objective = current_metrics.get("objective_value")
                row = format_result_row(
                    scenario_payload,
                    "MILP",
                    status=result["status"],
                    extra_fields={
                        "warm_start_method": warm_start_method,
                        "warm_start_objective": _warm_start_objective(result),
                        "time_to_first_feasible_sec": current_metrics.get("time_to_first_feasible_sec"),
                        "final_objective_value": current_objective,
                        "best_bound": current_metrics.get("best_bound"),
                        "final_optimality_gap_percent": current_gap,
                        "runtime_sec": current_metrics.get("runtime_sec"),
                        "branch_and_bound_nodes": current_metrics.get("branch_and_bound_nodes"),
                        "gap_reduction_vs_no_warm_start": 0.0
                        if warm_start_method == "none" or baseline_gap is None or current_gap is None
                        else float(baseline_gap) - float(current_gap),
                        "objective_improvement_vs_no_warm_start": 0.0
                        if warm_start_method == "none" or baseline_objective is None or current_objective is None
                        else float(baseline_objective) - float(current_objective),
                        "artifact_dir": result.get("artifact_dir", ""),
                        "error_message": result.get("error_message", ""),
                    },
                )
                save_results(output_path, [row])


if __name__ == "__main__":
    main()
