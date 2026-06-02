from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import product
from pathlib import Path

from dummy_app.experiments.runner_utils import create_scenario, format_result_row, run_method, save_results
from dummy_app.program_config import (
    EXPERIMENT_D_FORMULATIONS,
    EXPERIMENT_D_GRID,
    EXPERIMENT_DEFAULT_ENV,
    EXPERIMENT_DEFAULT_MEMORY_LIMIT,
    EXPERIMENT_DEFAULT_SCENARIO,
    EXPERIMENT_DEFAULT_SEEDS,
    EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
    EXPERIMENT_RESULTS_DIR,
)
from tqdm import tqdm


def _sum_optional(values: list[float | None]) -> float | None:
    if not values:
        return None
    if any(value is None for value in values):
        return None
    return float(sum(float(value) for value in values))


def _extract_stage_metrics(result: dict) -> dict:
    run_result = result.get("run_result")
    if run_result is None:
        return {
            "primary_objective_value": None,
            "secondary_objective_value": None,
            "final_reported_objective": None,
            "stage1_objective": None,
            "stage2_objective": None,
            "stage1_runtime_sec": None,
            "stage2_runtime_sec": None,
        }

    summary = dict(getattr(run_result, "summary", {}))
    diagnostics = dict(getattr(run_result, "diagnostics", {}))
    history = list(diagnostics.get("solve_status_history", []))
    history_by_cluster: dict[int, list[dict]] = defaultdict(list)
    for record in history:
        history_by_cluster[int(record.get("cluster_id", -1))].append(dict(record))

    stage1_records = [records[0] for records in history_by_cluster.values() if records]
    stage2_records = [records[1] for records in history_by_cluster.values() if len(records) > 1]

    return {
        "primary_objective_value": summary.get("summary_objective_value", summary.get("objective_value")),
        "secondary_objective_value": summary.get("makespan"),
        "final_reported_objective": getattr(run_result, "objective_value", None),
        "stage1_objective": _sum_optional([record.get("objective_value") for record in stage1_records]),
        "stage2_objective": _sum_optional([record.get("objective_value") for record in stage2_records]),
        "stage1_runtime_sec": _sum_optional([record.get("elapsed_time_seconds") for record in stage1_records]),
        "stage2_runtime_sec": _sum_optional([record.get("elapsed_time_seconds") for record in stage2_records]),
    }


def _percentage_change(current_value: float | None, baseline_value: float | None) -> float | None:
    if current_value is None or baseline_value is None:
        return None
    if abs(float(baseline_value)) <= 1e-9:
        return 0.0 if abs(float(current_value)) <= 1e-9 else None
    return float((float(current_value) - float(baseline_value)) / abs(float(baseline_value)) * 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment D: MILP objective staging comparison.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(EXPERIMENT_RESULTS_DIR) / "experiment_D_objective_staging.csv"),
    )
    parser.add_argument("--append", action="store_true", help="Append to the output CSV instead of replacing it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.append:
        output_path.unlink()

    total_runs = (
        len(EXPERIMENT_DEFAULT_SEEDS)
        * len(EXPERIMENT_D_GRID["areas"])
        * len(EXPERIMENT_D_GRID["uavs"])
        * len(EXPERIMENT_D_FORMULATIONS)
    )

    with tqdm(total=total_runs, desc="Experiment D", unit="run", dynamic_ncols=True) as progress:
        for seed in EXPERIMENT_DEFAULT_SEEDS:
            for node_count, uav_count in product(
                EXPERIMENT_D_GRID["areas"],
                EXPERIMENT_D_GRID["uavs"],
            ):
                scenario_payload = create_scenario(
                    node_count=node_count,
                    uav_count=uav_count,
                    coverage_time_profile="medium",
                    seed=seed,
                    scenario_name=EXPERIMENT_DEFAULT_SCENARIO,
                    objective_function="coverage",
                    env_type=EXPERIMENT_DEFAULT_ENV,
                )
                scenario_results = {}
                for formulation_type, stage_solution in EXPERIMENT_D_FORMULATIONS.items():
                    progress.set_postfix_str(
                        f"seed={seed} n={node_count} k={uav_count} | {formulation_type}",
                        refresh=False,
                    )
                    result = run_method(
                        scenario_payload,
                        "MILP",
                        model_name="milp",
                        stage_solution=stage_solution,
                        time_limit_seconds=EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
                        memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                    )
                    scenario_results[formulation_type] = result

                baseline_metrics = scenario_results.get("single_stage_milp", {}).get("metrics", {})
                for formulation_type in EXPERIMENT_D_FORMULATIONS:
                    result = scenario_results[formulation_type]
                    metrics = result.get("metrics", {})
                    stage_metrics = _extract_stage_metrics(result)
                    row = format_result_row(
                        scenario_payload,
                        "MILP",
                        status=result["status"],
                        extra_fields={
                            "formulation_type": formulation_type,
                            "coverage_ratio": metrics.get("coverage_ratio"),
                            "total_distance": metrics.get("total_distance"),
                            "total_energy": metrics.get("total_energy"),
                            "total_travel_time": metrics.get("total_travel_time"),
                            "workload_balance_distance_std": metrics.get("distance_std_across_uavs"),
                            "workload_balance_energy_std": metrics.get("energy_std_across_uavs"),
                            "objective_value": metrics.get("objective_value"),
                            "primary_objective_value": stage_metrics.get("primary_objective_value"),
                            "secondary_objective_value": stage_metrics.get("secondary_objective_value"),
                            "final_reported_objective": stage_metrics.get("final_reported_objective"),
                            "stage1_objective": stage_metrics.get("stage1_objective"),
                            "stage2_objective": stage_metrics.get("stage2_objective"),
                            "stage1_runtime_sec": stage_metrics.get("stage1_runtime_sec"),
                            "stage2_runtime_sec": stage_metrics.get("stage2_runtime_sec"),
                            "distance_change_percent": 0.0
                            if formulation_type == "single_stage_milp"
                            else _percentage_change(metrics.get("total_distance"), baseline_metrics.get("total_distance")),
                            "energy_change_percent": 0.0
                            if formulation_type == "single_stage_milp"
                            else _percentage_change(metrics.get("total_energy"), baseline_metrics.get("total_energy")),
                            "balance_distance_std_change_percent": 0.0
                            if formulation_type == "single_stage_milp"
                            else _percentage_change(
                                metrics.get("distance_std_across_uavs"),
                                baseline_metrics.get("distance_std_across_uavs"),
                            ),
                            "runtime_sec": metrics.get("runtime_sec"),
                            "optimality_gap_percent": metrics.get("optimality_gap_percent"),
                            "artifact_dir": result.get("artifact_dir", ""),
                            "error_message": result.get("error_message", ""),
                        },
                    )
                    save_results(output_path, [row])
                    progress.update(1)


if __name__ == "__main__":
    main()
