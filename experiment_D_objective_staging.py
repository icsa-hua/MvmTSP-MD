from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from experiments.runner_utils import create_scenario, format_result_row, run_method, save_results
from program_config import (
    EXPERIMENT_D_FORMULATIONS,
    EXPERIMENT_D_GRID,
    EXPERIMENT_DEFAULT_ENV,
    EXPERIMENT_DEFAULT_MEMORY_LIMIT,
    EXPERIMENT_DEFAULT_SCENARIO,
    EXPERIMENT_DEFAULT_SEEDS,
    EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
    EXPERIMENT_RESULTS_DIR,
)


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

    for seed in EXPERIMENT_DEFAULT_SEEDS:
        for node_count, uav_count, battery_level in product(
            EXPERIMENT_D_GRID["areas"],
            EXPERIMENT_D_GRID["uavs"],
            EXPERIMENT_D_GRID["battery_level"],
        ):
            scenario_payload = create_scenario(
                node_count=node_count,
                uav_count=uav_count,
                battery_level=battery_level,
                coverage_time_profile="medium",
                seed=seed,
                scenario_name=EXPERIMENT_DEFAULT_SCENARIO,
                objective_function="coverage",
                env_type=EXPERIMENT_DEFAULT_ENV,
            )
            for formulation_type, stage_solution in EXPERIMENT_D_FORMULATIONS.items():
                result = run_method(
                    scenario_payload,
                    "MILP",
                    model_name="milp",
                    stage_solution=stage_solution,
                    time_limit_seconds=EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
                    memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                )
                metrics = result.get("metrics", {})
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
                        "runtime_sec": metrics.get("runtime_sec"),
                        "optimality_gap_percent": metrics.get("optimality_gap_percent"),
                        "artifact_dir": result.get("artifact_dir", ""),
                        "error_message": result.get("error_message", ""),
                    },
                )
                save_results(output_path, [row])


if __name__ == "__main__":
    main()
