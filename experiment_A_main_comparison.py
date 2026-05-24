from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from experiments.runner_utils import create_scenario, format_result_row, run_method, save_results
from program_config import (
    EXPERIMENT_A_GRID,
    EXPERIMENT_DEFAULT_ENV,
    EXPERIMENT_DEFAULT_MEMORY_LIMIT,
    EXPERIMENT_DEFAULT_OBJECTIVE,
    EXPERIMENT_DEFAULT_SCENARIO,
    EXPERIMENT_DEFAULT_SEEDS,
    EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
    EXPERIMENT_RESULTS_DIR,
)


METHODS = [
    ("MILP", "milp"),
    ("Greedy NN", "greedy_nn"),
    ("Static Partitioning + Greedy NN", "greedy_partition_nn"),
    ("Genetic Algorithm", "ga"),
    ("ALNS", "alns"),
]

EMPTY_FIELDS = {
    "runtime_sec": None,
    "coverage_ratio": None,
    "total_distance": None,
    "total_energy": None,
    "total_travel_time": None,
    "max_route_distance": None,
    "max_route_energy": None,
    "distance_std_across_uavs": None,
    "energy_std_across_uavs": None,
    "jain_fairness_distance": None,
    "jain_fairness_energy": None,
    "unvisited_nodes": None,
    "objective_value": None,
    "distance_per_uav": "",
    "energy_per_uav": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment A: main method comparison.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(EXPERIMENT_RESULTS_DIR) / "experiment_A_main_comparison.csv"),
    )
    parser.add_argument("--append", action="store_true", help="Append to the output CSV instead of replacing it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.append:
        output_path.unlink()

    for seed in EXPERIMENT_DEFAULT_SEEDS:
        for node_count, users_per_area, uav_count, battery_level, coverage_time_profile in product(
            EXPERIMENT_A_GRID["areas"],
            EXPERIMENT_A_GRID["users_per_area"],
            EXPERIMENT_A_GRID["uavs"],
            EXPERIMENT_A_GRID["battery_level"],
            EXPERIMENT_A_GRID["coverage_time_profile"],
        ):
            scenario_payload = create_scenario(
                node_count=node_count,
                users_per_area=users_per_area,
                uav_count=uav_count,
                battery_level=battery_level,
                coverage_time_profile=coverage_time_profile,
                seed=seed,
                scenario_name=EXPERIMENT_DEFAULT_SCENARIO,
                objective_function=EXPERIMENT_DEFAULT_OBJECTIVE,
                env_type=EXPERIMENT_DEFAULT_ENV,
            )
            for method_name, model_name in METHODS:
                result = run_method(
                    scenario_payload,
                    method_name,
                    model_name=model_name,
                    time_limit_seconds=EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
                    memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                    stage_solution=1,
                )
                metrics = dict(EMPTY_FIELDS)
                metrics.update(result.get("metrics", {}))
                metrics["error_message"] = result.get("error_message", "")
                metrics["artifact_dir"] = result.get("artifact_dir", "")
                row = format_result_row(
                    scenario_payload,
                    method_name,
                    status=result["status"],
                    extra_fields=metrics,
                )
                save_results(output_path, [row])


if __name__ == "__main__":
    main()
