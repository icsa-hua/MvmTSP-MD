from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from dummy_app.experiments.runner_utils import create_scenario, format_result_row, run_method, save_results
from dummy_app.program_config import (
    EXPERIMENT_DEFAULT_ENV,
    EXPERIMENT_DEFAULT_MEMORY_LIMIT,
    EXPERIMENT_DEFAULT_SCENARIO,
    EXPERIMENT_DEFAULT_SEEDS,
    EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
    EXPERIMENT_RESULTS_DIR,
)
from tqdm import tqdm


OBJECTIVES = ["energy", "coverage"]

EXPERIMENT_I_GRID = {
    "areas": [25, 50, 100, 200],
    "uavs": [3, 4, 6, 8],
}

EMPTY_FIELDS = {
    "objective_function": "",
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
    "solver_objective_value": None,
    "summary_objective_value": None,
    "best_bound": None,
    "optimality_gap_percent": None,
    "feasible_solution_found": False,
    "time_to_first_feasible_sec": None,
    "branch_and_bound_nodes": None,
    "memory_usage_mb": None,
    "distance_per_uav": "",
    "energy_per_uav": "",
    "total_data_rate_mbps": None,
    "data_rate_per_hour_mbps": None,
    "data_rate_per_kwh_mbps": None,
    "avg_data_rate_per_cluster_mbps": None,
    "mean_sinr_db": None,
    "coverage_prob_at_0db": None,
    "coverage_prob_at_10db": None,
    "coverage_prob_at_20db": None,
    "artifact_dir": "",
    "error_message": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment I: energy vs coverage objective comparison.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(EXPERIMENT_RESULTS_DIR) / "experiment_I_objective_comparison.csv"),
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
        * len(EXPERIMENT_I_GRID["areas"])
        * len(EXPERIMENT_I_GRID["uavs"])
        * len(OBJECTIVES)
    )

    with tqdm(total=total_runs, desc="Experiment I", unit="run", dynamic_ncols=True) as progress:
        for seed in EXPERIMENT_DEFAULT_SEEDS:
            for node_count, uav_count in product(
                EXPERIMENT_I_GRID["areas"],
                EXPERIMENT_I_GRID["uavs"],
            ):
                for objective in OBJECTIVES:
                    progress.set_postfix_str(
                        f"seed={seed} n={node_count} k={uav_count} | {objective}",
                        refresh=False,
                    )
                    scenario_payload = create_scenario(
                        node_count=node_count,
                        uav_count=uav_count,
                        coverage_time_profile="medium",
                        seed=seed,
                        scenario_name=EXPERIMENT_DEFAULT_SCENARIO,
                        objective_function=objective,
                        env_type=EXPERIMENT_DEFAULT_ENV,
                    )
                    result = run_method(
                        scenario_payload,
                        "MILP",
                        model_name="milp",
                        time_limit_seconds=EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
                        memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                        stage_solution=1,
                    )
                    metrics = dict(EMPTY_FIELDS)
                    metrics.update(result.get("metrics", {}))
                    metrics["objective_function"] = objective
                    metrics["error_message"] = result.get("error_message", "")
                    metrics["artifact_dir"] = result.get("artifact_dir", "")
                    row = format_result_row(
                        scenario_payload,
                        "MILP",
                        status=result["status"],
                        extra_fields=metrics,
                    )
                    save_results(output_path, [row])
                    progress.update(1)


if __name__ == "__main__":
    main()
