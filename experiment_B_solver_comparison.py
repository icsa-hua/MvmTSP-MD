from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from experiments.runner_utils import create_scenario, format_result_row, run_method, save_results
from program_config import (
    EXPERIMENT_B_GRID,
    EXPERIMENT_B_SOLVERS,
    EXPERIMENT_DEFAULT_ENV,
    EXPERIMENT_DEFAULT_MEMORY_LIMIT,
    EXPERIMENT_DEFAULT_OBJECTIVE,
    EXPERIMENT_DEFAULT_SCENARIO,
    EXPERIMENT_DEFAULT_SEEDS,
    EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
    EXPERIMENT_RESULTS_DIR,
)
from tqdm import tqdm


EMPTY_FIELDS = {
    "solver_name": "",
    "runtime_sec": None,
    "objective_value": None,
    "best_bound": None,
    "optimality_gap_percent": None,
    "feasible_solution_found": False,
    "time_to_first_feasible_sec": None,
    "branch_and_bound_nodes": None,
    "memory_usage_mb": None,
    "artifact_dir": "",
    "error_message": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment B: MILP solver comparison.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(EXPERIMENT_RESULTS_DIR) / "experiment_B_solver_comparison.csv"),
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
        * len(EXPERIMENT_B_GRID["areas"])
        * len(EXPERIMENT_B_GRID["uavs"])
        * len(EXPERIMENT_B_SOLVERS)
    )

    with tqdm(total=total_runs, desc="Experiment B", unit="run", dynamic_ncols=True) as progress:
        for seed in EXPERIMENT_DEFAULT_SEEDS:
            for node_count, uav_count in product(
                EXPERIMENT_B_GRID["areas"],
                EXPERIMENT_B_GRID["uavs"],
            ):
                scenario_payload = create_scenario(
                    node_count=node_count,
                    uav_count=uav_count,
                    coverage_time_profile="medium",
                    seed=seed,
                    scenario_name=EXPERIMENT_DEFAULT_SCENARIO,
                    objective_function=EXPERIMENT_DEFAULT_OBJECTIVE,
                    env_type=EXPERIMENT_DEFAULT_ENV,
                )
                for solver_name in EXPERIMENT_B_SOLVERS:
                    progress.set_postfix_str(
                        f"seed={seed} n={node_count} k={uav_count} | {solver_name}",
                        refresh=False,
                    )
                    result = run_method(
                        scenario_payload,
                        "MILP",
                        model_name="milp",
                        solver_backend=solver_name,
                        time_limit_seconds=EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
                        memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                        stage_solution=1,
                    )
                    metrics = dict(EMPTY_FIELDS)
                    metrics.update(
                        {
                            "solver_name": solver_name,
                            "runtime_sec": result.get("metrics", {}).get("runtime_sec"),
                            "objective_value": result.get("metrics", {}).get("objective_value"),
                            "best_bound": result.get("metrics", {}).get("best_bound"),
                            "optimality_gap_percent": result.get("metrics", {}).get("optimality_gap_percent"),
                            "feasible_solution_found": result.get("metrics", {}).get("feasible_solution_found", False),
                            "time_to_first_feasible_sec": result.get("metrics", {}).get("time_to_first_feasible_sec"),
                            "branch_and_bound_nodes": result.get("metrics", {}).get("branch_and_bound_nodes"),
                            "memory_usage_mb": result.get("metrics", {}).get("memory_usage_mb"),
                            "artifact_dir": result.get("artifact_dir", ""),
                            "error_message": result.get("error_message", ""),
                        }
                    )
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
