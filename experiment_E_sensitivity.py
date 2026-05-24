from __future__ import annotations

import argparse
from pathlib import Path

from experiments.runner_utils import create_scenario, format_result_row, run_method, save_results
from program_config import (
    EXPERIMENT_DEFAULT_ENV,
    EXPERIMENT_DEFAULT_MEMORY_LIMIT,
    EXPERIMENT_DEFAULT_OBJECTIVE,
    EXPERIMENT_DEFAULT_SCENARIO,
    EXPERIMENT_DEFAULT_SEEDS,
    EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
    EXPERIMENT_E_BASELINE,
    EXPERIMENT_E_FAIRNESS_THRESHOLDS,
    EXPERIMENT_E_OBJECTIVE_WEIGHT_PROFILES,
    EXPERIMENT_E_TIME_STEP_SEC,
    EXPERIMENT_OBJECTIVE_WEIGHT_PROFILES,
    EXPERIMENT_RESULTS_DIR,
)
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment E: MILP sensitivity analysis.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(EXPERIMENT_RESULTS_DIR) / "experiment_E_sensitivity.csv"),
    )
    parser.add_argument("--append", action="store_true", help="Append to the output CSV instead of replacing it.")
    return parser.parse_args()


def _build_row(scenario_payload, result, tested_parameter: str, parameter_value) -> dict:
    metrics = result.get("metrics", {})
    return format_result_row(
        scenario_payload,
        "MILP",
        status=result["status"],
        extra_fields={
            "tested_parameter": tested_parameter,
            "parameter_value": parameter_value,
            "runtime_sec": metrics.get("runtime_sec"),
            "optimality_gap_percent": metrics.get("optimality_gap_percent"),
            "coverage_ratio": metrics.get("coverage_ratio"),
            "total_distance": metrics.get("total_distance"),
            "total_energy": metrics.get("total_energy"),
            "total_travel_time": metrics.get("total_travel_time"),
            "max_route_distance": metrics.get("max_route_distance"),
            "max_route_energy": metrics.get("max_route_energy"),
            "distance_std_across_uavs": metrics.get("distance_std_across_uavs"),
            "energy_std_across_uavs": metrics.get("energy_std_across_uavs"),
            "objective_value": metrics.get("objective_value"),
            "artifact_dir": result.get("artifact_dir", ""),
            "error_message": result.get("error_message", ""),
        },
    )


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.append:
        output_path.unlink()

    total_runs = len(EXPERIMENT_DEFAULT_SEEDS) * (
        len(EXPERIMENT_E_FAIRNESS_THRESHOLDS)
        + len(EXPERIMENT_E_TIME_STEP_SEC)
        + len(EXPERIMENT_E_OBJECTIVE_WEIGHT_PROFILES)
    )

    with tqdm(total=total_runs, desc="Experiment E", unit="run", dynamic_ncols=True) as progress:
        for seed in EXPERIMENT_DEFAULT_SEEDS:
            scenario_payload = create_scenario(
                node_count=EXPERIMENT_E_BASELINE["areas"],
                users_per_area=EXPERIMENT_E_BASELINE["users_per_area"],
                uav_count=EXPERIMENT_E_BASELINE["uavs"],
                battery_level=EXPERIMENT_E_BASELINE["battery_level"],
                coverage_time_profile=EXPERIMENT_E_BASELINE["coverage_time_profile"],
                seed=seed,
                scenario_name=EXPERIMENT_DEFAULT_SCENARIO,
                objective_function=EXPERIMENT_DEFAULT_OBJECTIVE,
                env_type=EXPERIMENT_DEFAULT_ENV,
            )

            for fairness_threshold in EXPERIMENT_E_FAIRNESS_THRESHOLDS:
                progress.set_postfix_str(f"seed={seed} | fairness_threshold={fairness_threshold}", refresh=False)
                result = run_method(
                    scenario_payload,
                    "MILP",
                    model_name="milp",
                    fairness_tolerance=fairness_threshold,
                    time_limit_seconds=EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
                    memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                )
                save_results(output_path, [_build_row(scenario_payload, result, "fairness_threshold", fairness_threshold)])
                progress.update(1)

            for time_step_sec in EXPERIMENT_E_TIME_STEP_SEC:
                progress.set_postfix_str(f"seed={seed} | time_step_sec={time_step_sec}", refresh=False)
                result = run_method(
                    scenario_payload,
                    "MILP",
                    model_name="milp",
                    time_step_sec=time_step_sec,
                    time_limit_seconds=EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
                    memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                )
                save_results(output_path, [_build_row(scenario_payload, result, "time_step_sec", time_step_sec)])
                progress.update(1)

            for profile_name in EXPERIMENT_E_OBJECTIVE_WEIGHT_PROFILES:
                progress.set_postfix_str(f"seed={seed} | objective_weight_profile={profile_name}", refresh=False)
                result = run_method(
                    scenario_payload,
                    "MILP",
                    model_name="milp",
                    objective_weights=EXPERIMENT_OBJECTIVE_WEIGHT_PROFILES[profile_name],
                    time_limit_seconds=EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
                    memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                )
                save_results(output_path, [_build_row(scenario_payload, result, "objective_weight_profile", profile_name)])
                progress.update(1)


if __name__ == "__main__":
    main()
