from __future__ import annotations

import argparse
from pathlib import Path

from dummy_app.experiments.metrics import aggregate_agent_routes
from dummy_app.experiments.runner_utils import create_scenario, format_result_row, run_method, save_results
from dummy_app.program_config import (
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


EXPERIMENT_E_SCENARIOS = [
    {"areas": 50, "users_per_area": 3, "uavs": 4},
    {"areas": 100, "users_per_area": 3, "uavs": 4},
    {"areas": 150, "users_per_area": 3, "uavs": 6},
]
EXPERIMENT_E_BASELINE_FAIRNESS = 2
EXPERIMENT_E_BASELINE_TIME_STEP_SEC = 600
EXPERIMENT_E_BASELINE_OBJECTIVE_WEIGHT_PROFILE = "balanced"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment E: MILP sensitivity analysis.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(EXPERIMENT_RESULTS_DIR) / "experiment_E_sensitivity.csv"),
    )
    parser.add_argument("--append", action="store_true", help="Append to the output CSV instead of replacing it.")
    return parser.parse_args()


def _percentage_change(current_value: float | None, baseline_value: float | None) -> float | None:
    if current_value is None or baseline_value is None:
        return None
    if abs(float(baseline_value)) <= 1e-9:
        return 0.0 if abs(float(current_value)) <= 1e-9 else None
    return float((float(current_value) - float(baseline_value)) / abs(float(baseline_value)) * 100.0)


def _scenario_variant_label(scenario_spec: dict) -> str:
    return f"n{int(scenario_spec['areas'])}_u{int(scenario_spec['users_per_area'])}_k{int(scenario_spec['uavs'])}"


def _compress_agent_path(agent_path) -> list[tuple[int, int]]:
    if not agent_path:
        return []

    compressed: list[tuple[int, int]] = []
    current_source, current_target, _ = agent_path[0]
    compressed.append((int(current_source), int(current_target)))
    for source, target, _ in agent_path[1:]:
        arc = (int(source), int(target))
        if arc == compressed[-1]:
            continue
        compressed.append(arc)
    return compressed


def _route_signature(result: dict) -> tuple | None:
    run_result = result.get("run_result")
    if run_result is None:
        return None
    routes = aggregate_agent_routes(run_result.cluster_results)
    return tuple(
        (int(agent_id), tuple(_compress_agent_path(agent_path)))
        for agent_id, agent_path in sorted(routes.items(), key=lambda item: int(item[0]))
    )


def _build_row(
    scenario_payload,
    result,
    tested_parameter: str,
    parameter_value,
    *,
    scenario_variant: str,
    is_baseline: bool,
    baseline_metrics: dict,
    same_route_as_baseline: bool | None = None,
) -> dict:
    metrics = result.get("metrics", {})
    return format_result_row(
        scenario_payload,
        "MILP",
        status=result["status"],
        extra_fields={
            "scenario_variant": scenario_variant,
            "tested_parameter": tested_parameter,
            "parameter_value": parameter_value,
            "is_baseline": bool(is_baseline),
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
            "distance_change_percent": 0.0
            if is_baseline
            else _percentage_change(metrics.get("total_distance"), baseline_metrics.get("total_distance")),
            "energy_change_percent": 0.0
            if is_baseline
            else _percentage_change(metrics.get("total_energy"), baseline_metrics.get("total_energy")),
            "travel_time_change_percent": 0.0
            if is_baseline
            else _percentage_change(metrics.get("total_travel_time"), baseline_metrics.get("total_travel_time")),
            "balance_distance_std_change_percent": 0.0
            if is_baseline
            else _percentage_change(
                metrics.get("distance_std_across_uavs"),
                baseline_metrics.get("distance_std_across_uavs"),
            ),
            "balance_energy_std_change_percent": 0.0
            if is_baseline
            else _percentage_change(
                metrics.get("energy_std_across_uavs"),
                baseline_metrics.get("energy_std_across_uavs"),
            ),
            "same_route_as_baseline": same_route_as_baseline,
            "artifact_dir": result.get("artifact_dir", ""),
            "error_message": result.get("error_message", ""),
        },
    )


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.append:
        output_path.unlink()

    fairness_values = list(EXPERIMENT_E_FAIRNESS_THRESHOLDS)
    time_step_values = [EXPERIMENT_E_BASELINE_TIME_STEP_SEC] + [
        int(value) for value in EXPERIMENT_E_TIME_STEP_SEC if int(value) != int(EXPERIMENT_E_BASELINE_TIME_STEP_SEC)
    ]
    objective_weight_profiles = [EXPERIMENT_E_BASELINE_OBJECTIVE_WEIGHT_PROFILE] + [
        str(profile)
        for profile in EXPERIMENT_E_OBJECTIVE_WEIGHT_PROFILES
        if str(profile) != EXPERIMENT_E_BASELINE_OBJECTIVE_WEIGHT_PROFILE
    ]

    total_runs = len(EXPERIMENT_DEFAULT_SEEDS) * len(EXPERIMENT_E_SCENARIOS) * (
        len(fairness_values)
        + len(time_step_values)
        + len(objective_weight_profiles)
    )

    with tqdm(total=total_runs, desc="Experiment E", unit="run", dynamic_ncols=True) as progress:
        for seed in EXPERIMENT_DEFAULT_SEEDS:
            for scenario_spec in EXPERIMENT_E_SCENARIOS:
                scenario_variant = _scenario_variant_label(scenario_spec)
                scenario_payload = create_scenario(
                    node_count=scenario_spec["areas"],
                    users_per_area=scenario_spec["users_per_area"],
                    uav_count=scenario_spec["uavs"],
                    coverage_time_profile=EXPERIMENT_E_BASELINE["coverage_time_profile"],
                    seed=seed,
                    scenario_name=EXPERIMENT_DEFAULT_SCENARIO,
                    objective_function=EXPERIMENT_DEFAULT_OBJECTIVE,
                    env_type=EXPERIMENT_DEFAULT_ENV,
                )

                fairness_results = {}
                for fairness_threshold in fairness_values:
                    progress.set_postfix_str(
                        f"seed={seed} {scenario_variant} | fairness_threshold={fairness_threshold}",
                        refresh=False,
                    )
                    result = run_method(
                        scenario_payload,
                        "MILP",
                        model_name="milp",
                        fairness_tolerance=fairness_threshold,
                        time_limit_seconds=EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
                        memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                    )
                    fairness_results[int(fairness_threshold)] = result
                    progress.update(1)

                fairness_baseline_metrics = fairness_results.get(EXPERIMENT_E_BASELINE_FAIRNESS, {}).get("metrics", {})
                for fairness_threshold in fairness_values:
                    save_results(
                        output_path,
                        [
                            _build_row(
                                scenario_payload,
                                fairness_results[int(fairness_threshold)],
                                "fairness_threshold",
                                fairness_threshold,
                                scenario_variant=scenario_variant,
                                is_baseline=int(fairness_threshold) == int(EXPERIMENT_E_BASELINE_FAIRNESS),
                                baseline_metrics=fairness_baseline_metrics,
                            )
                        ],
                    )

                time_step_results = {}
                for time_step_sec in time_step_values:
                    progress.set_postfix_str(
                        f"seed={seed} {scenario_variant} | time_step_sec={time_step_sec}",
                        refresh=False,
                    )
                    result = run_method(
                        scenario_payload,
                        "MILP",
                        model_name="milp",
                        time_step_sec=time_step_sec,
                        time_limit_seconds=EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
                        memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                    )
                    time_step_results[int(time_step_sec)] = result
                    progress.update(1)

                time_step_baseline_metrics = time_step_results.get(EXPERIMENT_E_BASELINE_TIME_STEP_SEC, {}).get("metrics", {})
                for time_step_sec in time_step_values:
                    save_results(
                        output_path,
                        [
                            _build_row(
                                scenario_payload,
                                time_step_results[int(time_step_sec)],
                                "time_step_sec",
                                time_step_sec,
                                scenario_variant=scenario_variant,
                                is_baseline=int(time_step_sec) == int(EXPERIMENT_E_BASELINE_TIME_STEP_SEC),
                                baseline_metrics=time_step_baseline_metrics,
                            )
                        ],
                    )

                objective_weight_results = {}
                for profile_name in objective_weight_profiles:
                    progress.set_postfix_str(
                        f"seed={seed} {scenario_variant} | objective_weight_profile={profile_name}",
                        refresh=False,
                    )
                    result = run_method(
                        scenario_payload,
                        "MILP",
                        model_name="milp",
                        objective_weights=EXPERIMENT_OBJECTIVE_WEIGHT_PROFILES[profile_name],
                        time_limit_seconds=EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
                        memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                    )
                    objective_weight_results[str(profile_name)] = result
                    progress.update(1)

                objective_weight_baseline_result = objective_weight_results.get(EXPERIMENT_E_BASELINE_OBJECTIVE_WEIGHT_PROFILE, {})
                objective_weight_baseline_metrics = objective_weight_baseline_result.get("metrics", {})
                baseline_route_signature = _route_signature(objective_weight_baseline_result)
                for profile_name in objective_weight_profiles:
                    current_result = objective_weight_results[str(profile_name)]
                    current_route_signature = _route_signature(current_result)
                    save_results(
                        output_path,
                        [
                            _build_row(
                                scenario_payload,
                                current_result,
                                "objective_weight_profile",
                                profile_name,
                                scenario_variant=scenario_variant,
                                is_baseline=str(profile_name) == EXPERIMENT_E_BASELINE_OBJECTIVE_WEIGHT_PROFILE,
                                baseline_metrics=objective_weight_baseline_metrics,
                                same_route_as_baseline=(
                                    None
                                    if baseline_route_signature is None or current_route_signature is None
                                    else current_route_signature == baseline_route_signature
                                ),
                            )
                        ],
                    )


if __name__ == "__main__":
    main()
