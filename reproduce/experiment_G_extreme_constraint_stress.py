from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
from tqdm import tqdm

from dummy_app.experiments.runner_utils import (
    _build_runtime_config,
    _enforce_memory_limit,
    _extract_failure_metrics,
    _suppress_nested_output,
    create_scenario,
    enforce_time_limit,
    extract_common_run_metrics,
    format_result_row,
    save_results,
    set_random_seed,
)
from dummy_app.program_config import (
    ALTITUDE,
    EXPERIMENT_DEFAULT_MEMORY_LIMIT,
    EXPERIMENT_DEFAULT_OBJECTIVE,
    EXPERIMENT_DEFAULT_PRIORITY,
    EXPERIMENT_DEFAULT_SCENARIO,
    EXPERIMENT_DEFAULT_SEEDS,
    EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
    EXPERIMENT_RESULTS_DIR,
    HORIZONTAL_VELOCITY,
    MAX_BATTERY,
    SUBTOUR_MODE,
    VERTICAL_VELOCITY,
)
from dummy_app.tools.common import call_builder


EXPERIMENT_G_CASES = [
    {"areas": 100, "users_per_area": 3, "uavs": 4},
    {"areas": 200, "users_per_area": 3, "uavs": 6},
]
EXPERIMENT_G_USABLE_CAPACITY_FACTORS = [1.0, 0.75, 0.50]
EXPERIMENT_G_ENVIRONMENTS = ["urban", "rural", "forest"]
EXPERIMENT_G_COVERAGE_PROFILES = ["low", "medium", "high"]
EXPERIMENT_E_RESULTS_PATH = Path(EXPERIMENT_RESULTS_DIR) / "experiment_E_sensitivity.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment G: extreme constraint stress test.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(EXPERIMENT_RESULTS_DIR) / "experiment_G_extreme_constraint_stress.csv"),
    )
    parser.add_argument("--append", action="store_true", help="Append to the output CSV instead of replacing it.")
    return parser.parse_args()


def _load_fairness_baseline() -> int:
    if not EXPERIMENT_E_RESULTS_PATH.exists():
        return 2

    with EXPERIMENT_E_RESULTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    baseline_values = sorted(
        {
            int(str(row.get("parameter_value", "")).strip())
            for row in rows
            if str(row.get("tested_parameter", "")).strip() == "fairness_threshold"
            and str(row.get("is_baseline", "")).strip().lower() == "true"
            and str(row.get("parameter_value", "")).strip()
        }
    )
    if len(baseline_values) == 1:
        return int(baseline_values[0])
    return 2


def _case_label(case_spec: Mapping[str, int]) -> str:
    return f"n{int(case_spec['areas'])}_u{int(case_spec['users_per_area'])}_k{int(case_spec['uavs'])}"


def _count_infeasible_clusters(
    run_result: Any,
    solve_status_history: List[Dict[str, Any]],
    error_message: str,
) -> int | None:
    if run_result is not None:
        return int(
            sum(
                1
                for cluster_result in getattr(run_result, "cluster_results", [])
                if str(getattr(cluster_result, "normalized_status", "")).strip().lower() == "infeasible"
            )
        )

    if solve_status_history:
        return int(
            sum(
                1
                for record in solve_status_history
                if str(record.get("normalized_status", "")).strip().lower() == "infeasible"
            )
        )

    normalized_error = str(error_message or "").lower()
    if "insufficient battery capacity" in normalized_error:
        return 1
    if "error processing cluster" in normalized_error or "error in creating the problem for cluster" in normalized_error:
        return 1
    return 0


def _energy_used_percent(
    total_energy: float | None,
    usable_battery_capacity_wh: float,
    active_uav_count: int | None,
    configured_uav_count: int,
) -> float | None:
    if total_energy is None:
        return None
    denominator_agents = int(active_uav_count or 0)
    if denominator_agents <= 0:
        denominator_agents = int(configured_uav_count)
    usable_capacity_total = float(usable_battery_capacity_wh) * float(max(denominator_agents, 1))
    if usable_capacity_total <= 1e-9:
        return None
    return float(float(total_energy) / usable_capacity_total * 100.0)


def _run_constraint_stress_case(
    scenario_payload: Mapping[str, Any],
    *,
    fairness_tolerance: int,
    usable_battery_capacity_wh: float,
    memory_limit_bytes: int | None,
) -> Dict[str, Any]:
    builder = None
    total_started_at = time.perf_counter()

    try:
        _enforce_memory_limit(memory_limit_bytes)
        set_random_seed(int(scenario_payload["seed"]))
        config = _build_runtime_config(
            scenario_payload,
            model_name="milp",
            solver_backend="cplex",
            subtour_mode=SUBTOUR_MODE,
            warm_start_mode="none",
            stage_solution=1,
            objective_weights=None,
            fairness_tolerance=int(fairness_tolerance),
            time_step_sec=600,
            time_limit_seconds=enforce_time_limit(EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS),
            priority=EXPERIMENT_DEFAULT_PRIORITY,
        )
        config["max_battery"] = float(usable_battery_capacity_wh)

        with _suppress_nested_output():
            builder = call_builder(config, 1)
            data = builder.preprocess_generated_data(
                distance_matrix=np.array(scenario_payload["distance_matrix"], copy=True),
                centroids=copy.deepcopy(scenario_payload["centroids"]),
                depots=np.array(scenario_payload["depots"], copy=True),
                num_of_agents=int(scenario_payload["uav_count"]),
                v_hor=HORIZONTAL_VELOCITY,
                v_ver=VERTICAL_VELOCITY,
                altitude=ALTITUDE,
                coverage_time=int(scenario_payload["coverage_time"]),
                user_points=copy.deepcopy(scenario_payload["user_points"]),
            )
            builder.run_model(
                distance_matrix=np.array(scenario_payload["distance_matrix"], copy=True),
                data=data,
                cue_groups=copy.deepcopy(scenario_payload["user_points"]),
            )
            builder.gather_results()

        total_runtime_sec = time.perf_counter() - total_started_at
        if builder.latest_model_run_result is None:
            raise RuntimeError("No ModelRunResult was produced.")

        run_result = builder.latest_model_run_result
        summary = dict(getattr(run_result, "summary", {}))
        metrics = extract_common_run_metrics(run_result, scenario_payload)
        solve_status_history = list(dict(getattr(run_result, "diagnostics", {})).get("solve_status_history", []))
        return {
            "status": "ok",
            "run_result": run_result,
            "summary": summary,
            "metrics": metrics,
            "solve_status_history": solve_status_history,
            "artifact_dir": getattr(builder, "latest_artifact_dir", ""),
            "error_message": "",
            "total_runtime_sec": float(total_runtime_sec),
        }
    except Exception as exc:
        total_runtime_sec = time.perf_counter() - total_started_at
        error_message = f"{exc.__class__.__name__}: {exc}"
        summary = {}
        solve_status_history: List[Dict[str, Any]] = []
        if builder is not None:
            try:
                summary = dict(builder.build_failed_run_summary(exc))
            except Exception:
                summary = dict(getattr(builder, "latest_run_summary", {}) or {})
            solve_status_history = list(getattr(builder, "solve_status_history", []))

        return {
            "status": "failed",
            "run_result": getattr(builder, "latest_model_run_result", None) if builder is not None else None,
            "summary": summary,
            "metrics": _extract_failure_metrics(builder, error_message),
            "solve_status_history": solve_status_history,
            "artifact_dir": getattr(builder, "latest_artifact_dir", "") if builder is not None else "",
            "error_message": error_message,
            "total_runtime_sec": float(total_runtime_sec),
        }


def _build_row(
    scenario_payload: Mapping[str, Any],
    result: Dict[str, Any],
    *,
    case_label: str,
    fairness_tolerance: int,
    usable_capacity_factor: float,
    usable_battery_capacity_wh: float,
    memory_limit_bytes: int | None,
) -> Dict[str, Any]:
    run_result = result.get("run_result")
    metrics = dict(result.get("metrics", {}))
    summary = dict(result.get("summary", {}))
    solve_status_history = list(result.get("solve_status_history", []))

    raw_status = getattr(run_result, "raw_status", None) if run_result is not None else None
    normalized_status = getattr(run_result, "normalized_status", None) if run_result is not None else None
    termination_reason = getattr(run_result, "termination_reason", None) if run_result is not None else None
    if termination_reason is None and solve_status_history:
        termination_reason = str(solve_status_history[-1].get("termination_reason", "") or "")
    if termination_reason is None and result.get("error_message"):
        termination_reason = str(result["error_message"])

    active_uav_count = summary.get("num_uavs_used")
    energy_used_percent = _energy_used_percent(
        total_energy=metrics.get("total_energy"),
        usable_battery_capacity_wh=float(usable_battery_capacity_wh),
        active_uav_count=int(active_uav_count) if active_uav_count is not None else None,
        configured_uav_count=int(scenario_payload["uav_count"]),
    )

    return format_result_row(
        scenario_payload,
        "MILP",
        status=result["status"],
        extra_fields={
            "case_label": case_label,
            "environment_type": scenario_payload["env_type"],
            "coverage_time_profile": scenario_payload["coverage_time_profile"],
            "coverage_time_window_minutes": scenario_payload["coverage_time"],
            "configured_uav_count": scenario_payload["uav_count"],
            "active_uav_count": active_uav_count,
            "fairness_tolerance_baseline": int(fairness_tolerance),
            "usable_capacity_factor": float(usable_capacity_factor),
            "usable_battery_capacity_wh": float(usable_battery_capacity_wh),
            "original_battery_capacity_wh": float(MAX_BATTERY),
            "runtime_sec": result.get("total_runtime_sec"),
            "optimality_gap_percent": metrics.get("optimality_gap_percent"),
            "coverage_ratio": metrics.get("coverage_ratio", summary.get("coverage_ratio")),
            "feasible_solution_found": metrics.get("feasible_solution_found", False),
            "unvisited_nodes": metrics.get("unvisited_nodes"),
            "energy_used_percent": energy_used_percent,
            "total_energy": metrics.get("total_energy"),
            "total_distance": metrics.get("total_distance"),
            "total_travel_time": metrics.get("total_travel_time"),
            "number_of_infeasible_clusters": _count_infeasible_clusters(
                run_result,
                solve_status_history,
                result.get("error_message", ""),
            ),
            "num_clusters": summary.get("num_clusters"),
            "raw_status": raw_status or result["status"],
            "normalized_status": normalized_status or result["status"],
            "termination_reason": termination_reason or "",
            "time_limit_feasible": metrics.get("time_limit_feasible", False),
            "time_limit_no_solution": metrics.get("time_limit_no_solution", False),
            "model_build_error": metrics.get("model_build_error", False),
            "solver_error": metrics.get("solver_error", False),
            "artifact_dir": result.get("artifact_dir", ""),
            "error_message": result.get("error_message", ""),
        },
    )


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.append:
        output_path.unlink()

    fairness_tolerance = _load_fairness_baseline()
    total_runs = (
        len(EXPERIMENT_DEFAULT_SEEDS)
        * len(EXPERIMENT_G_CASES)
        * len(EXPERIMENT_G_USABLE_CAPACITY_FACTORS)
        * len(EXPERIMENT_G_ENVIRONMENTS)
        * len(EXPERIMENT_G_COVERAGE_PROFILES)
    )

    with tqdm(total=total_runs, desc="Experiment G", unit="run", dynamic_ncols=True) as progress:
        for seed in EXPERIMENT_DEFAULT_SEEDS:
            for case_spec in EXPERIMENT_G_CASES:
                case_name = _case_label(case_spec)
                for env_type in EXPERIMENT_G_ENVIRONMENTS:
                    for coverage_time_profile in EXPERIMENT_G_COVERAGE_PROFILES:
                        scenario_payload = create_scenario(
                            node_count=case_spec["areas"],
                            users_per_area=case_spec["users_per_area"],
                            uav_count=case_spec["uavs"],
                            coverage_time_profile=coverage_time_profile,
                            seed=seed,
                            scenario_name=EXPERIMENT_DEFAULT_SCENARIO,
                            objective_function=EXPERIMENT_DEFAULT_OBJECTIVE,
                            env_type=env_type,
                        )
                        for usable_capacity_factor in EXPERIMENT_G_USABLE_CAPACITY_FACTORS:
                            usable_battery_capacity_wh = float(MAX_BATTERY) * float(usable_capacity_factor)
                            progress.set_postfix_str(
                                (
                                    f"seed={seed} {case_name} {env_type} "
                                    f"{coverage_time_profile} cap={int(round(usable_capacity_factor * 100))}%"
                                ),
                                refresh=False,
                            )
                            result = _run_constraint_stress_case(
                                scenario_payload,
                                fairness_tolerance=fairness_tolerance,
                                usable_battery_capacity_wh=usable_battery_capacity_wh,
                                memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                            )
                            save_results(
                                output_path,
                                [
                                    _build_row(
                                        scenario_payload,
                                        result,
                                        case_label=case_name,
                                        fairness_tolerance=fairness_tolerance,
                                        usable_capacity_factor=usable_capacity_factor,
                                        usable_battery_capacity_wh=usable_battery_capacity_wh,
                                        memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                                    )
                                ],
                            )
                            progress.update(1)


if __name__ == "__main__":
    main()
