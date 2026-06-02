from __future__ import annotations

import copy
import csv
import json
import logging
import random
import resource
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from dummy_app.core.statuses import compute_relative_gap_percent
from dummy_app.designs.voronoi_map import MapGenerator
from dummy_app.tools.common import call_builder
from dummy_app.tools.logger import logger as app_logger
from dummy_app.experiments.metrics import (
    aggregate_agent_routes,
    compute_coverage_ratio,
    compute_jain_fairness,
    compute_std,
    count_unvisited_nodes,
)
from dummy_app.program_config import (
    ALTITUDE,
    EXPERIMENT_COVERAGE_TIME_PROFILES,
    EXPERIMENT_DEFAULT_ENV,
    EXPERIMENT_DEFAULT_MEMORY_LIMIT,
    EXPERIMENT_DEFAULT_OBJECTIVE,
    EXPERIMENT_DEFAULT_PRIORITY,
    EXPERIMENT_DEFAULT_SCENARIO,
    EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
    EXPERIMENT_OBJECTIVE_WEIGHT_PROFILES,
    HIGH_BOUND,
    HORIZONTAL_VELOCITY,
    LATITUDE_COORDS,
    LONGITUDE_COORDS,
    LOW_BOUND,
    MAX_BATTERY,
    NUMBER_OF_USERS,
    OBJECTIVE_STRATEGY,
    PRIORITY,
    SCENARIO_CONSTRAINT_SET,
    SOLVER_BACKEND,
    SUBTOUR_MODE,
    SUBTOUR_STRATEGY,
    VERTICAL_VELOCITY,
)


METHOD_MODEL_MAP = {
    "MILP": "milp",
    "Greedy NN": "greedy_nn",
    "Static Partitioning + Greedy NN": "greedy_partition_nn",
    "Genetic Algorithm": "ga",
    "ALNS": "alns",
}


def set_random_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def enforce_time_limit(limit_seconds: int | None) -> int | None:
    return None if limit_seconds in {None, 0} else int(limit_seconds)


def _enforce_memory_limit(max_memory_bytes: int | None) -> None:
    if not max_memory_bytes:
        return
    try:
        resource.setrlimit(resource.RLIMIT_AS, (int(max_memory_bytes), int(max_memory_bytes)))
    except Exception:
        return


@contextmanager
def _suppress_nested_output():
    previous_level = app_logger.level
    previous_handler_levels = [handler.level for handler in app_logger.handlers]
    devnull_path = Path("/dev/null")
    with devnull_path.open("w", encoding="utf-8") as sink:
        try:
            app_logger.setLevel(logging.WARNING)
            for handler in app_logger.handlers:
                handler.setLevel(logging.WARNING)
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
        finally:
            app_logger.setLevel(previous_level)
            for handler, level in zip(app_logger.handlers, previous_handler_levels):
                handler.setLevel(level)


def create_scenario(
    node_count: int,
    users_per_area: int = NUMBER_OF_USERS,
    uav_count: int = 3,
    coverage_time_profile: str = "medium",
    seed: int = 42,
    scenario_name: str = EXPERIMENT_DEFAULT_SCENARIO,
    objective_function: str = EXPERIMENT_DEFAULT_OBJECTIVE,
    env_type: str = EXPERIMENT_DEFAULT_ENV,
) -> Dict[str, Any]:
    set_random_seed(seed)
    coverage_time = int(EXPERIMENT_COVERAGE_TIME_PROFILES[str(coverage_time_profile)])
    map_generator = MapGenerator(
        num_areas=int(node_count),
        users_per_area=int(users_per_area),
        lon=LONGITUDE_COORDS,
        lat=LATITUDE_COORDS,
        low=LOW_BOUND,
        high=HIGH_BOUND,
        seed=int(seed),
    )
    regions, centroids, user_points, depots, distance_matrix, all_users = map_generator.create_environment(
        show_map=False,
        show_3d_map=False,
    )
    depot_list = [int(depot) for depot in np.array(depots).tolist()]
    all_nodes = sorted(int(node_id) for node_id in user_points.keys())
    target_nodes = [int(node_id) for node_id in all_nodes if int(node_id) not in set(depot_list)]
    scenario_id = (
        f"seed{int(seed)}_n{int(node_count)}_u{int(users_per_area)}"
        f"_k{int(uav_count)}_{coverage_time_profile}"
    )
    return {
        "scenario_id": scenario_id,
        "node_count": int(node_count),
        "users_per_area": int(users_per_area),
        "uav_count": int(uav_count),
        "battery_capacity": float(MAX_BATTERY),
        "coverage_time_profile": str(coverage_time_profile),
        "coverage_time": int(coverage_time),
        "seed": int(seed),
        "scenario_name": str(scenario_name),
        "objective_function": str(objective_function),
        "env_type": str(env_type),
        "regions": regions,
        "centroids": centroids,
        "user_points": user_points,
        "depots": np.array(depot_list, dtype=int),
        "depot_ids": depot_list,
        "distance_matrix": np.array(distance_matrix, copy=True),
        "all_users": all_users,
        "all_nodes": all_nodes,
        "target_nodes": target_nodes,
    }


def _build_runtime_config(
    scenario_payload: Mapping[str, Any],
    *,
    model_name: str,
    solver_backend: str,
    warm_start_mode: str,
    stage_solution: int,
    objective_weights: Optional[Dict[str, float]],
    fairness_tolerance: int,
    time_step_sec: int,
    time_limit_seconds: int | None,
    priority: str,
) -> Dict[str, Any]:
    config = {
        "model_name": model_name,
        "genetic_algorithm": warm_start_mode == "ga",
        "env_type": scenario_payload["env_type"],
        "max_battery": float(scenario_payload["battery_capacity"]),
        "max_coverage_time": int(scenario_payload["coverage_time"]),
        "scenario": scenario_payload["scenario_name"],
        "enable_ga": "yes" if warm_start_mode == "ga" else "no",
        "objective_function": scenario_payload["objective_function"],
        "stage_solution": int(stage_solution),
        "priority": priority,
        "validate": False,
        "solver_backend": solver_backend,
        "subtour_mode": SUBTOUR_MODE,
        "subtour_strategy": SUBTOUR_STRATEGY,
        "objective_strategy": OBJECTIVE_STRATEGY,
        "scenario_constraint_set": SCENARIO_CONSTRAINT_SET,
        "solver_time_limit_seconds": time_limit_seconds,
        "warm_start_mode": warm_start_mode,
        "random_seed": int(scenario_payload["seed"]),
        "solver_seed": int(scenario_payload["seed"]),
        "fairness_tolerance": int(fairness_tolerance),
        "time_step_sec": int(time_step_sec),
        "NUMBER_OF_AGENTS": int(scenario_payload["uav_count"]),
        "NUMBER_OF_USERS": int(scenario_payload["users_per_area"]),
        "NUMBER_OF_AREAS": int(scenario_payload["node_count"]),
        "altitude": ALTITUDE,
        "learning_enabled": False,
        "learning_alpha": 0.75,
    }
    if objective_weights:
        config["objective_weights"] = dict(objective_weights)
    return config


def _aggregate_per_uav_metrics(run_result: Any) -> tuple[Dict[str, float], Dict[str, float], float]:
    distance_per_uav: Dict[str, float] = defaultdict(float)
    energy_per_uav: Dict[str, float] = defaultdict(float)
    total_travel_time = 0.0
    for cluster_result in run_result.cluster_results:
        for agent_id, metrics in dict(cluster_result.agent_metrics).items():
            agent_key = str(agent_id)
            distance_per_uav[agent_key] += float(metrics.get("distance", 0.0))
            energy_per_uav[agent_key] += float(metrics.get("energy", 0.0))
            total_travel_time += float(metrics.get("travel_time", 0.0))
    return dict(distance_per_uav), dict(energy_per_uav), float(total_travel_time)


def _cluster_first_feasible_time(cluster_result: Any) -> float | None:
    diagnostics = dict(cluster_result.diagnostics)
    first_feasible_time = diagnostics.get("first_feasible_time_seconds")
    if first_feasible_time is not None:
        return float(first_feasible_time)
    if cluster_result.normalized_status in {"optimal", "feasible", "feasible_time_limit"}:
        return float(cluster_result.elapsed_time_seconds or 0.0)
    return None


def _aggregate_first_feasible_time(run_result: Any) -> float | None:
    elapsed_prefix = 0.0
    aggregate_time = None
    for cluster_result in run_result.cluster_results:
        cluster_time = _cluster_first_feasible_time(cluster_result)
        if cluster_time is None:
            return None
        aggregate_time = elapsed_prefix + float(cluster_time)
        elapsed_prefix += float(cluster_result.elapsed_time_seconds or 0.0)
    return aggregate_time


def _sum_branch_and_bound_nodes(run_result: Any) -> int:
    return int(
        sum(int(dict(cluster_result.diagnostics).get("explored_bnb_nodes", 0) or 0) for cluster_result in run_result.cluster_results)
    )


def _aggregate_solver_incumbent(run_result: Any) -> float | None:
    incumbents = [cluster_result.incumbent_value for cluster_result in run_result.cluster_results]
    if not incumbents or any(value is None for value in incumbents):
        return None
    return float(sum(float(value) for value in incumbents))


def _aggregate_solver_best_bound(run_result: Any) -> float | None:
    best_bounds = [cluster_result.best_bound for cluster_result in run_result.cluster_results]
    if not best_bounds or any(value is None for value in best_bounds):
        return None
    return float(sum(float(value) for value in best_bounds))


def _aggregate_history_incumbent(status_history: Iterable[Mapping[str, Any]]) -> float | None:
    records = list(status_history)
    incumbents = [record.get("incumbent_value") for record in records]
    if not records or any(value is None for value in incumbents):
        return None
    return float(sum(float(value) for value in incumbents))


def _aggregate_history_best_bound(status_history: Iterable[Mapping[str, Any]]) -> float | None:
    records = list(status_history)
    best_bounds = [record.get("best_bound") for record in records]
    if not records or any(value is None for value in best_bounds):
        return None
    return float(sum(float(value) for value in best_bounds))


def _failure_category_flags(error_message: str, status_history: Iterable[Mapping[str, Any]]) -> Dict[str, bool]:
    records = list(status_history)
    time_limit_feasible = any(
        bool(record.get("time_limit_reached")) and record.get("incumbent_value") is not None for record in records
    )
    time_limit_no_solution = any(
        bool(record.get("time_limit_reached")) and record.get("incumbent_value") is None for record in records
    )
    normalized_error = str(error_message or "")
    model_build_error = any(
        marker in normalized_error
        for marker in (
            "Error in creating the problem for Cluster",
            "Error processing cluster",
        )
    )
    solver_error = bool(normalized_error) and not (time_limit_feasible or time_limit_no_solution or model_build_error)
    return {
        "time_limit_feasible": bool(time_limit_feasible),
        "time_limit_no_solution": bool(time_limit_no_solution),
        "model_build_error": bool(model_build_error),
        "solver_error": bool(solver_error),
    }


def _extract_failure_metrics(builder: Any, error_message: str) -> Dict[str, Any]:
    if builder is None:
        flags = _failure_category_flags(error_message, [])
        return {
            "objective_value": None,
            "summary_objective_value": None,
            "best_bound": None,
            "optimality_gap_percent": None,
            "feasible_solution_found": False,
            "time_to_first_feasible_sec": None,
            "branch_and_bound_nodes": None,
            "memory_usage_mb": None,
            **flags,
        }

    status_history = list(getattr(builder, "solve_status_history", []))
    flags = _failure_category_flags(error_message, status_history)
    objective_value = _aggregate_history_incumbent(status_history)
    best_bound = _aggregate_history_best_bound(status_history)
    feasible_solution_found = bool(status_history) and all(record.get("incumbent_value") is not None for record in status_history)

    first_feasible_times = [
        float(record.get("first_feasible_time_seconds"))
        for record in status_history
        if record.get("first_feasible_time_seconds") is not None
    ]
    return {
        "objective_value": objective_value,
        "summary_objective_value": None,
        "best_bound": best_bound,
        "optimality_gap_percent": compute_relative_gap_percent(objective_value, best_bound),
        "feasible_solution_found": feasible_solution_found,
        "time_to_first_feasible_sec": min(first_feasible_times) if first_feasible_times else None,
        "branch_and_bound_nodes": int(
            sum(int(record.get("explored_bnb_nodes", 0) or 0) for record in status_history)
        ) if status_history else None,
        "memory_usage_mb": float(getattr(builder.metrics, "memory_usage", 0.0) or 0.0) if getattr(builder, "metrics", None) else None,
        **flags,
    }


def extract_common_run_metrics(run_result: Any, scenario_payload: Mapping[str, Any]) -> Dict[str, Any]:
    routes = aggregate_agent_routes(run_result.cluster_results)
    distance_per_uav, energy_per_uav, total_travel_time = _aggregate_per_uav_metrics(run_result)
    distance_values = list(distance_per_uav.values())
    energy_values = list(energy_per_uav.values())
    summary = dict(run_result.summary)
    solver_objective_value = _aggregate_solver_incumbent(run_result)
    solver_best_bound = _aggregate_solver_best_bound(run_result)
    summary_objective_value = getattr(run_result, "summary_objective_value", None)
    if summary_objective_value is None:
        summary_objective_value = summary.get("objective_value")
    comparable_objective_value = solver_objective_value if solver_objective_value is not None else run_result.objective_value
    feasible_solution_found = bool(run_result.cluster_results) and all(
        cluster_result.incumbent_value is not None for cluster_result in run_result.cluster_results
    )
    time_limit_feasible = any(
        bool(dict(cluster_result.diagnostics).get("time_limit_reached")) and cluster_result.incumbent_value is not None
        for cluster_result in run_result.cluster_results
    )
    time_limit_no_solution = any(
        bool(dict(cluster_result.diagnostics).get("time_limit_reached")) and cluster_result.incumbent_value is None
        for cluster_result in run_result.cluster_results
    )
    model_build_error = False
    solver_error = any(
        getattr(cluster_result, "termination_reason", "") == "solver_error"
        for cluster_result in run_result.cluster_results
    )

    coverage_ratio = compute_coverage_ratio(routes, scenario_payload["target_nodes"])
    unvisited_nodes = count_unvisited_nodes(routes, scenario_payload["target_nodes"])
    return {
        "runtime_sec": float(run_result.elapsed_time_seconds or summary.get("solve_time_seconds", 0.0) or 0.0),
        "coverage_ratio": float(coverage_ratio),
        "total_distance": float(sum(distance_values)),
        "total_energy": float(sum(energy_values)),
        "total_travel_time": float(total_travel_time),
        "max_route_distance": float(max(distance_values, default=0.0)),
        "max_route_energy": float(max(energy_values, default=0.0)),
        "distance_std_across_uavs": float(compute_std(distance_values)),
        "energy_std_across_uavs": float(compute_std(energy_values)),
        "jain_fairness_distance": float(compute_jain_fairness(distance_values)),
        "jain_fairness_energy": float(compute_jain_fairness(energy_values)),
        "unvisited_nodes": int(unvisited_nodes),
        "objective_value": comparable_objective_value,
        "solver_objective_value": solver_objective_value,
        "summary_objective_value": summary_objective_value,
        "best_bound": solver_best_bound,
        "optimality_gap_percent": compute_relative_gap_percent(comparable_objective_value, solver_best_bound),
        "feasible_solution_found": feasible_solution_found,
        "time_to_first_feasible_sec": _aggregate_first_feasible_time(run_result),
        "branch_and_bound_nodes": _sum_branch_and_bound_nodes(run_result),
        "memory_usage_mb": summary.get("memory_usage_mb"),
        "time_limit_feasible": time_limit_feasible,
        "time_limit_no_solution": time_limit_no_solution,
        "model_build_error": model_build_error,
        "solver_error": solver_error,
        "status": run_result.normalized_status,
        "distance_per_uav": json.dumps(distance_per_uav, sort_keys=True),
        "energy_per_uav": json.dumps(energy_per_uav, sort_keys=True),
    }


def run_method(
    scenario_payload: Mapping[str, Any],
    method_name: str,
    *,
    model_name: str | None = None,
    solver_backend: str = SOLVER_BACKEND,
    warm_start_mode: str = "none",
    stage_solution: int = 1,
    objective_weights: Optional[Dict[str, float]] = None,
    fairness_tolerance: int = 2,
    time_step_sec: int = 600,
    time_limit_seconds: int | None = EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
    memory_limit_bytes: int | None = EXPERIMENT_DEFAULT_MEMORY_LIMIT,
    priority: str = PRIORITY,
) -> Dict[str, Any]:
    effective_model_name = model_name or METHOD_MODEL_MAP.get(method_name, "milp")
    builder = None
    try:
        _enforce_memory_limit(memory_limit_bytes)
        set_random_seed(int(scenario_payload["seed"]))
        config = _build_runtime_config(
            scenario_payload,
            model_name=effective_model_name,
            solver_backend=solver_backend,
            warm_start_mode=warm_start_mode,
            stage_solution=stage_solution,
            objective_weights=objective_weights,
            fairness_tolerance=fairness_tolerance,
            time_step_sec=time_step_sec,
            time_limit_seconds=enforce_time_limit(time_limit_seconds),
            priority=priority,
        )
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
        if builder.latest_model_run_result is None:
            raise RuntimeError("No ModelRunResult was produced.")

        return {
            "status": "ok",
            "method_name": method_name,
            "model_name": effective_model_name,
            "solver_backend": solver_backend,
            "warm_start_mode": warm_start_mode,
            "run_result": builder.latest_model_run_result,
            "artifact_dir": builder.latest_artifact_dir,
            "metrics": extract_common_run_metrics(builder.latest_model_run_result, scenario_payload),
        }
    except Exception as exc:
        error_message = f"{exc.__class__.__name__}: {exc}"
        return {
            "status": "failed",
            "method_name": method_name,
            "model_name": effective_model_name,
            "solver_backend": solver_backend,
            "warm_start_mode": warm_start_mode,
            "run_result": None,
            "artifact_dir": getattr(builder, "latest_artifact_dir", "") if builder is not None else "",
            "error_message": error_message,
            "metrics": _extract_failure_metrics(builder, error_message),
        }


def format_result_row(
    scenario_payload: Mapping[str, Any],
    method_name: str,
    *,
    status: str,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "scenario_id": scenario_payload["scenario_id"],
        "node_count": int(scenario_payload["node_count"]),
        "users_per_area": int(scenario_payload["users_per_area"]),
        "uav_count": int(scenario_payload["uav_count"]),
        "coverage_time_profile": str(scenario_payload["coverage_time_profile"]),
        "method_name": method_name,
        "seed": int(scenario_payload["seed"]),
        "status": status,
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def save_results(csv_path: str | Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    fieldnames = list(rows[0].keys())
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
