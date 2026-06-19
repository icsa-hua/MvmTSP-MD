from __future__ import annotations

import argparse
import csv
import math
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from dummy_app.experiments.runner_utils import create_scenario, format_result_row, run_method, save_results
from dummy_app.program_config import (
    EXPERIMENT_DEFAULT_ENV,
    EXPERIMENT_DEFAULT_MEMORY_LIMIT,
    EXPERIMENT_DEFAULT_RUN_TIME_LIMIT_SECONDS,
    EXPERIMENT_DEFAULT_SEEDS,
    EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
    EXPERIMENT_RESULTS_DIR,
    WARM_START_MODE,
)
from tqdm import tqdm


EXPERIMENT_NAME = "COOP_vs_INDI"
AREAS = [50, 100, 150, 200]
UAVS = [3, 5]
GROUND_USERS = [5, 15]
OBJECTIVES = ["energy", "coverage"]
SCENARIOS = ["cooperative", "individual"]

SOLVER_BACKEND = "cplex"
STAGE_SOLUTION = 1
SUBTOUR_MODE = "mtz"
COVERAGE_TIME_PROFILE = "medium"

RAW_METRIC_FIELDS = [
    "energy_cost_wh",
    "distance_cost_km",
    "mission_time_mins",
    "node_coverage_ratio",
    "nodes_per_kwh_npe",
    "nodes_per_hour_nph",
    "num_constraints",
    "avg_makespan_per_cluster_mins",
    "avg_data_rate_per_cluster_mb",
    "num_clusters",
    "computational_time_s",
    "memory_usage_mb",
    "total_achievable_dr_gb",
    "total_makespan_hour",
    "energy_cost_kwh",
    "dr_per_hour_gb_h_drh",
    "dr_per_energy_unit_gb_kwh_dre",
]

EMPTY_METRICS = {
    "runtime_sec": None,
    "coverage_ratio": None,
    "total_distance": None,
    "total_energy": None,
    "total_travel_time": None,
    "objective_value": None,
    "solver_objective_value": None,
    "summary_objective_value": None,
    "best_bound": None,
    "optimality_gap_percent": None,
    "feasible_solution_found": False,
    "time_to_first_feasible_sec": None,
    "branch_and_bound_nodes": None,
    "memory_usage_mb": None,
    "num_constraints": None,
    "num_variables": None,
    "num_binary_variables": None,
    "num_continuous_variables": None,
    "artifact_dir": "",
    "error_message": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment COOP_vs_INDI: cooperative vs individual MILP with CPLEX, "
            "GA warm start, single-stage objective, and MTZ subtour constraints."
        )
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(EXPERIMENT_RESULTS_DIR) / "experiment_COOP_vs_INDI.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=str(Path(EXPERIMENT_RESULTS_DIR) / "experiment_COOP_vs_INDI_summary.csv"),
    )
    parser.add_argument("--append", action="store_true", help="Append to the raw CSV instead of replacing it.")
    parser.add_argument("--areas", type=int, nargs="+", default=AREAS)
    parser.add_argument("--uavs", type=int, nargs="+", default=UAVS)
    parser.add_argument("--ground-users", type=int, nargs="+", default=GROUND_USERS)
    parser.add_argument("--objectives", type=str, nargs="+", default=OBJECTIVES, choices=OBJECTIVES)
    parser.add_argument("--scenarios", type=str, nargs="+", default=SCENARIOS, choices=SCENARIOS)
    parser.add_argument("--seeds", type=int, nargs="+", default=EXPERIMENT_DEFAULT_SEEDS)
    parser.add_argument(
        "--time-limit-seconds",
        type=int,
        default=EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
        help="Per-cluster CPLEX time limit; 0 disables the solver limit.",
    )
    parser.add_argument(
        "--run-time-limit-seconds",
        type=int,
        default=EXPERIMENT_DEFAULT_RUN_TIME_LIMIT_SECONDS,
        help="Whole-run wall-clock limit; 0 disables the run limit.",
    )
    return parser.parse_args()


def _scale_category(node_count: int) -> str:
    return "small_scale" if int(node_count) <= 100 else "large_scale"


def _safe_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) <= 1e-12:
        return None
    return float(numerator) / float(denominator)


def _summary_from_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    run_result = result.get("run_result")
    if run_result is None:
        return {}
    return dict(getattr(run_result, "summary", {}) or {})


def _extract_requested_metrics(result: Mapping[str, Any]) -> Dict[str, Any]:
    source_metrics = dict(result.get("metrics", {}) or {})
    metrics = {
        field: source_metrics.get(field, default_value)
        for field, default_value in EMPTY_METRICS.items()
    }
    summary = _summary_from_result(result)

    energy_wh = _safe_float(summary.get("energy_cost"))
    if energy_wh is None:
        energy_wh = _safe_float(metrics.get("total_energy"))
    distance_km = _safe_float(summary.get("distance"))
    if distance_km is None:
        distance_km = _safe_float(metrics.get("total_distance"))
    mission_time_mins = _safe_float(summary.get("mission_time_cost"))
    if mission_time_mins is None:
        mission_time_mins = _safe_float(metrics.get("total_travel_time"))

    coverage_ratio = _safe_float(summary.get("coverage_ratio"))
    if coverage_ratio is None:
        coverage_ratio = _safe_float(metrics.get("coverage_ratio"))

    total_nodes = _safe_float(summary.get("total_nodes_visited"))
    if total_nodes is None:
        total_nodes = _safe_float(summary.get("covered_nodes"))

    total_data_mb = _safe_float(summary.get("total_data_rate"))
    average_data_mb = _safe_float(summary.get("average_data_rate_per_cluster"))
    makespan_mins = _safe_float(summary.get("makespan"))
    average_makespan_mins = _safe_float(summary.get("average_makespan_per_cluster"))
    energy_kwh = _safe_div(energy_wh, 1000.0)
    mission_time_hours = _safe_div(mission_time_mins, 60.0)
    makespan_hours = _safe_div(makespan_mins, 60.0)
    total_data_gb = _safe_div(total_data_mb, 1024.0)

    extracted = {
        "energy_cost_wh": energy_wh,
        "distance_cost_km": distance_km,
        "mission_time_mins": mission_time_mins,
        "node_coverage_ratio": coverage_ratio,
        "nodes_per_kwh_npe": _safe_div(total_nodes, energy_kwh),
        "nodes_per_hour_nph": _safe_div(total_nodes, mission_time_hours),
        "num_constraints": metrics.get("num_constraints") if metrics.get("num_constraints") is not None else summary.get("num_constraints"),
        "avg_makespan_per_cluster_mins": average_makespan_mins,
        "avg_data_rate_per_cluster_mb": average_data_mb,
        "num_clusters": summary.get("num_clusters"),
        "computational_time_s": metrics.get("runtime_sec") if metrics.get("runtime_sec") is not None else summary.get("solve_time_seconds"),
        "memory_usage_mb": metrics.get("memory_usage_mb") if metrics.get("memory_usage_mb") is not None else summary.get("memory_usage_mb"),
        "total_achievable_dr_gb": total_data_gb,
        "total_makespan_hour": makespan_hours,
        "energy_cost_kwh": energy_kwh,
        "dr_per_hour_gb_h_drh": _safe_div(total_data_gb, makespan_hours),
        "dr_per_energy_unit_gb_kwh_dre": _safe_div(total_data_gb, energy_kwh),
    }
    return {**metrics, **extracted}


def _mean(values: Iterable[Any]) -> float | None:
    numeric_values = [_safe_float(value) for value in values]
    numeric_values = [value for value in numeric_values if value is not None]
    if not numeric_values:
        return None
    return float(sum(numeric_values)) / float(len(numeric_values))


def _read_raw_rows(raw_path: Path) -> List[Dict[str, Any]]:
    if not raw_path.exists():
        return []
    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_category_averages(raw_path: Path, summary_path: Path) -> None:
    rows = _read_raw_rows(raw_path)
    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("scale_category", "")),
            str(row.get("scenario_type", "")),
            str(row.get("objective_function", "")),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (scale_category, scenario_type, objective), group_rows in sorted(grouped.items()):
        successful_rows = [row for row in group_rows if str(row.get("status", "")) == "ok"]
        source_rows = successful_rows or group_rows
        summary_row = {
            "experiment_name": EXPERIMENT_NAME,
            "scale_category": scale_category,
            "scenario_type": scenario_type,
            "objective_function": objective,
            "solver_backend": SOLVER_BACKEND,
            "warm_start_mode": WARM_START_MODE,
            "stage_solution": STAGE_SOLUTION,
            "subtour_mode": SUBTOUR_MODE,
            "n_runs": len(group_rows),
            "successful_runs": len(successful_rows),
            "failed_runs": len(group_rows) - len(successful_rows),
        }
        for field in RAW_METRIC_FIELDS:
            summary_row[f"avg_{field}"] = _mean(row.get(field) for row in source_rows)
        summary_rows.append(summary_row)

    if summary_path.exists():
        summary_path.unlink()
    save_results(summary_path, summary_rows)


def _scenario_payload_for_run(
    *,
    seed: int,
    node_count: int,
    users_per_area: int,
    uav_count: int,
    scenario_type: str,
    objective: str,
) -> Dict[str, Any]:
    scenario_payload = create_scenario(
        node_count=node_count,
        users_per_area=users_per_area,
        uav_count=uav_count,
        coverage_time_profile=COVERAGE_TIME_PROFILE,
        seed=seed,
        scenario_name=scenario_type,
        objective_function=objective,
        env_type=EXPERIMENT_DEFAULT_ENV,
    )
    scenario_payload = dict(scenario_payload)
    scenario_payload["scenario_id"] = f"{scenario_payload['scenario_id']}_{scenario_type}_{objective}"
    return scenario_payload


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    if output_path.exists() and not args.append:
        output_path.unlink()
    if summary_path.exists() and not args.append:
        summary_path.unlink()

    total_runs = (
        len(args.seeds)
        * len(args.areas)
        * len(args.ground_users)
        * len(args.uavs)
        * len(args.objectives)
        * len(args.scenarios)
    )

    with tqdm(total=total_runs, desc=f"Experiment {EXPERIMENT_NAME}", unit="run", dynamic_ncols=True) as progress:
        for seed, node_count, users_per_area, uav_count, objective, scenario_type in product(
            args.seeds,
            args.areas,
            args.ground_users,
            args.uavs,
            args.objectives,
            args.scenarios,
        ):
            progress.set_postfix_str(
                (
                    f"seed={seed} scale={_scale_category(node_count)} n={node_count} "
                    f"users={users_per_area} k={uav_count} {objective} {scenario_type}"
                ),
                refresh=False,
            )
            scenario_payload = _scenario_payload_for_run(
                seed=seed,
                node_count=node_count,
                users_per_area=users_per_area,
                uav_count=uav_count,
                scenario_type=scenario_type,
                objective=objective,
            )
            result = run_method(
                scenario_payload,
                "MILP + GA warmup",
                model_name="milp",
                solver_backend=SOLVER_BACKEND,
                subtour_mode=SUBTOUR_MODE,
                warm_start_mode=WARM_START_MODE,
                stage_solution=STAGE_SOLUTION,
                time_limit_seconds=args.time_limit_seconds,
                run_time_limit_seconds=args.run_time_limit_seconds,
                memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
            )
            requested_metrics = _extract_requested_metrics(result)
            requested_metrics["error_message"] = result.get("error_message", "")
            requested_metrics["artifact_dir"] = result.get("artifact_dir", "")
            row = format_result_row(
                scenario_payload,
                "MILP + GA warmup",
                status=result["status"],
                extra_fields={
                    "experiment_name": EXPERIMENT_NAME,
                    "scale_category": _scale_category(node_count),
                    "scenario_type": scenario_type,
                    "objective_function": objective,
                    "solver_backend": SOLVER_BACKEND,
                    "warm_start_mode": WARM_START_MODE,
                    "stage_solution": STAGE_SOLUTION,
                    "subtour_mode": SUBTOUR_MODE,
                    **requested_metrics,
                },
            )
            save_results(output_path, [row])
            progress.update(1)

    write_category_averages(output_path, summary_path)


if __name__ == "__main__":
    main()
