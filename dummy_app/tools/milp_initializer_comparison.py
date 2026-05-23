from __future__ import annotations

import argparse
import copy
import csv
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from dummy_app.designs.voronoi_map import MapGenerator
from dummy_app.tools.common import call_builder
from program_config import (
    ALTITUDE,
    ENVIRONMENT_OPTIONS,
    HIGH_BOUND,
    HORIZONTAL_VELOCITY,
    LATITUDE_COORDS,
    LONGITUDE_COORDS,
    LOW_BOUND,
    MAX_BATTERY,
    MAX_COVERAGE_TIME,
    NUMBER_OF_AGENTS,
    NUMBER_OF_AREAS,
    NUMBER_OF_USERS,
    OBJECTIVE_OPTIONS,
    OBJECTIVE_STRATEGY,
    PRIORITY,
    SCENARIO_CONSTRAINT_SET,
    SCENARIO_OPTIONS,
    SOLVER_BACKEND,
    SOLVER_TIME_LIMIT,
    STAGE_OPTIONS,
    SUBTOUR_STRATEGY,
    VERTICAL_VELOCITY,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MILP runs with no initializer, ALNS initializer, and GA initializer.")
    parser.add_argument("--scenario", type=str, default=SCENARIO_OPTIONS[0], choices=SCENARIO_OPTIONS)
    parser.add_argument("--objective", type=str, default=OBJECTIVE_OPTIONS[0], choices=OBJECTIVE_OPTIONS)
    parser.add_argument("--env", type=str, default=ENVIRONMENT_OPTIONS[0], choices=ENVIRONMENT_OPTIONS)
    parser.add_argument("--stage_solution", type=int, default=STAGE_OPTIONS[0], choices=STAGE_OPTIONS)
    parser.add_argument("--num_agents", type=int, default=NUMBER_OF_AGENTS)
    parser.add_argument("--num_users", type=int, default=NUMBER_OF_USERS)
    parser.add_argument("--num_areas", type=int, default=NUMBER_OF_AREAS)
    parser.add_argument("--max_battery", type=float, default=MAX_BATTERY)
    parser.add_argument("--max_coverage_time", type=int, default=MAX_COVERAGE_TIME)
    parser.add_argument("--solver_backend", type=str, default=SOLVER_BACKEND)
    parser.add_argument("--time_limit_seconds", type=int, default=SOLVER_TIME_LIMIT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--output_csv",
        type=str,
        default="assets/results/milp_initializer_comparison/summary.csv",
        help="Path to the summary CSV file.",
    )
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_base_config(args: argparse.Namespace, warm_start_mode: str) -> Dict[str, Any]:
    enable_ga = "yes" if warm_start_mode == "ga" else "no"
    return {
        "model_name": "milp",
        "genetic_algorithm": enable_ga == "yes",
        "env_type": args.env,
        "max_battery": args.max_battery,
        "max_coverage_time": args.max_coverage_time,
        "scenario": args.scenario,
        "enable_ga": enable_ga,
        "objective_function": args.objective,
        "stage_solution": args.stage_solution,
        "priority": PRIORITY,
        "validate": False,
        "solver_backend": args.solver_backend,
        "subtour_strategy": SUBTOUR_STRATEGY,
        "objective_strategy": OBJECTIVE_STRATEGY,
        "scenario_constraint_set": SCENARIO_CONSTRAINT_SET,
        "solver_time_limit_seconds": None if args.time_limit_seconds == 0 else args.time_limit_seconds,
        "warm_start_mode": warm_start_mode,
        "random_seed": args.seed,
        "solver_seed": args.seed,
        "NUMBER_OF_AGENTS": args.num_agents,
        "NUMBER_OF_USERS": args.num_users,
        "NUMBER_OF_AREAS": args.num_areas,
        "altitude": ALTITUDE,
        "learning_enabled": False,
        "learning_alpha": 0.75,
    }


def generate_shared_instance(args: argparse.Namespace) -> Dict[str, Any]:
    map_generator = MapGenerator(
        num_areas=args.num_areas,
        users_per_area=args.num_users,
        lon=LONGITUDE_COORDS,
        lat=LATITUDE_COORDS,
        low=LOW_BOUND,
        high=HIGH_BOUND,
        seed=args.seed,
    )
    regions, centroids, user_points, depots, distance_matrix, all_users = map_generator.create_environment(
        show_map=False,
        show_3d_map=False,
    )
    return {
        "regions": regions,
        "centroids": centroids,
        "user_points": user_points,
        "depots": np.array(depots) if isinstance(depots, list) else depots,
        "distance_matrix": distance_matrix,
        "all_users": all_users,
    }


def _cluster_progress_events(cluster_result: Any) -> List[Dict[str, Any]]:
    diagnostics = dict(getattr(cluster_result, "diagnostics", {}))
    events = list(diagnostics.get("progress_events", []))
    if not events and cluster_result.objective_value is not None:
        gap_percent = float(cluster_result.relative_gap) * 100.0 if cluster_result.relative_gap is not None else None
        events = [
            {
                "elapsed_time_seconds": float(cluster_result.elapsed_time_seconds or 0.0),
                "incumbent_value": float(cluster_result.objective_value),
                "best_bound": float(cluster_result.best_bound) if cluster_result.best_bound is not None else None,
                "relative_gap_percent": gap_percent,
                "active_bnb_nodes": diagnostics.get("active_bnb_nodes", 0),
                "explored_bnb_nodes": diagnostics.get("explored_bnb_nodes", 0),
                "raw_line": "synthetic_final_event",
            }
        ]
    return events


def _time_to_target(events: Iterable[Dict[str, Any]], key: str, target: float | None) -> float | None:
    if target is None:
        return None
    for event in events:
        value = event.get(key)
        if value is None:
            continue
        if float(value) <= float(target) + 1e-9:
            return float(event["elapsed_time_seconds"])
    return None


def run_strategy(args: argparse.Namespace, label: str, warm_start_mode: str, shared_instance: Dict[str, Any]) -> Dict[str, Any]:
    config = build_base_config(args, warm_start_mode)
    builder = call_builder(config, args.trials)
    data = builder.preprocess_generated_data(
        distance_matrix=np.array(shared_instance["distance_matrix"], copy=True),
        centroids=copy.deepcopy(shared_instance["centroids"]),
        depots=np.array(shared_instance["depots"], copy=True),
        num_of_agents=args.num_agents,
        v_hor=HORIZONTAL_VELOCITY,
        v_ver=VERTICAL_VELOCITY,
        altitude=ALTITUDE,
        coverage_time=args.max_coverage_time,
        user_points=copy.deepcopy(shared_instance["user_points"]),
    )
    builder.run_model(
        distance_matrix=np.array(shared_instance["distance_matrix"], copy=True),
        data=data,
        cue_groups=copy.deepcopy(shared_instance["user_points"]),
    )
    builder.gather_results()

    run_result = builder.latest_model_run_result
    if run_result is None:
        raise RuntimeError(f"Run '{label}' did not produce a ModelRunResult.")

    cluster_rows: List[Dict[str, Any]] = []
    for cluster_index, cluster_result in enumerate(run_result.cluster_results):
        diagnostics = dict(cluster_result.diagnostics)
        warm_start_summary = dict(diagnostics.get("warm_start_summary", {}))
        events = _cluster_progress_events(cluster_result)
        cluster_rows.append(
            {
                "strategy": label,
                "warm_start_mode": warm_start_mode,
                "cluster_id": int(cluster_result.cluster_id),
                "cluster_sequence_index": int(cluster_index),
                "first_feasible_time_seconds": diagnostics.get("first_feasible_time_seconds"),
                "first_optimality_gap_percent": diagnostics.get("first_optimality_gap_percent"),
                "final_objective_value": cluster_result.objective_value,
                "best_bound": cluster_result.best_bound,
                "final_relative_gap_percent": float(cluster_result.relative_gap) * 100.0 if cluster_result.relative_gap is not None else None,
                "elapsed_time_seconds": float(cluster_result.elapsed_time_seconds or 0.0),
                "explored_bnb_nodes": diagnostics.get("explored_bnb_nodes"),
                "active_bnb_nodes": diagnostics.get("active_bnb_nodes"),
                "optimality_proven": diagnostics.get("optimality_proven"),
                "warm_start_objective_quality": warm_start_summary.get("objective_value"),
                "warm_start_timeframe_estimate": warm_start_summary.get("timeframe_estimate"),
                "solver_log_path": diagnostics.get("solver_log_path", ""),
                "progress_events": events,
            }
        )

    return {
        "label": label,
        "warm_start_mode": warm_start_mode,
        "artifact_dir": builder.latest_artifact_dir,
        "run_result": asdict(run_result),
        "cluster_rows": cluster_rows,
    }


def aggregate_summary_row(
    args: argparse.Namespace,
    strategy_payload: Dict[str, Any],
    cluster_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    run_result = strategy_payload["run_result"]
    objective_value = run_result.get("objective_value")
    best_bound = run_result.get("best_bound")
    complete_feasible_time = None
    elapsed_prefix = 0.0
    for cluster_row in sorted(cluster_rows, key=lambda item: item["cluster_sequence_index"]):
        first_feasible = cluster_row.get("first_feasible_time_seconds")
        if first_feasible is None:
            complete_feasible_time = None
            break
        complete_feasible_time = elapsed_prefix + float(first_feasible)
        elapsed_prefix += float(cluster_row.get("elapsed_time_seconds") or 0.0)

    def _mean(values: Iterable[Any]) -> float | None:
        series = [float(value) for value in values if value is not None]
        if not series:
            return None
        return float(sum(series) / len(series))

    final_gap_percent = None
    if objective_value is not None and best_bound is not None and float(objective_value) != 0.0:
        final_gap_percent = abs(float(objective_value) - float(best_bound)) / max(abs(float(objective_value)), 1e-9) * 100.0

    return {
        "strategy": strategy_payload["label"],
        "warm_start_mode": strategy_payload["warm_start_mode"],
        "seed": args.seed,
        "time_limit_seconds": args.time_limit_seconds,
        "num_areas": args.num_areas,
        "num_users": args.num_users,
        "num_agents": args.num_agents,
        "num_clusters": len(cluster_rows),
        "first_complete_feasible_time_seconds": complete_feasible_time,
        "mean_first_feasible_time_seconds": _mean(row.get("first_feasible_time_seconds") for row in cluster_rows),
        "mean_first_optimality_gap_percent": _mean(row.get("first_optimality_gap_percent") for row in cluster_rows),
        "final_objective_value": objective_value,
        "best_bound": best_bound,
        "final_relative_gap_percent": final_gap_percent,
        "mean_time_to_reference_objective_seconds": _mean(row.get("time_to_reference_objective_seconds") for row in cluster_rows),
        "mean_time_to_reference_gap_seconds": _mean(row.get("time_to_reference_gap_seconds") for row in cluster_rows),
        "total_explored_bnb_nodes": int(sum(int(row.get("explored_bnb_nodes") or 0) for row in cluster_rows)),
        "optimality_proven": bool(all(bool(row.get("optimality_proven")) for row in cluster_rows)) if cluster_rows else False,
        "warm_start_objective_quality_total": float(sum(float(row.get("warm_start_objective_quality") or 0.0) for row in cluster_rows)),
        "warm_start_objective_quality_mean": _mean(row.get("warm_start_objective_quality") for row in cluster_rows),
        "artifact_dir": strategy_payload["artifact_dir"],
    }


def main() -> None:
    args = parse_args()
    output_csv = Path(args.output_csv)
    shared_instance = generate_shared_instance(args)

    strategies = [
        ("MILP alone", "none"),
        ("MILP + ALNS", "alns"),
        ("MILP + GA", "ga"),
    ]

    strategy_payloads = [run_strategy(args, label, warm_start_mode, shared_instance) for label, warm_start_mode in strategies]
    cluster_rows = [row for payload in strategy_payloads for row in payload["cluster_rows"]]

    reference_by_cluster: Dict[int, Dict[str, float | None]] = defaultdict(dict)
    for cluster_id in sorted({int(row["cluster_id"]) for row in cluster_rows}):
        cluster_slice = [row for row in cluster_rows if int(row["cluster_id"]) == cluster_id]
        objective_values = [float(row["final_objective_value"]) for row in cluster_slice if row.get("final_objective_value") is not None]
        gap_values = [float(row["final_relative_gap_percent"]) for row in cluster_slice if row.get("final_relative_gap_percent") is not None]
        reference_by_cluster[cluster_id] = {
            "objective": min(objective_values) if objective_values else None,
            "gap_percent": min(gap_values) if gap_values else None,
        }

    for row in cluster_rows:
        cluster_id = int(row["cluster_id"])
        reference = reference_by_cluster[cluster_id]
        row["time_to_reference_objective_seconds"] = _time_to_target(
            row["progress_events"],
            "incumbent_value",
            reference.get("objective"),
        )
        row["time_to_reference_gap_seconds"] = _time_to_target(
            row["progress_events"],
            "relative_gap_percent",
            reference.get("gap_percent"),
        )

    summary_rows = [
        aggregate_summary_row(
            args=args,
            strategy_payload=payload,
            cluster_rows=[row for row in cluster_rows if row["strategy"] == payload["label"]],
        )
        for payload in strategy_payloads
    ]

    summary_fieldnames = [
        "strategy",
        "warm_start_mode",
        "seed",
        "time_limit_seconds",
        "num_areas",
        "num_users",
        "num_agents",
        "num_clusters",
        "first_complete_feasible_time_seconds",
        "mean_first_feasible_time_seconds",
        "mean_first_optimality_gap_percent",
        "final_objective_value",
        "best_bound",
        "final_relative_gap_percent",
        "mean_time_to_reference_objective_seconds",
        "mean_time_to_reference_gap_seconds",
        "total_explored_bnb_nodes",
        "optimality_proven",
        "warm_start_objective_quality_total",
        "warm_start_objective_quality_mean",
        "artifact_dir",
    ]
    detailed_fieldnames = [
        "strategy",
        "warm_start_mode",
        "cluster_id",
        "cluster_sequence_index",
        "first_feasible_time_seconds",
        "first_optimality_gap_percent",
        "final_objective_value",
        "best_bound",
        "final_relative_gap_percent",
        "elapsed_time_seconds",
        "time_to_reference_objective_seconds",
        "time_to_reference_gap_seconds",
        "explored_bnb_nodes",
        "active_bnb_nodes",
        "optimality_proven",
        "warm_start_objective_quality",
        "warm_start_timeframe_estimate",
        "solver_log_path",
    ]

    write_csv(output_csv, summary_rows, summary_fieldnames)
    detailed_path = output_csv.with_name(f"{output_csv.stem}_clusters.csv")
    write_csv(
        detailed_path,
        [{key: row.get(key) for key in detailed_fieldnames} for row in cluster_rows],
        detailed_fieldnames,
    )


if __name__ == "__main__":
    main()
