from __future__ import annotations

import argparse
import copy
import csv
import os
import threading
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import psutil
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
    EXPERIMENT_B_SOLVERS,
    EXPERIMENT_DEFAULT_ENV,
    EXPERIMENT_DEFAULT_MEMORY_LIMIT,
    EXPERIMENT_DEFAULT_OBJECTIVE,
    EXPERIMENT_DEFAULT_PRIORITY,
    EXPERIMENT_DEFAULT_SCENARIO,
    EXPERIMENT_DEFAULT_SEEDS,
    EXPERIMENT_RESULTS_DIR,
    HORIZONTAL_VELOCITY,
    VERTICAL_VELOCITY,
)
from dummy_app.tools.common import call_builder


EXPERIMENT_F_AREAS = [500, 1000, 1500, 2000]
EXPERIMENT_F_USERS_PER_AREA = [1, 3]
EXPERIMENT_F_UAVS = [8, 10, 12, 15]
EXPERIMENT_F_TIME_LIMIT_SECONDS = 2 * 60 * 60
EXPERIMENT_F_OPTIMAL_GAP_THRESHOLD_PERCENT = 1e-6
EXPERIMENT_E_RESULTS_PATH = Path(EXPERIMENT_RESULTS_DIR) / "experiment_E_sensitivity.csv"


class _PeakRSSMonitor:
    def __init__(self, interval_sec: float = 0.05) -> None:
        self.interval_sec = float(interval_sec)
        self.peak_rss_mb = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process(os.getpid())

    def start(self) -> None:
        self._stop_event.clear()
        self.peak_rss_mb = self._rss_mb()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 4.0))
        self.peak_rss_mb = max(self.peak_rss_mb, self._rss_mb())
        return float(self.peak_rss_mb)

    def _rss_mb(self) -> float:
        return float(self._process.memory_info().rss) / float(1024 ** 2)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            self.peak_rss_mb = max(self.peak_rss_mb, self._rss_mb())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment F: solver scalability boundaries.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(EXPERIMENT_RESULTS_DIR) / "experiment_F_scalability.csv"),
    )
    parser.add_argument(
        "--boundaries-output",
        type=str,
        default=str(Path(EXPERIMENT_RESULTS_DIR) / "experiment_F_scalability_boundaries.csv"),
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


def _scenario_rank(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("node_count", 0) or 0),
        int(row.get("users_per_area", 0) or 0),
        int(row.get("uav_count", 0) or 0),
    )


def _sum_solver_runtime_seconds(solve_status_history: List[Dict[str, Any]]) -> float:
    return float(
        sum(float(record.get("elapsed_time_seconds", 0.0) or 0.0) for record in solve_status_history)
    )


def _derive_binary_variable_count(run_result: Any) -> int | None:
    if run_result is None:
        return None

    total_binary_variables = 0
    for cluster_result in getattr(run_result, "cluster_results", []):
        cluster_metrics = dict(getattr(cluster_result, "cluster_metrics", {}))
        diagnostics = dict(getattr(cluster_result, "diagnostics", {}))
        original_node_count = int(cluster_metrics.get("node_count", 0) or 0)
        agent_count = int(cluster_metrics.get("agent_count", 0) or 0)
        virtual_nodes = dict(diagnostics.get("virtual_nodes", {}))
        removed_original_bridges = len({int(node_id) for node_id in virtual_nodes.values()})
        formulation_node_count = max(original_node_count - removed_original_bridges, 0)
        service_node_count = max(formulation_node_count - 1, 0)

        # Cooperative formulation: x(i,j,k) binaries over VxV and visit(j,k) over service nodes.
        total_binary_variables += formulation_node_count * formulation_node_count * agent_count
        total_binary_variables += service_node_count * agent_count

    return int(total_binary_variables)


def _is_full_coverage(row: Mapping[str, Any]) -> bool:
    coverage_ratio = row.get("coverage_ratio")
    return coverage_ratio is not None and float(coverage_ratio) >= 1.0 - 1e-9


def _is_optimal_boundary_candidate(row: Mapping[str, Any]) -> bool:
    gap_percent = row.get("optimality_gap_percent")
    return (
        str(row.get("status", "")) == "ok"
        and str(row.get("normalized_status", "")) == "optimal"
        and _is_full_coverage(row)
        and gap_percent is not None
        and float(gap_percent) <= float(EXPERIMENT_F_OPTIMAL_GAP_THRESHOLD_PERCENT)
    )


def _is_feasible_boundary_candidate(row: Mapping[str, Any]) -> bool:
    return bool(row.get("feasible_solution_found")) and _is_full_coverage(row)


def _is_failed_attempt_boundary_candidate(row: Mapping[str, Any]) -> bool:
    if not bool(row.get("attempted")):
        return False
    if bool(row.get("time_limit_no_solution")) or bool(row.get("model_build_error")) or bool(row.get("solver_error")):
        return True
    if str(row.get("status", "")) != "ok":
        return True
    return not _is_feasible_boundary_candidate(row)


def _mark_solver_boundaries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for row in rows:
        row["is_largest_solved_optimal_for_solver"] = False
        row["is_largest_feasible_for_solver"] = False
        row["is_largest_attempted_failed_for_solver"] = False

    boundary_summary_rows: List[Dict[str, Any]] = []

    for solver_name in EXPERIMENT_B_SOLVERS:
        solver_rows = [row for row in rows if str(row.get("solver_name", "")) == str(solver_name)]
        boundary_specs = [
            ("largest_solved_optimal", _is_optimal_boundary_candidate, "is_largest_solved_optimal_for_solver"),
            ("largest_feasible", _is_feasible_boundary_candidate, "is_largest_feasible_for_solver"),
            ("largest_attempted_failed", _is_failed_attempt_boundary_candidate, "is_largest_attempted_failed_for_solver"),
        ]
        for boundary_type, predicate, flag_name in boundary_specs:
            candidates = [row for row in solver_rows if predicate(row)]
            if not candidates:
                boundary_summary_rows.append(
                    {
                        "solver_name": solver_name,
                        "boundary_type": boundary_type,
                        "scenario_id": "",
                        "node_count": None,
                        "users_per_area": None,
                        "uav_count": None,
                        "status": "",
                        "normalized_status": "",
                        "termination_reason": "",
                        "coverage_ratio": None,
                        "optimality_gap_percent": None,
                    }
                )
                continue

            selected = max(candidates, key=_scenario_rank)
            selected[flag_name] = True
            boundary_summary_rows.append(
                {
                    "solver_name": solver_name,
                    "boundary_type": boundary_type,
                    "scenario_id": selected.get("scenario_id", ""),
                    "node_count": selected.get("node_count"),
                    "users_per_area": selected.get("users_per_area"),
                    "uav_count": selected.get("uav_count"),
                    "status": selected.get("status", ""),
                    "normalized_status": selected.get("normalized_status", ""),
                    "termination_reason": selected.get("termination_reason", ""),
                    "coverage_ratio": selected.get("coverage_ratio"),
                    "optimality_gap_percent": selected.get("optimality_gap_percent"),
                }
            )

    return boundary_summary_rows


def _run_scalability_case(
    scenario_payload: Mapping[str, Any],
    *,
    solver_name: str,
    fairness_tolerance: int,
    memory_limit_bytes: int | None,
) -> Dict[str, Any]:
    builder = None
    peak_monitor = _PeakRSSMonitor()
    total_started_at = time.perf_counter()
    run_model_elapsed_sec = 0.0

    try:
        _enforce_memory_limit(memory_limit_bytes)
        set_random_seed(int(scenario_payload["seed"]))
        config = _build_runtime_config(
            scenario_payload,
            model_name="milp",
            solver_backend=solver_name,
            warm_start_mode="none",
            stage_solution=1,
            objective_weights=None,
            fairness_tolerance=int(fairness_tolerance),
            time_step_sec=600,
            time_limit_seconds=enforce_time_limit(EXPERIMENT_F_TIME_LIMIT_SECONDS),
            priority=EXPERIMENT_DEFAULT_PRIORITY,
        )

        with _suppress_nested_output():
            peak_monitor.start()
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
            run_model_started_at = time.perf_counter()
            builder.run_model(
                distance_matrix=np.array(scenario_payload["distance_matrix"], copy=True),
                data=data,
                cue_groups=copy.deepcopy(scenario_payload["user_points"]),
            )
            run_model_elapsed_sec = time.perf_counter() - run_model_started_at
            builder.gather_results()

        total_runtime_sec = time.perf_counter() - total_started_at
        memory_peak_mb = peak_monitor.stop()

        if builder.latest_model_run_result is None:
            raise RuntimeError("No ModelRunResult was produced.")

        run_result = builder.latest_model_run_result
        summary = dict(getattr(run_result, "summary", {}))
        solve_status_history = list(dict(getattr(run_result, "diagnostics", {})).get("solve_status_history", []))
        metrics = extract_common_run_metrics(run_result, scenario_payload)
        solver_runtime_sec = _sum_solver_runtime_seconds(solve_status_history)

        return {
            "status": "ok",
            "solver_name": solver_name,
            "run_result": run_result,
            "artifact_dir": getattr(builder, "latest_artifact_dir", ""),
            "error_message": "",
            "summary": summary,
            "metrics": metrics,
            "solve_status_history": solve_status_history,
            "model_build_time_sec": max(float(run_model_elapsed_sec) - float(solver_runtime_sec), 0.0),
            "solver_runtime_sec": float(solver_runtime_sec),
            "total_runtime_sec": float(total_runtime_sec),
            "memory_peak_mb": float(memory_peak_mb),
            "binary_variables": _derive_binary_variable_count(run_result),
        }
    except Exception as exc:
        total_runtime_sec = time.perf_counter() - total_started_at
        memory_peak_mb = peak_monitor.stop()
        error_message = f"{exc.__class__.__name__}: {exc}"
        summary = {}
        solve_status_history: List[Dict[str, Any]] = []
        if builder is not None:
            try:
                summary = dict(builder.build_failed_run_summary(exc))
            except Exception:
                summary = dict(getattr(builder, "latest_run_summary", {}) or {})
            solve_status_history = list(getattr(builder, "solve_status_history", []))
        metrics = _extract_failure_metrics(builder, error_message)
        solver_runtime_sec = _sum_solver_runtime_seconds(solve_status_history)
        latest_model_run_result = getattr(builder, "latest_model_run_result", None) if builder is not None else None

        return {
            "status": "failed",
            "solver_name": solver_name,
            "run_result": latest_model_run_result,
            "artifact_dir": getattr(builder, "latest_artifact_dir", "") if builder is not None else "",
            "error_message": error_message,
            "summary": summary,
            "metrics": metrics,
            "solve_status_history": solve_status_history,
            "model_build_time_sec": max(float(run_model_elapsed_sec or total_runtime_sec) - float(solver_runtime_sec), 0.0),
            "solver_runtime_sec": float(solver_runtime_sec),
            "total_runtime_sec": float(total_runtime_sec),
            "memory_peak_mb": float(memory_peak_mb),
            "binary_variables": _derive_binary_variable_count(latest_model_run_result),
        }


def _build_result_row(
    scenario_payload: Mapping[str, Any],
    result: Dict[str, Any],
    *,
    fairness_tolerance: int,
    memory_limit_bytes: int | None,
) -> Dict[str, Any]:
    run_result = result.get("run_result")
    metrics = dict(result.get("metrics", {}))
    summary = dict(result.get("summary", {}))
    solve_status_history = list(result.get("solve_status_history", []))

    normalized_status = getattr(run_result, "normalized_status", None) if run_result is not None else None
    raw_status = getattr(run_result, "raw_status", None) if run_result is not None else None
    termination_reason = getattr(run_result, "termination_reason", None) if run_result is not None else None
    if termination_reason is None and solve_status_history:
        termination_reason = solve_status_history[-1].get("termination_reason")
    if termination_reason is None and result.get("error_message"):
        termination_reason = str(result["error_message"])

    row = format_result_row(
        scenario_payload,
        "MILP",
        status=result["status"],
        extra_fields={
            "solver_name": result.get("solver_name", ""),
            "time_limit_seconds": EXPERIMENT_F_TIME_LIMIT_SECONDS,
            "memory_limit_bytes": int(memory_limit_bytes) if memory_limit_bytes is not None else None,
            "fairness_tolerance_baseline": int(fairness_tolerance),
            "raw_status": raw_status or result["status"],
            "normalized_status": normalized_status or result["status"],
            "termination_reason": termination_reason or "",
            "attempted": True,
            "num_clusters": summary.get("num_clusters"),
            "num_variables": summary.get("num_variables"),
            "num_binary_variables": result.get("binary_variables"),
            "num_constraints": summary.get("num_constraints"),
            "largest_cluster_size": summary.get("largest_cluster_size"),
            "model_build_time_sec": result.get("model_build_time_sec"),
            "solver_runtime_sec": result.get("solver_runtime_sec"),
            "total_runtime_sec": result.get("total_runtime_sec"),
            "runtime_sec": result.get("total_runtime_sec"),
            "memory_peak_mb": result.get("memory_peak_mb"),
            "memory_usage_mb": metrics.get("memory_usage_mb", summary.get("memory_usage_mb")),
            "objective_value": metrics.get("objective_value"),
            "optimality_gap_percent": metrics.get("optimality_gap_percent"),
            "coverage_ratio": metrics.get("coverage_ratio", summary.get("coverage_ratio")),
            "total_distance": metrics.get("total_distance"),
            "total_energy": metrics.get("total_energy"),
            "total_travel_time": metrics.get("total_travel_time"),
            "active_uav_count": summary.get("num_uavs_used"),
            "feasible_solution_found": metrics.get("feasible_solution_found", False),
            "time_limit_feasible": metrics.get("time_limit_feasible", False),
            "time_limit_no_solution": metrics.get("time_limit_no_solution", False),
            "model_build_error": metrics.get("model_build_error", False),
            "solver_error": metrics.get("solver_error", False),
            "artifact_dir": result.get("artifact_dir", ""),
            "error_message": result.get("error_message", ""),
        },
    )
    return row


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    boundaries_output_path = Path(args.boundaries_output)
    if output_path.exists() and not args.append:
        output_path.unlink()
    if boundaries_output_path.exists() and not args.append:
        boundaries_output_path.unlink()

    fairness_tolerance = _load_fairness_baseline()
    total_runs = (
        len(EXPERIMENT_DEFAULT_SEEDS)
        * len(EXPERIMENT_F_AREAS)
        * len(EXPERIMENT_F_USERS_PER_AREA)
        * len(EXPERIMENT_F_UAVS)
        * len(EXPERIMENT_B_SOLVERS)
    )

    rows: List[Dict[str, Any]] = []

    with tqdm(total=total_runs, desc="Experiment F", unit="run", dynamic_ncols=True) as progress:
        for seed in EXPERIMENT_DEFAULT_SEEDS:
            for node_count, users_per_area, uav_count in product(
                EXPERIMENT_F_AREAS,
                EXPERIMENT_F_USERS_PER_AREA,
                EXPERIMENT_F_UAVS,
            ):
                scenario_payload = create_scenario(
                    node_count=node_count,
                    users_per_area=users_per_area,
                    uav_count=uav_count,
                    coverage_time_profile="medium",
                    seed=seed,
                    scenario_name=EXPERIMENT_DEFAULT_SCENARIO,
                    objective_function=EXPERIMENT_DEFAULT_OBJECTIVE,
                    env_type=EXPERIMENT_DEFAULT_ENV,
                )
                for solver_name in EXPERIMENT_B_SOLVERS:
                    progress.set_postfix_str(
                        f"seed={seed} n={node_count} u={users_per_area} k={uav_count} | {solver_name}",
                        refresh=False,
                    )
                    result = _run_scalability_case(
                        scenario_payload,
                        solver_name=solver_name,
                        fairness_tolerance=fairness_tolerance,
                        memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                    )
                    rows.append(
                        _build_result_row(
                            scenario_payload,
                            result,
                            fairness_tolerance=fairness_tolerance,
                            memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                        )
                    )
                    progress.update(1)

    boundary_rows = _mark_solver_boundaries(rows)
    save_results(output_path, rows)
    save_results(boundaries_output_path, boundary_rows)


if __name__ == "__main__":
    main()
