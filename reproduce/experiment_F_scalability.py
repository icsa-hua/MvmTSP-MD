from __future__ import annotations

import argparse
import copy
import csv
import gc
import math
import os
import multiprocessing as mp
import queue
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
from dummy_app.pipeline.instance_builder import build_problem_instance
from dummy_app.program_config import (
    ALTITUDE,
    EXPERIMENT_B_SOLVERS,
    EXPERIMENT_DEFAULT_ENV,
    EXPERIMENT_DEFAULT_MEMORY_LIMIT,
    EXPERIMENT_DEFAULT_OBJECTIVE,
    EXPERIMENT_DEFAULT_PRIORITY,
    EXPERIMENT_DEFAULT_RUN_TIME_LIMIT_SECONDS,
    EXPERIMENT_DEFAULT_SCENARIO,
    EXPERIMENT_DEFAULT_SEEDS,
    EXPERIMENT_RESULTS_DIR,
    HORIZONTAL_VELOCITY,
    VERTICAL_VELOCITY,
    SUBTOUR_MODE,
)
from dummy_app.tools.common import call_builder


EXPERIMENT_F_AREAS = [500, 1000, 1500, 2000]
EXPERIMENT_F_USERS_PER_AREA = [1, 3]
EXPERIMENT_F_UAVS = [8, 10, 12, 15]
EXPERIMENT_F_TIME_LIMIT_SECONDS = 2 * 60 * 60
EXPERIMENT_F_OPTIMAL_GAP_THRESHOLD_PERCENT = 1e-6
EXPERIMENT_F_CHECKPOINT_PERCENT = 10.0
EXPERIMENT_F_HARD_MEMORY_CAP_GB = 100.0
EXPERIMENT_F_MONITOR_INTERVAL_SECONDS = 0.5
EXPERIMENT_F_WORKER_SHUTDOWN_GRACE_SECONDS = 10.0
EXPERIMENT_E_RESULTS_PATH = Path(EXPERIMENT_RESULTS_DIR) / "experiment_E_sensitivity.csv"

_INT_FIELDS = {
    "node_count",
    "users_per_area",
    "uav_count",
    "seed",
    "time_limit_seconds",
    "wall_clock_limit_seconds",
    "memory_limit_bytes",
    "process_memory_cap_bytes",
    "fairness_tolerance_baseline",
    "num_clusters",
    "num_variables",
    "num_binary_variables",
    "num_constraints",
    "largest_cluster_size",
    "active_uav_count",
    "unvisited_nodes",
    "failed_cluster_id",
    "failed_cluster_order_index",
    "failed_cluster_node_count",
    "failed_cluster_virtual_node_count",
    "failed_cluster_uav_count",
    "failed_cluster_time_horizon",
    "failed_cluster_arc_count",
}
_FLOAT_FIELDS = {
    "model_build_time_sec",
    "solver_runtime_sec",
    "total_runtime_sec",
    "runtime_sec",
    "memory_peak_mb",
    "memory_usage_mb",
    "objective_value",
    "optimality_gap_percent",
    "coverage_ratio",
    "total_distance",
    "total_energy",
    "total_travel_time",
}
_BOOL_FIELDS = {
    "attempted",
    "feasible_solution_found",
    "time_limit_feasible",
    "time_limit_no_solution",
    "model_build_error",
    "solver_error",
    "is_largest_solved_optimal_for_solver",
    "is_largest_feasible_for_solver",
    "is_largest_attempted_failed_for_solver",
}


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
    parser.add_argument(
        "--checkpoint-percent",
        type=float,
        default=EXPERIMENT_F_CHECKPOINT_PERCENT,
        help="Persist completed results every N percent of total progress.",
    )
    parser.add_argument(
        "--process-memory-cap-gb",
        type=float,
        default=EXPERIMENT_F_HARD_MEMORY_CAP_GB,
        help="Hard per-run process RSS cap in GB for the spawned worker; 0 disables the monitor.",
    )
    parser.add_argument(
        "--wall-clock-limit-seconds",
        type=int,
        default=EXPERIMENT_F_TIME_LIMIT_SECONDS,
        help="Hard per-run wall-clock limit enforced by the parent monitor; 0 disables the monitor.",
    )
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


def _scenario_stub(
    *,
    seed: int,
    node_count: int,
    users_per_area: int,
    uav_count: int,
    coverage_time_profile: str = "medium",
) -> Dict[str, Any]:
    return {
        "scenario_id": (
            f"seed{int(seed)}_n{int(node_count)}_u{int(users_per_area)}"
            f"_k{int(uav_count)}_{coverage_time_profile}"
        ),
        "node_count": int(node_count),
        "users_per_area": int(users_per_area),
        "uav_count": int(uav_count),
        "coverage_time_profile": str(coverage_time_profile),
        "seed": int(seed),
    }


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _normalize_loaded_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    for field in _INT_FIELDS:
        value = normalized.get(field)
        if value in {None, ""}:
            normalized[field] = None
            continue
        try:
            normalized[field] = int(float(value))
        except (TypeError, ValueError):
            normalized[field] = None
    for field in _FLOAT_FIELDS:
        value = normalized.get(field)
        if value in {None, ""}:
            normalized[field] = None
            continue
        try:
            normalized[field] = float(value)
        except (TypeError, ValueError):
            normalized[field] = None
    for field in _BOOL_FIELDS:
        normalized[field] = _parse_bool(normalized.get(field))
    return normalized


def _load_existing_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [_normalize_loaded_row(row) for row in csv.DictReader(handle)]


def _case_key(
    *,
    seed: int,
    node_count: int,
    users_per_area: int,
    uav_count: int,
    solver_name: str,
) -> tuple[int, int, int, int, str]:
    return (
        int(seed),
        int(node_count),
        int(users_per_area),
        int(uav_count),
        str(solver_name),
    )


def _case_key_from_row(row: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
    return _case_key(
        seed=int(row.get("seed", 0) or 0),
        node_count=int(row.get("node_count", 0) or 0),
        users_per_area=int(row.get("users_per_area", 0) or 0),
        uav_count=int(row.get("uav_count", 0) or 0),
        solver_name=str(row.get("solver_name", "") or ""),
    )


def _memory_cap_bytes_from_gb(memory_cap_gb: float | None) -> int | None:
    if memory_cap_gb is None or float(memory_cap_gb) <= 0.0:
        return None
    return int(float(memory_cap_gb) * float(1024 ** 3))


def _checkpoint_interval(total_runs: int, checkpoint_percent: float) -> int:
    if checkpoint_percent <= 0:
        return max(1, total_runs)
    return max(1, int(math.ceil(float(total_runs) * float(checkpoint_percent) / 100.0)))


def _rewrite_results(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    if csv_path.exists():
        csv_path.unlink()
    save_results(csv_path, rows)


def _persist_checkpoint(
    output_path: Path,
    boundaries_output_path: Path,
    *,
    all_rows: List[Dict[str, Any]],
    pending_rows: List[Dict[str, Any]],
) -> None:
    if pending_rows:
        pending_rows.clear()
    boundary_rows = _mark_solver_boundaries(all_rows)
    _rewrite_results(output_path, all_rows)
    _rewrite_results(boundaries_output_path, boundary_rows)


def _release_case_memory(builder: Any, scenario_payload: Dict[str, Any] | None) -> None:
    if builder is not None:
        for attr_name, reset_value in (
            ("current_problem_instance", None),
            ("run_request", None),
            ("latest_model_run_result", None),
            ("latest_playback_rows", []),
            ("latest_playback_metadata", {}),
            ("latest_run_report", {}),
            ("latest_run_summary", {}),
            ("coordinated_plan", {}),
            ("plan_with_nodes", {}),
            ("problem_results", {}),
            ("solve_status_history", []),
            ("cluster_status_records", []),
            ("user_points", {}),
            ("move_energy", None),
            ("V", None),
        ):
            if hasattr(builder, attr_name):
                try:
                    setattr(builder, attr_name, copy.deepcopy(reset_value))
                except Exception:
                    continue
        metrics = getattr(builder, "metrics", None)
        if metrics is not None:
            try:
                metrics.cluster_records.clear()
                metrics.coverage_records.clear()
                metrics.latest_run_report = {}
            except Exception:
                pass

    if scenario_payload is not None:
        for key in (
            "regions",
            "centroids",
            "user_points",
            "depots",
            "distance_matrix",
            "all_users",
            "all_nodes",
            "target_nodes",
        ):
            if key in scenario_payload:
                scenario_payload[key] = None

    gc.collect()


def _preflight_audit_filename(
    scenario_payload: Mapping[str, Any],
    *,
    solver_name: str,
) -> str:
    return (
        f"preflight_{scenario_payload['scenario_id']}_{solver_name}_"
        f"n{int(scenario_payload['node_count'])}_u{int(scenario_payload['users_per_area'])}_k{int(scenario_payload['uav_count'])}.csv"
    )


def _run_scalability_preflight(
    scenario_payload: Mapping[str, Any],
    *,
    solver_name: str,
    fairness_tolerance: int,
    memory_limit_bytes: int | None,
) -> Dict[str, Any]:
    builder = None
    data = None
    audit_records: List[Dict[str, Any]] = []
    audit_path = ""
    started_at = time.perf_counter()

    try:
        _enforce_memory_limit(memory_limit_bytes)
        set_random_seed(int(scenario_payload["seed"]))
        config = _build_runtime_config(
            scenario_payload,
            model_name="milp",
            solver_backend=solver_name,
            subtour_mode=SUBTOUR_MODE,
            warm_start_mode="none",
            stage_solution=1,
            objective_weights=None,
            fairness_tolerance=int(fairness_tolerance),
            time_step_sec=600,
            time_limit_seconds=enforce_time_limit(EXPERIMENT_F_TIME_LIMIT_SECONDS),
            run_time_limit_seconds=enforce_time_limit(EXPERIMENT_DEFAULT_RUN_TIME_LIMIT_SECONDS),
            priority=EXPERIMENT_DEFAULT_PRIORITY,
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
            builder._prepare_run_state()
            builder.run_request = builder.build_run_request()
            builder.current_problem_instance = build_problem_instance(
                builder,
                np.array(scenario_payload["distance_matrix"], copy=True),
                data,
                copy.deepcopy(scenario_payload["user_points"]),
            )
            builder.total_number_cluster = len(builder.current_problem_instance.prepared_clusters)
            for cluster_solve_order, prepared_cluster in enumerate(builder.current_problem_instance.prepared_clusters):
                prepared_cluster.metadata["cluster_solve_order"] = int(cluster_solve_order)
                audit_records.append(
                    builder.build_cluster_model_only(
                        builder.current_problem_instance,
                        prepared_cluster,
                        builder.run_request,
                    )
                )

        audit_dir = getattr(builder, "exported_problems_dir", Path(os.getcwd()) / "exported_problems")
        audit_path_obj = Path(audit_dir) / _preflight_audit_filename(scenario_payload, solver_name=solver_name)
        if audit_path_obj.exists():
            audit_path_obj.unlink()
        save_results(audit_path_obj, audit_records)
        audit_path = str(audit_path_obj)

        return {
            "status": "ok",
            "audit_records": audit_records,
            "audit_path": audit_path,
            "failed_cluster_context": dict(getattr(builder, "latest_failed_cluster_context", {})),
            "preflight_runtime_sec": float(time.perf_counter() - started_at),
            "error_message": "",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "audit_records": audit_records,
            "audit_path": audit_path,
            "failed_cluster_context": dict(getattr(builder, "latest_failed_cluster_context", {})) if builder is not None else {},
            "preflight_runtime_sec": float(time.perf_counter() - started_at),
            "error_message": f"{exc.__class__.__name__}: {exc}",
        }
    finally:
        data = None
        _release_case_memory(builder, None)


def _run_scalability_case(
    scenario_payload: Mapping[str, Any],
    *,
    solver_name: str,
    fairness_tolerance: int,
    memory_limit_bytes: int | None,
) -> Dict[str, Any]:
    builder = None
    data = None
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
            subtour_mode=SUBTOUR_MODE,
            warm_start_mode="none",
            stage_solution=1,
            objective_weights=None,
            fairness_tolerance=int(fairness_tolerance),
            time_step_sec=600,
            time_limit_seconds=enforce_time_limit(EXPERIMENT_F_TIME_LIMIT_SECONDS),
            run_time_limit_seconds=enforce_time_limit(EXPERIMENT_DEFAULT_RUN_TIME_LIMIT_SECONDS),
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
    finally:
        data = None
        _release_case_memory(builder, scenario_payload if isinstance(scenario_payload, dict) else None)


def _build_monitored_failure_row(
    *,
    seed: int,
    node_count: int,
    users_per_area: int,
    uav_count: int,
    solver_name: str,
    fairness_tolerance: int,
    memory_limit_bytes: int | None,
    process_memory_cap_bytes: int | None,
    wall_clock_limit_seconds: int | None,
    termination_reason: str,
    error_message: str,
    total_runtime_sec: float,
    memory_peak_mb: float,
) -> Dict[str, Any]:
    scenario_payload = _scenario_stub(
        seed=seed,
        node_count=node_count,
        users_per_area=users_per_area,
        uav_count=uav_count,
    )
    return format_result_row(
        scenario_payload,
        "MILP",
        status="failed",
        extra_fields={
            "solver_name": solver_name,
            "time_limit_seconds": EXPERIMENT_F_TIME_LIMIT_SECONDS,
            "wall_clock_limit_seconds": int(wall_clock_limit_seconds) if wall_clock_limit_seconds is not None else None,
            "memory_limit_bytes": int(memory_limit_bytes) if memory_limit_bytes is not None else None,
            "process_memory_cap_bytes": int(process_memory_cap_bytes) if process_memory_cap_bytes is not None else None,
            "fairness_tolerance_baseline": int(fairness_tolerance),
            "raw_status": "terminated_by_monitor",
            "normalized_status": "failed",
            "termination_reason": termination_reason,
            "attempted": True,
            "num_clusters": None,
            "num_variables": None,
            "num_binary_variables": None,
            "num_constraints": None,
            "largest_cluster_size": None,
            "cluster_solve_order": [],
            "failed_cluster_id": None,
            "failed_cluster_order_index": None,
            "failed_cluster_node_count": None,
            "failed_cluster_virtual_node_count": None,
            "failed_cluster_uav_count": None,
            "failed_cluster_time_horizon": None,
            "failed_cluster_arc_count": None,
            "model_build_time_sec": None,
            "solver_runtime_sec": None,
            "total_runtime_sec": float(total_runtime_sec),
            "runtime_sec": float(total_runtime_sec),
            "memory_peak_mb": float(memory_peak_mb),
            "memory_usage_mb": None,
            "objective_value": None,
            "optimality_gap_percent": None,
            "coverage_ratio": None,
            "total_distance": None,
            "total_energy": None,
            "total_travel_time": None,
            "active_uav_count": None,
            "feasible_solution_found": False,
            "time_limit_feasible": False,
            "time_limit_no_solution": termination_reason == "wall_clock_time_limit_exceeded",
            "model_build_error": False,
            "solver_error": False,
            "preflight_audit_path": "",
            "artifact_dir": "",
            "error_message": error_message,
        },
    )


def _scalability_case_worker(
    result_queue: mp.Queue,
    *,
    seed: int,
    node_count: int,
    users_per_area: int,
    uav_count: int,
    solver_name: str,
    fairness_tolerance: int,
    memory_limit_bytes: int | None,
    process_memory_cap_bytes: int | None,
    wall_clock_limit_seconds: int | None,
) -> None:
    scenario_payload: Dict[str, Any] | None = None
    result_row: Dict[str, Any] | None = None

    try:
        if process_memory_cap_bytes is not None:
            _enforce_memory_limit(process_memory_cap_bytes)
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
        preflight = _run_scalability_preflight(
            scenario_payload,
            solver_name=solver_name,
            fairness_tolerance=fairness_tolerance,
            memory_limit_bytes=memory_limit_bytes,
        )
        if preflight["status"] != "ok":
            result_row = _build_monitored_failure_row(
                seed=seed,
                node_count=node_count,
                users_per_area=users_per_area,
                uav_count=uav_count,
                solver_name=solver_name,
                fairness_tolerance=fairness_tolerance,
                memory_limit_bytes=memory_limit_bytes,
                process_memory_cap_bytes=process_memory_cap_bytes,
                wall_clock_limit_seconds=wall_clock_limit_seconds,
                termination_reason="preflight_model_build_failed",
                error_message=str(preflight.get("error_message", "") or "Preflight model build failed."),
                total_runtime_sec=float(preflight.get("preflight_runtime_sec", 0.0) or 0.0),
                memory_peak_mb=0.0,
            )
            failed_cluster_context = dict(preflight.get("failed_cluster_context", {}))
            if failed_cluster_context:
                result_row.update(
                    {
                        "cluster_solve_order": [record.get("cluster_id") for record in preflight.get("audit_records", [])],
                        "failed_cluster_id": failed_cluster_context.get("cluster_id"),
                        "failed_cluster_order_index": failed_cluster_context.get("cluster_solve_order"),
                        "failed_cluster_node_count": failed_cluster_context.get("original_node_count"),
                        "failed_cluster_virtual_node_count": failed_cluster_context.get("virtual_node_count"),
                        "failed_cluster_uav_count": failed_cluster_context.get("assigned_uav_count"),
                        "failed_cluster_time_horizon": failed_cluster_context.get("time_horizon"),
                        "failed_cluster_arc_count": failed_cluster_context.get("arc_count"),
                    }
                )
            result_row["preflight_audit_path"] = str(preflight.get("audit_path", "") or "")
            return
        result = _run_scalability_case(
            scenario_payload,
            solver_name=solver_name,
            fairness_tolerance=fairness_tolerance,
            memory_limit_bytes=memory_limit_bytes,
        )
        result["preflight_audit_path"] = str(preflight.get("audit_path", "") or "")
        result_row = _build_result_row(
            scenario_payload,
            result,
            fairness_tolerance=fairness_tolerance,
            memory_limit_bytes=memory_limit_bytes,
            process_memory_cap_bytes=process_memory_cap_bytes,
            wall_clock_limit_seconds=wall_clock_limit_seconds,
        )
    except Exception as exc:
        result_row = _build_monitored_failure_row(
            seed=seed,
            node_count=node_count,
            users_per_area=users_per_area,
            uav_count=uav_count,
            solver_name=solver_name,
            fairness_tolerance=fairness_tolerance,
            memory_limit_bytes=memory_limit_bytes,
            process_memory_cap_bytes=process_memory_cap_bytes,
            wall_clock_limit_seconds=wall_clock_limit_seconds,
            termination_reason="worker_exception",
            error_message=f"{exc.__class__.__name__}: {exc}",
            total_runtime_sec=0.0,
            memory_peak_mb=0.0,
        )
    finally:
        _release_case_memory(None, scenario_payload)
        if result_row is not None:
            try:
                result_queue.put(result_row, block=False)
            except Exception:
                pass


def _build_result_row(
    scenario_payload: Mapping[str, Any],
    result: Dict[str, Any],
    *,
    fairness_tolerance: int,
    memory_limit_bytes: int | None,
    process_memory_cap_bytes: int | None,
    wall_clock_limit_seconds: int | None,
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
            "wall_clock_limit_seconds": int(wall_clock_limit_seconds) if wall_clock_limit_seconds is not None else None,
            "memory_limit_bytes": int(memory_limit_bytes) if memory_limit_bytes is not None else None,
            "process_memory_cap_bytes": int(process_memory_cap_bytes) if process_memory_cap_bytes is not None else None,
            "fairness_tolerance_baseline": int(fairness_tolerance),
            "raw_status": raw_status or result["status"],
            "normalized_status": normalized_status or result["status"],
            "termination_reason": termination_reason or "",
            "attempted": True,
            "num_clusters": summary.get("num_clusters"),
            "num_variables": summary.get("num_variables"),
            "num_binary_variables": summary.get("num_binary_variables"),
            "num_constraints": summary.get("num_constraints"),
            "largest_cluster_size": summary.get("largest_cluster_size"),
            "cluster_solve_order": summary.get("cluster_solve_order", []),
            "failed_cluster_id": summary.get("failed_cluster_id"),
            "failed_cluster_order_index": summary.get("failed_cluster_order_index"),
            "failed_cluster_node_count": summary.get("failed_cluster_node_count"),
            "failed_cluster_virtual_node_count": summary.get("failed_cluster_virtual_node_count"),
            "failed_cluster_uav_count": summary.get("failed_cluster_uav_count"),
            "failed_cluster_time_horizon": summary.get("failed_cluster_time_horizon"),
            "failed_cluster_arc_count": summary.get("failed_cluster_arc_count"),
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
            "preflight_audit_path": result.get("preflight_audit_path", ""),
            "artifact_dir": result.get("artifact_dir", ""),
            "error_message": result.get("error_message", ""),
            "total_data_rate_mbps": metrics.get("total_data_rate_mbps"),
            "data_rate_per_hour_mbps": metrics.get("data_rate_per_hour_mbps"),
            "data_rate_per_kwh_mbps": metrics.get("data_rate_per_kwh_mbps"),
            "avg_data_rate_per_cluster_mbps": metrics.get("avg_data_rate_per_cluster_mbps"),
            "mean_sinr_db": metrics.get("mean_sinr_db"),
            "coverage_prob_at_0db": metrics.get("coverage_prob_at_0db"),
            "coverage_prob_at_10db": metrics.get("coverage_prob_at_10db"),
            "coverage_prob_at_20db": metrics.get("coverage_prob_at_20db"),
        },
    )
    return row


def _execute_scalability_case(
    *,
    seed: int,
    node_count: int,
    users_per_area: int,
    uav_count: int,
    solver_name: str,
    fairness_tolerance: int,
    memory_limit_bytes: int | None,
    process_memory_cap_bytes: int | None,
    wall_clock_limit_seconds: int | None,
) -> Dict[str, Any]:
    context = mp.get_context("spawn")
    result_queue: mp.Queue = context.Queue(maxsize=1)
    worker = context.Process(
        target=_scalability_case_worker,
        kwargs={
            "result_queue": result_queue,
            "seed": seed,
            "node_count": node_count,
            "users_per_area": users_per_area,
            "uav_count": uav_count,
            "solver_name": solver_name,
            "fairness_tolerance": fairness_tolerance,
            "memory_limit_bytes": memory_limit_bytes,
            "process_memory_cap_bytes": process_memory_cap_bytes,
            "wall_clock_limit_seconds": wall_clock_limit_seconds,
        },
    )

    worker.start()
    started_at = time.perf_counter()
    peak_rss_bytes = 0
    termination_reason: str | None = None
    monitor_error_message = ""
    monitored_process = None
    if worker.pid is not None:
        try:
            monitored_process = psutil.Process(worker.pid)
        except psutil.Error:
            monitored_process = None

    while worker.is_alive():
        elapsed = time.perf_counter() - started_at
        try:
            rss_bytes = int(monitored_process.memory_info().rss) if monitored_process is not None else 0
            peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
        except psutil.Error:
            rss_bytes = 0

        if process_memory_cap_bytes is not None and rss_bytes > process_memory_cap_bytes:
            termination_reason = "process_memory_cap_exceeded"
            monitor_error_message = (
                f"Parent monitor terminated worker after RSS reached "
                f"{rss_bytes / float(1024 ** 3):.2f} GB, above the configured cap "
                f"of {process_memory_cap_bytes / float(1024 ** 3):.2f} GB."
            )
            worker.terminate()
            break

        if wall_clock_limit_seconds is not None and elapsed > float(wall_clock_limit_seconds):
            termination_reason = "wall_clock_time_limit_exceeded"
            monitor_error_message = (
                f"Parent monitor terminated worker after {elapsed:.2f} seconds, above the configured wall-clock "
                f"limit of {float(wall_clock_limit_seconds):.2f} seconds."
            )
            worker.terminate()
            break

        time.sleep(EXPERIMENT_F_MONITOR_INTERVAL_SECONDS)

    worker.join(timeout=EXPERIMENT_F_WORKER_SHUTDOWN_GRACE_SECONDS)
    if worker.is_alive():
        worker.kill()
        worker.join(timeout=EXPERIMENT_F_WORKER_SHUTDOWN_GRACE_SECONDS)

    total_runtime_sec = time.perf_counter() - started_at
    try:
        if monitored_process is not None:
            peak_rss_bytes = max(peak_rss_bytes, int(monitored_process.memory_info().rss))
    except psutil.Error:
        pass
    peak_rss_mb = float(peak_rss_bytes) / float(1024 ** 2)

    try:
        row = result_queue.get_nowait()
    except queue.Empty:
        row = None

    if row is not None:
        row["memory_peak_mb"] = max(float(row.get("memory_peak_mb") or 0.0), peak_rss_mb)
        if row.get("total_runtime_sec") in {None, ""}:
            row["total_runtime_sec"] = float(total_runtime_sec)
        if row.get("runtime_sec") in {None, ""}:
            row["runtime_sec"] = float(total_runtime_sec)
        return row

    if termination_reason is None:
        termination_reason = "worker_exited_without_result"
        monitor_error_message = f"Worker exited with code {worker.exitcode} without returning a result row."

    return _build_monitored_failure_row(
        seed=seed,
        node_count=node_count,
        users_per_area=users_per_area,
        uav_count=uav_count,
        solver_name=solver_name,
        fairness_tolerance=fairness_tolerance,
        memory_limit_bytes=memory_limit_bytes,
        process_memory_cap_bytes=process_memory_cap_bytes,
        wall_clock_limit_seconds=wall_clock_limit_seconds,
        termination_reason=termination_reason,
        error_message=monitor_error_message,
        total_runtime_sec=total_runtime_sec,
        memory_peak_mb=peak_rss_mb,
    )


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    boundaries_output_path = Path(args.boundaries_output)
    process_memory_cap_bytes = _memory_cap_bytes_from_gb(args.process_memory_cap_gb)
    wall_clock_limit_seconds = enforce_time_limit(args.wall_clock_limit_seconds)
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
    checkpoint_interval = _checkpoint_interval(total_runs, args.checkpoint_percent)

    rows: List[Dict[str, Any]] = _load_existing_rows(output_path) if args.append else []
    pending_rows: List[Dict[str, Any]] = []
    completed_case_keys = {_case_key_from_row(row) for row in rows}

    with tqdm(
        total=total_runs,
        initial=min(len(completed_case_keys), total_runs),
        desc="Experiment F",
        unit="run",
        dynamic_ncols=True,
    ) as progress:
        for seed in EXPERIMENT_DEFAULT_SEEDS:
            for node_count, users_per_area, uav_count in product(
                EXPERIMENT_F_AREAS,
                EXPERIMENT_F_USERS_PER_AREA,
                EXPERIMENT_F_UAVS,
            ):
                for solver_name in EXPERIMENT_B_SOLVERS:
                    current_case_key = _case_key(
                        seed=seed,
                        node_count=node_count,
                        users_per_area=users_per_area,
                        uav_count=uav_count,
                        solver_name=solver_name,
                    )
                    if current_case_key in completed_case_keys:
                        continue

                    progress.set_postfix_str(
                        f"seed={seed} n={node_count} u={users_per_area} k={uav_count} | {solver_name}",
                        refresh=False,
                    )
                    row = _execute_scalability_case(
                        seed=seed,
                        node_count=node_count,
                        users_per_area=users_per_area,
                        uav_count=uav_count,
                        solver_name=solver_name,
                        fairness_tolerance=fairness_tolerance,
                        memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                        process_memory_cap_bytes=process_memory_cap_bytes,
                        wall_clock_limit_seconds=wall_clock_limit_seconds,
                    )
                    rows.append(row)
                    pending_rows.append(row)
                    completed_case_keys.add(current_case_key)
                    progress.update(1)
                    if len(pending_rows) >= checkpoint_interval:
                        _persist_checkpoint(
                            output_path,
                            boundaries_output_path,
                            all_rows=rows,
                            pending_rows=pending_rows,
                        )

    _persist_checkpoint(
        output_path,
        boundaries_output_path,
        all_rows=rows,
        pending_rows=pending_rows,
    )


if __name__ == "__main__":
    main()
