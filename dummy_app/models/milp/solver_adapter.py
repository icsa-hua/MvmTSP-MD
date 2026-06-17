from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pulp as pl

from dummy_app.core.statuses import (
    compute_absolute_gap,
    compute_relative_gap,
    compute_relative_gap_percent,
    infer_termination_reason,
    normalize_solver_status,
)
from dummy_app.models.milp.subtour.dfj import add_dfj_cuts, find_disconnected_subtours
from dummy_app.models.milp.subtour.strategies import normalize_subtour_mode
from dummy_app.models.milp.warm_start import apply_warm_start_to_model

_TIMESTAMPED_LINE_RE = re.compile(r"^\[(?P<elapsed>\d+(?:\.\d+)?)\]\s(?P<line>.*)$")
_MIP_PROGRESS_RE = re.compile(
    r"mip\s*=\s*(?P<incumbent>not found yet|[-+0-9.eE]+)\s*<=\s*(?P<bound>tree is empty|[-+0-9.eE]+|\+?inf)\s*(?:(?P<gap>[-+0-9.]+)%\s*)?\((?P<active>\d+);\s*(?P<processed>\d+)\)"
)
_CBC_CONTINUOUS_BOUND_RE = re.compile(
    r"Continuous objective value is\s+(?P<bound>[-+0-9.eE]+)\s+-\s+(?P<elapsed>[-+0-9.eE]+)\s+seconds"
)
_CBC_INTEGER_SOLUTION_RE = re.compile(
    r"Cbc0012I Integer solution of\s+(?P<incumbent>[-+0-9.eE]+)\s+found.*\((?P<elapsed>[-+0-9.eE]+)\s+seconds\)"
)
_CBC_SEARCH_COMPLETED_RE = re.compile(
    r"Cbc0001I Search completed - best objective\s+(?P<incumbent>[-+0-9.eE]+),\s+took\s+\d+\s+iterations and\s+(?P<nodes>\d+)\s+nodes\s+\((?P<elapsed>[-+0-9.eE]+)\s+seconds\)"
)
_CBC_PARTIAL_SEARCH_RE = re.compile(
    r"Cbc0005I Partial search - best objective\s+(?P<incumbent>[-+0-9.eE]+)\s+\(best possible\s+(?P<bound>[-+0-9.eE]+)\),\s+took\s+\d+\s+iterations and\s+(?P<nodes>\d+)\s+nodes\s+\((?P<elapsed>[-+0-9.eE]+)\s+seconds\)"
)
_CBC_LOWER_BOUND_RE = re.compile(r"Lower bound:\s+(?P<bound>[-+0-9.eE]+)")
_CBC_OBJECTIVE_VALUE_RE = re.compile(r"Objective value:\s+(?P<incumbent>[-+0-9.eE]+)")
_GUROBI_HEURISTIC_RE = re.compile(r"Found heuristic solution:\s+objective\s+(?P<incumbent>[-+0-9.eE]+)")
_GUROBI_PROGRESS_RE = re.compile(
    r"^\s*(?P<explored>\d+)\s+(?P<active>\d+)\s+(?P<node_bound>[-+0-9.eE]+|-)\s+\d+\s+\d+\s+(?P<incumbent>[-+0-9.eE]+)\s+(?P<bound>[-+0-9.eE]+)\s+(?P<gap>[-+0-9.]+)%\s+.*?(?P<elapsed>\d+(?:\.\d+)?)s\s*$"
)
_GUROBI_HEURISTIC_PROGRESS_RE = re.compile(
    r"^\s*H\s+(?P<explored>\d+)\s+(?P<active>\d+)\s+(?P<incumbent>[-+0-9.eE]+)\s+(?P<bound>[-+0-9.eE]+)\s+(?P<gap>[-+0-9.]+)%\s+.*?(?P<elapsed>\d+(?:\.\d+)?)s\s*$"
)
_GUROBI_FINAL_RE = re.compile(
    r"Best objective\s+(?P<incumbent>[-+0-9.eE]+),\s+best bound\s+(?P<bound>[-+0-9.eE]+),\s+gap\s+(?P<gap>[-+0-9.]+)%"
)
_GUROBI_EXPLORED_RE = re.compile(
    r"Explored\s+(?P<nodes>\d+)\s+nodes.*in\s+(?P<elapsed>[-+0-9.eE]+)\s+seconds"
)
_CPLEX_FOUND_INCUMBENT_RE = re.compile(
    r"Found incumbent of value\s+(?P<incumbent>[-+0-9.eE]+)\s+after\s+(?P<elapsed>[-+0-9.eE]+)\s+sec\."
)
_CPLEX_PROGRESS_RE = re.compile(
    r"^\s*(?:\*?\s*)?(?P<explored>\d+)\+?\s+(?P<active>\d+)\s+.+?\s+(?P<incumbent>[-+0-9.eE]+)\s+(?P<bound>[-+0-9.eE]+)\s+(?:\d+\s+)?(?P<gap>[-+0-9.]+)%\s*$"
)
_CPLEX_TIME_LIMIT_FEASIBLE_RE = re.compile(
    r"MIP - Time limit exceeded, integer feasible:\s+Objective =\s+(?P<incumbent>[-+0-9.eE]+)"
)
_CPLEX_TIME_LIMIT_NO_SOLUTION_RE = re.compile(r"MIP - Time limit exceeded, no integer solution")
_CPLEX_OPTIMAL_RE = re.compile(r"MIP - Integer optimal solution:\s+Objective =\s+(?P<incumbent>[-+0-9.eE]+)")
_CPLEX_FINAL_BOUND_RE = re.compile(
    r"Current MIP best bound =\s+(?P<bound>[-+0-9.eE]+)\s+\(gap = .*?,\s*(?P<gap>[-+0-9.]+)%\)"
)
_CPLEX_SOLUTION_TIME_RE = re.compile(
    r"Solution time =\s+(?P<elapsed>[-+0-9.eE]+)\s+sec\.\s+Iterations =\s+\d+\s+Nodes =\s+(?P<nodes>\d+)"
)
_CBC_MIP_START_ACCEPTED_RE = re.compile(
    r"MIPStart(?: provided solution with cost| values read for).*?(?P<objective>[-+0-9.eE]+)?",
    re.IGNORECASE,
)
_CBC_MIP_START_REJECTED_RE = re.compile(
    r"(mipstart values could not be used|mipstart rejected|mipstart infeasible)",
    re.IGNORECASE,
)
_GUROBI_MIP_START_ACCEPTED_RE = re.compile(
    r"User MIP start produced solution with objective\s+(?P<objective>[-+0-9.eE]+)",
    re.IGNORECASE,
)
_GUROBI_MIP_START_REJECTED_RE = re.compile(
    r"(User MIP start did not produce a new incumbent solution|User MIP start violates constraint)",
    re.IGNORECASE,
)
_CPLEX_MIP_START_ACCEPTED_RE = re.compile(
    r"MIP start .*?(?:defined initial solution|provided solution).*?objective\s*=?\s*(?P<objective>[-+0-9.eE]+)",
    re.IGNORECASE,
)
_CPLEX_MIP_START_ACCEPTED_COUNT_RE = re.compile(
    r"(?P<accepted>\d+)\s+of\s+(?P<provided>\d+)\s+MIP starts provided solutions",
    re.IGNORECASE,
)
_CPLEX_MIP_START_REJECTED_RE = re.compile(
    r"(No solution found from \d+ MIP starts|MIP start .*?(?:did not produce a new incumbent solution|rejected|infeasible|invalid))",
    re.IGNORECASE,
)


def _parse_float_token(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    normalized = str(raw_value).strip().lower()
    if normalized in {"", "not found yet", "tree is empty", "inf", "+inf"}:
        return None
    try:
        return float(raw_value)
    except ValueError:
        return None


def _parse_incumbent_token(raw_value: str | None) -> float | None:
    value = _parse_float_token(raw_value)
    if value is None:
        return None
    if abs(float(value)) >= 1e49:
        return None
    return float(value)


def _make_progress_event(
    *,
    elapsed_time_seconds: float,
    incumbent_value: float | None,
    best_bound: float | None,
    active_bnb_nodes: int = 0,
    explored_bnb_nodes: int = 0,
    relative_gap_percent: float | None = None,
    raw_line: str,
) -> Dict[str, Any]:
    # Normalize progress gaps across solvers from the parsed incumbent and bound.
    computed_gap = compute_relative_gap_percent(incumbent_value, best_bound)
    if computed_gap is None:
        computed_gap = relative_gap_percent
    return {
        "elapsed_time_seconds": float(elapsed_time_seconds),
        "incumbent_value": float(incumbent_value) if incumbent_value is not None else None,
        "best_bound": float(best_bound) if best_bound is not None else None,
        "relative_gap_percent": float(computed_gap) if computed_gap is not None else None,
        "active_bnb_nodes": int(active_bnb_nodes),
        "explored_bnb_nodes": int(explored_bnb_nodes),
        "raw_line": raw_line,
    }


def _build_progress_summary(
    *,
    progress_events: List[Dict[str, Any]],
    raw_status: str,
    log_path: str,
    optimality_proven: bool = False,
    feasible_solution_found: bool | None = None,
    status_hint: str | None = None,
    time_limit_reached: bool = False,
) -> Dict[str, Any]:
    feasible_events = [event for event in progress_events if event.get("incumbent_value") is not None]
    first_feasible_event = feasible_events[0] if feasible_events else None
    final_event = progress_events[-1] if progress_events else {}
    has_feasible = bool(feasible_events) if feasible_solution_found is None else bool(feasible_solution_found)
    effective_optimality_proven = bool(
        optimality_proven or (raw_status == "Optimal" and status_hint in {"", None, "Optimal"})
    )

    return {
        "progress_events": progress_events,
        "first_feasible_time_seconds": first_feasible_event.get("elapsed_time_seconds") if first_feasible_event else None,
        "first_optimality_gap_percent": first_feasible_event.get("relative_gap_percent") if first_feasible_event else None,
        "explored_bnb_nodes": final_event.get("explored_bnb_nodes", 0) if progress_events else 0,
        "active_bnb_nodes": final_event.get("active_bnb_nodes", 0) if progress_events else 0,
        "optimality_proven": effective_optimality_proven,
        "feasible_solution_found": has_feasible or effective_optimality_proven,
        "final_incumbent_value": final_event.get("incumbent_value"),
        "final_best_bound": final_event.get("best_bound"),
        "final_relative_gap_percent": final_event.get("relative_gap_percent"),
        "status_hint": status_hint,
        "time_limit_reached": bool(time_limit_reached),
        "log_path": log_path,
    }


def _wrapper_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "tools" / "glpk_wrapper.py")


def _instantiate_solver_command(factory, *, warm_start_requested: bool, **kwargs) -> tuple[Any, bool]:
    if warm_start_requested:
        try:
            return factory(warmStart=True, **kwargs), True
        except TypeError:
            return factory(**kwargs), False
    return factory(**kwargs), False


def _parse_warm_start_acceptance_from_log(log_path: str, solver_backend: str) -> Dict[str, Any]:
    if not log_path or not os.path.exists(log_path):
        return {"accepted": None, "source": "", "objective_value": None}

    accepted_patterns = []
    rejected_patterns = []
    if solver_backend == "cbc":
        accepted_patterns = [_CBC_MIP_START_ACCEPTED_RE]
        rejected_patterns = [_CBC_MIP_START_REJECTED_RE]
    elif solver_backend == "gurobi":
        accepted_patterns = [_GUROBI_MIP_START_ACCEPTED_RE]
        rejected_patterns = [_GUROBI_MIP_START_REJECTED_RE]
    elif solver_backend == "cplex":
        accepted_patterns = [_CPLEX_MIP_START_ACCEPTED_RE, _CPLEX_MIP_START_ACCEPTED_COUNT_RE]
        rejected_patterns = [_CPLEX_MIP_START_REJECTED_RE]

    with open(log_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            for pattern in rejected_patterns:
                if pattern.search(line):
                    return {"accepted": False, "source": "solver_log_rejected", "objective_value": None}
            for pattern in accepted_patterns:
                match = pattern.search(line)
                if match is None:
                    continue
                objective_value = _parse_incumbent_token(match.groupdict().get("objective"))
                if pattern is _CPLEX_MIP_START_ACCEPTED_COUNT_RE:
                    accepted_count = int(match.group("accepted"))
                    provided_count = int(match.group("provided"))
                    return {
                        "accepted": accepted_count > 0 and provided_count > 0,
                        "source": "solver_log_accepted" if accepted_count > 0 else "solver_log_rejected",
                        "objective_value": None,
                    }
                return {"accepted": True, "source": "solver_log_accepted", "objective_value": objective_value}

    return {"accepted": None, "source": "", "objective_value": None}


def _infer_warm_start_acceptance_from_progress(progress_summary: Dict[str, Any], expected_objective: float | None) -> Dict[str, Any]:
    if expected_objective is None:
        return {"accepted": None, "source": "", "objective_value": None}

    feasible_events = [
        event
        for event in progress_summary.get("progress_events", [])
        if event.get("incumbent_value") is not None
    ]
    if not feasible_events:
        return {"accepted": None, "source": "", "objective_value": None}

    first_event = feasible_events[0]
    incumbent_value = float(first_event["incumbent_value"])
    elapsed_time_seconds = float(first_event.get("elapsed_time_seconds", 0.0) or 0.0)
    tolerance = max(1e-4, abs(float(expected_objective)) * 1e-4)
    if elapsed_time_seconds <= 1e-6 and abs(incumbent_value - float(expected_objective)) <= tolerance:
        return {
            "accepted": True,
            "source": "first_incumbent_match",
            "objective_value": incumbent_value,
        }
    return {"accepted": None, "source": "", "objective_value": None}


def _update_cluster_warm_start_summary(cluster: Any, **fields: Any) -> None:
    warm_start_summary = dict(getattr(cluster, "warm_start_summary", {}))
    warm_start_summary.update(fields)
    cluster.warm_start_summary = warm_start_summary


def _parse_glpk_progress(log_path: str, objective_value: float | None, raw_status: str) -> Dict[str, Any]:
    progress_events: List[Dict[str, Any]] = []
    optimality_proven = False

    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                timestamp_match = _TIMESTAMPED_LINE_RE.match(line)
                if timestamp_match is None:
                    continue

                elapsed_seconds = float(timestamp_match.group("elapsed"))
                payload = timestamp_match.group("line").strip()
                if "INTEGER OPTIMAL SOLUTION FOUND" in payload.upper():
                    optimality_proven = True

                progress_match = _MIP_PROGRESS_RE.search(payload)
                if progress_match is None:
                    continue

                incumbent_value = _parse_float_token(progress_match.group("incumbent"))
                best_bound = _parse_float_token(progress_match.group("bound"))
                gap_value = _parse_float_token(progress_match.group("gap"))
                if best_bound is None and str(progress_match.group("bound")).strip().lower() == "tree is empty":
                    best_bound = incumbent_value
                    gap_value = 0.0

                progress_events.append(
                    _make_progress_event(
                        elapsed_time_seconds=elapsed_seconds,
                        incumbent_value=incumbent_value,
                        best_bound=best_bound,
                        active_bnb_nodes=int(progress_match.group("active")),
                        explored_bnb_nodes=int(progress_match.group("processed")),
                        relative_gap_percent=gap_value,
                        raw_line=payload,
                    )
                )

    feasible_events = [event for event in progress_events if event.get("incumbent_value") is not None]
    if objective_value is not None and (raw_status == "Optimal" or feasible_events):
        final_elapsed = progress_events[-1]["elapsed_time_seconds"] if progress_events else 0.0
        final_bound = progress_events[-1]["best_bound"] if progress_events else None
        final_gap_percent = progress_events[-1]["relative_gap_percent"] if progress_events else None
        if raw_status == "Optimal" and optimality_proven:
            final_bound = objective_value
            final_gap_percent = 0.0
        progress_events.append(
            _make_progress_event(
                elapsed_time_seconds=float(final_elapsed),
                incumbent_value=float(objective_value),
                best_bound=final_bound,
                active_bnb_nodes=int(progress_events[-1]["active_bnb_nodes"]) if progress_events else 0,
                explored_bnb_nodes=int(progress_events[-1]["explored_bnb_nodes"]) if progress_events else 0,
                relative_gap_percent=final_gap_percent,
                raw_line="final_solution",
            )
        )
    return _build_progress_summary(
        progress_events=progress_events,
        raw_status=raw_status,
        log_path=log_path,
        optimality_proven=optimality_proven,
    )


def _parse_cbc_progress(log_path: str, objective_value: float | None, raw_status: str) -> Dict[str, Any]:
    progress_events: List[Dict[str, Any]] = []
    final_bound = None
    final_incumbent = None
    final_gap_percent = None
    final_elapsed = 0.0
    explored_nodes = 0
    optimality_proven = False
    feasible_solution_found = False
    time_limit_reached = False
    status_hint = None

    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if "Result - Optimal solution found" in line:
                    optimality_proven = True
                    status_hint = "Optimal"
                elif "Result - Stopped on time limit" in line or "Cbc0020I Exiting on maximum time" in line:
                    time_limit_reached = True
                    status_hint = "Not Solved"
                elif "No feasible solution found" in line:
                    feasible_solution_found = False

                continuous_match = _CBC_CONTINUOUS_BOUND_RE.search(line)
                if continuous_match is not None:
                    final_bound = _parse_float_token(continuous_match.group("bound"))
                    final_elapsed = float(continuous_match.group("elapsed"))
                    continue

                integer_match = _CBC_INTEGER_SOLUTION_RE.search(line)
                if integer_match is not None:
                    feasible_solution_found = True
                    final_incumbent = _parse_incumbent_token(integer_match.group("incumbent"))
                    final_elapsed = float(integer_match.group("elapsed"))
                    progress_events.append(
                        _make_progress_event(
                            elapsed_time_seconds=final_elapsed,
                            incumbent_value=final_incumbent,
                            best_bound=final_bound,
                            explored_bnb_nodes=explored_nodes,
                            raw_line=line,
                        )
                    )
                    continue

                partial_match = _CBC_PARTIAL_SEARCH_RE.search(line)
                if partial_match is not None:
                    final_elapsed = float(partial_match.group("elapsed"))
                    explored_nodes = int(partial_match.group("nodes"))
                    final_incumbent = _parse_incumbent_token(partial_match.group("incumbent"))
                    final_bound = _parse_float_token(partial_match.group("bound"))
                    final_gap_percent = compute_relative_gap_percent(final_incumbent, final_bound)
                    feasible_solution_found = feasible_solution_found or final_incumbent is not None
                    progress_events.append(
                        _make_progress_event(
                            elapsed_time_seconds=final_elapsed,
                            incumbent_value=final_incumbent,
                            best_bound=final_bound,
                            explored_bnb_nodes=explored_nodes,
                            relative_gap_percent=final_gap_percent,
                            raw_line=line,
                        )
                    )
                    continue

                search_match = _CBC_SEARCH_COMPLETED_RE.search(line)
                if search_match is not None:
                    final_elapsed = float(search_match.group("elapsed"))
                    explored_nodes = int(search_match.group("nodes"))
                    final_incumbent = _parse_incumbent_token(search_match.group("incumbent"))
                    feasible_solution_found = feasible_solution_found or final_incumbent is not None
                    if optimality_proven:
                        final_bound = final_incumbent
                        final_gap_percent = 0.0
                    progress_events.append(
                        _make_progress_event(
                            elapsed_time_seconds=final_elapsed,
                            incumbent_value=final_incumbent,
                            best_bound=final_bound,
                            explored_bnb_nodes=explored_nodes,
                            relative_gap_percent=final_gap_percent,
                            raw_line=line,
                        )
                    )
                    continue

                lower_bound_match = _CBC_LOWER_BOUND_RE.search(line)
                if lower_bound_match is not None:
                    final_bound = _parse_float_token(lower_bound_match.group("bound"))
                    continue

                objective_match = _CBC_OBJECTIVE_VALUE_RE.search(line)
                if objective_match is not None:
                    final_incumbent = _parse_incumbent_token(objective_match.group("incumbent"))
                    feasible_solution_found = feasible_solution_found or final_incumbent is not None

    if raw_status == "Optimal" and objective_value is not None and optimality_proven:
        final_incumbent = float(objective_value)
        final_bound = float(objective_value)
        final_gap_percent = 0.0
        status_hint = "Optimal"
    elif final_incumbent is None and feasible_solution_found and objective_value is not None:
        final_incumbent = float(objective_value)

    if final_incumbent is not None or final_bound is not None:
        progress_events.append(
            _make_progress_event(
                elapsed_time_seconds=final_elapsed,
                incumbent_value=final_incumbent,
                best_bound=final_bound,
                explored_bnb_nodes=explored_nodes,
                relative_gap_percent=final_gap_percent,
                raw_line="final_solution",
            )
        )

    return _build_progress_summary(
        progress_events=progress_events,
        raw_status=raw_status,
        log_path=log_path,
        optimality_proven=optimality_proven,
        feasible_solution_found=feasible_solution_found,
        status_hint=status_hint,
        time_limit_reached=time_limit_reached,
    )


def _parse_gurobi_progress(log_path: str, objective_value: float | None, raw_status: str) -> Dict[str, Any]:
    progress_events: List[Dict[str, Any]] = []
    final_incumbent = None
    final_bound = None
    final_gap_percent = None
    final_elapsed = 0.0
    explored_nodes = 0
    active_nodes = 0
    optimality_proven = False
    feasible_solution_found = False
    time_limit_reached = False
    status_hint = None

    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if "Optimal solution found" in line:
                    optimality_proven = True
                    status_hint = "Optimal"
                elif "Model is infeasible" in line:
                    status_hint = "Infeasible"
                elif "Time limit reached" in line:
                    time_limit_reached = True
                    status_hint = "Feasible" if feasible_solution_found else "Not Solved"

                heuristic_match = _GUROBI_HEURISTIC_RE.search(line)
                if heuristic_match is not None:
                    feasible_solution_found = True
                    final_incumbent = _parse_incumbent_token(heuristic_match.group("incumbent"))
                    progress_events.append(
                        _make_progress_event(
                            elapsed_time_seconds=0.0,
                            incumbent_value=final_incumbent,
                            best_bound=final_bound,
                            raw_line=line,
                        )
                    )
                    continue

                heuristic_progress_match = _GUROBI_HEURISTIC_PROGRESS_RE.search(line)
                if heuristic_progress_match is not None:
                    feasible_solution_found = True
                    explored_nodes = int(heuristic_progress_match.group("explored"))
                    active_nodes = int(heuristic_progress_match.group("active"))
                    final_elapsed = float(heuristic_progress_match.group("elapsed"))
                    final_incumbent = _parse_incumbent_token(heuristic_progress_match.group("incumbent"))
                    final_bound = _parse_float_token(heuristic_progress_match.group("bound"))
                    final_gap_percent = _parse_float_token(heuristic_progress_match.group("gap"))
                    progress_events.append(
                        _make_progress_event(
                            elapsed_time_seconds=final_elapsed,
                            incumbent_value=final_incumbent,
                            best_bound=final_bound,
                            active_bnb_nodes=active_nodes,
                            explored_bnb_nodes=explored_nodes,
                            relative_gap_percent=final_gap_percent,
                            raw_line=line,
                        )
                    )
                    continue

                progress_match = _GUROBI_PROGRESS_RE.search(line)
                if progress_match is not None:
                    explored_nodes = int(progress_match.group("explored"))
                    active_nodes = int(progress_match.group("active"))
                    final_elapsed = float(progress_match.group("elapsed"))
                    final_incumbent = _parse_incumbent_token(progress_match.group("incumbent"))
                    final_bound = _parse_float_token(progress_match.group("bound"))
                    final_gap_percent = _parse_float_token(progress_match.group("gap"))
                    feasible_solution_found = feasible_solution_found or final_incumbent is not None
                    progress_events.append(
                        _make_progress_event(
                            elapsed_time_seconds=final_elapsed,
                            incumbent_value=final_incumbent,
                            best_bound=final_bound,
                            active_bnb_nodes=active_nodes,
                            explored_bnb_nodes=explored_nodes,
                            relative_gap_percent=final_gap_percent,
                            raw_line=line,
                        )
                    )
                    continue

                final_match = _GUROBI_FINAL_RE.search(line)
                if final_match is not None:
                    final_incumbent = _parse_incumbent_token(final_match.group("incumbent"))
                    final_bound = _parse_float_token(final_match.group("bound"))
                    final_gap_percent = _parse_float_token(final_match.group("gap"))
                    feasible_solution_found = feasible_solution_found or final_incumbent is not None
                    continue

                explored_match = _GUROBI_EXPLORED_RE.search(line)
                if explored_match is not None:
                    explored_nodes = int(explored_match.group("nodes"))
                    final_elapsed = float(explored_match.group("elapsed"))

    if raw_status == "Optimal" and objective_value is not None and optimality_proven:
        final_incumbent = float(objective_value)
        final_bound = float(objective_value)
        final_gap_percent = 0.0
        status_hint = "Optimal"
    elif final_incumbent is None and feasible_solution_found and objective_value is not None:
        final_incumbent = float(objective_value)

    if final_incumbent is not None or final_bound is not None:
        progress_events.append(
            _make_progress_event(
                elapsed_time_seconds=final_elapsed,
                incumbent_value=final_incumbent,
                best_bound=final_bound,
                active_bnb_nodes=active_nodes,
                explored_bnb_nodes=explored_nodes,
                relative_gap_percent=final_gap_percent,
                raw_line="final_solution",
            )
        )

    return _build_progress_summary(
        progress_events=progress_events,
        raw_status=raw_status,
        log_path=log_path,
        optimality_proven=optimality_proven,
        feasible_solution_found=feasible_solution_found,
        status_hint=status_hint,
        time_limit_reached=time_limit_reached,
    )


def _parse_cplex_progress(log_path: str, objective_value: float | None, raw_status: str) -> Dict[str, Any]:
    progress_events: List[Dict[str, Any]] = []
    final_incumbent = None
    final_bound = None
    final_gap_percent = None
    final_elapsed = 0.0
    explored_nodes = 0
    active_nodes = 0
    optimality_proven = False
    feasible_solution_found = False
    time_limit_reached = False
    status_hint = None

    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")

                found_incumbent_match = _CPLEX_FOUND_INCUMBENT_RE.search(line)
                if found_incumbent_match is not None:
                    feasible_solution_found = True
                    final_incumbent = _parse_incumbent_token(found_incumbent_match.group("incumbent"))
                    final_elapsed = float(found_incumbent_match.group("elapsed"))
                    progress_events.append(
                        _make_progress_event(
                            elapsed_time_seconds=final_elapsed,
                            incumbent_value=final_incumbent,
                            best_bound=final_bound,
                            raw_line=line,
                        )
                    )
                    continue

                progress_match = _CPLEX_PROGRESS_RE.search(line)
                if progress_match is not None:
                    explored_nodes = int(progress_match.group("explored"))
                    active_nodes = int(progress_match.group("active"))
                    final_incumbent = _parse_incumbent_token(progress_match.group("incumbent"))
                    final_bound = _parse_float_token(progress_match.group("bound"))
                    final_gap_percent = _parse_float_token(progress_match.group("gap"))
                    feasible_solution_found = feasible_solution_found or final_incumbent is not None
                    progress_events.append(
                        _make_progress_event(
                            elapsed_time_seconds=final_elapsed,
                            incumbent_value=final_incumbent,
                            best_bound=final_bound,
                            active_bnb_nodes=active_nodes,
                            explored_bnb_nodes=explored_nodes,
                            relative_gap_percent=final_gap_percent,
                            raw_line=line,
                        )
                    )
                    continue

                optimal_match = _CPLEX_OPTIMAL_RE.search(line)
                if optimal_match is not None:
                    optimality_proven = True
                    status_hint = "Optimal"
                    final_incumbent = _parse_incumbent_token(optimal_match.group("incumbent"))
                    final_bound = final_incumbent
                    final_gap_percent = 0.0
                    feasible_solution_found = feasible_solution_found or final_incumbent is not None
                    continue

                time_limit_feasible_match = _CPLEX_TIME_LIMIT_FEASIBLE_RE.search(line)
                if time_limit_feasible_match is not None:
                    time_limit_reached = True
                    status_hint = "Feasible"
                    final_incumbent = _parse_incumbent_token(time_limit_feasible_match.group("incumbent"))
                    feasible_solution_found = feasible_solution_found or final_incumbent is not None
                    continue

                if _CPLEX_TIME_LIMIT_NO_SOLUTION_RE.search(line) is not None:
                    time_limit_reached = True
                    status_hint = "Not Solved"
                    continue

                final_bound_match = _CPLEX_FINAL_BOUND_RE.search(line)
                if final_bound_match is not None:
                    final_bound = _parse_float_token(final_bound_match.group("bound"))
                    final_gap_percent = _parse_float_token(final_bound_match.group("gap"))
                    continue

                solution_time_match = _CPLEX_SOLUTION_TIME_RE.search(line)
                if solution_time_match is not None:
                    final_elapsed = float(solution_time_match.group("elapsed"))
                    explored_nodes = int(solution_time_match.group("nodes"))

    if raw_status == "Optimal" and objective_value is not None and optimality_proven:
        final_incumbent = float(objective_value)
        final_bound = float(objective_value)
        final_gap_percent = 0.0
        status_hint = "Optimal"
    elif final_incumbent is None and feasible_solution_found and objective_value is not None:
        final_incumbent = float(objective_value)

    if final_incumbent is not None or final_bound is not None:
        progress_events.append(
            _make_progress_event(
                elapsed_time_seconds=final_elapsed,
                incumbent_value=final_incumbent,
                best_bound=final_bound,
                active_bnb_nodes=active_nodes,
                explored_bnb_nodes=explored_nodes,
                relative_gap_percent=final_gap_percent,
                raw_line="final_solution",
            )
        )

    return _build_progress_summary(
        progress_events=progress_events,
        raw_status=raw_status,
        log_path=log_path,
        optimality_proven=optimality_proven,
        feasible_solution_found=feasible_solution_found,
        status_hint=status_hint,
        time_limit_reached=time_limit_reached,
    )


def _create_solver_log_path(solver_backend: str, cluster_id: Any) -> str:
    log_fd, log_path = tempfile.mkstemp(prefix=f"mvmtsp-{solver_backend}-cluster-{cluster_id}-", suffix=".log")
    os.close(log_fd)
    return log_path


def _resolve_effective_raw_status(raw_status: str, progress_summary: Dict[str, Any]) -> str:
    if progress_summary.get("optimality_proven"):
        return "Optimal"
    if progress_summary.get("time_limit_reached"):
        return "Not Solved"
    status_hint = str(progress_summary.get("status_hint") or "").strip()
    if status_hint:
        return status_hint
    if progress_summary.get("feasible_solution_found"):
        return "Feasible"
    return raw_status


def _annotate_dfj_metadata(
    solve_metadata: Dict[str, Any],
    *,
    dfj_rounds: int,
    dfj_solve_passes: int,
    dfj_cuts_added: int,
    dfj_round_history: List[Dict[str, Any]],
    violated_subtours: List[tuple[int, tuple[int, ...]]] | None = None,
) -> Dict[str, Any]:
    annotated = dict(solve_metadata)
    annotated["dfj_rounds"] = int(dfj_rounds)
    annotated["dfj_solve_passes"] = int(dfj_solve_passes)
    annotated["dfj_cuts_added"] = int(dfj_cuts_added)
    annotated["dfj_round_history"] = [dict(entry) for entry in dfj_round_history]
    annotated["violated_subtours"] = [
        {"agent_id": int(agent_id), "nodes": list(subset)}
        for agent_id, subset in (violated_subtours or [])
    ]
    return annotated


def _mark_dfj_cut_limit_reached(
    cluster: Any,
    last_solve_metadata: Dict[str, Any],
    *,
    dfj_rounds: int,
    dfj_solve_passes: int,
    dfj_cuts_added: int,
    dfj_round_history: List[Dict[str, Any]],
    violated_subtours: List[tuple[int, tuple[int, ...]]],
) -> Dict[str, Any]:
    cluster.problem.status = pl.LpStatusNotSolved
    elapsed_time_seconds = float(
        sum(float(round_info.get("elapsed_time_seconds", 0.0) or 0.0) for round_info in dfj_round_history)
    )
    solve_metadata = dict(last_solve_metadata)
    solve_metadata.update(
        {
            "status_code": int(cluster.problem.status),
            "raw_status": "Not Solved",
            "normalized_status": normalize_solver_status("Not Solved"),
            "objective_value": None,
            "incumbent_value": None,
            "best_bound": None,
            "absolute_gap": None,
            "relative_gap": None,
            "relative_gap_percent": None,
            "elapsed_time_seconds": elapsed_time_seconds,
            "termination_reason": "dfj_cut_limit_reached",
            "feasible_solution_found": False,
            "optimality_proven": False,
        }
    )
    return _annotate_dfj_metadata(
        solve_metadata,
        dfj_rounds=dfj_rounds,
        dfj_solve_passes=dfj_solve_passes,
        dfj_cuts_added=dfj_cuts_added,
        dfj_round_history=dfj_round_history,
        violated_subtours=violated_subtours,
    )


def _solve_cluster_problem_once(
    cluster: Any,
    builder: Any,
    time_limit_seconds=None,
    solver_backend: str = "glpk",
    solver_seed: int = 42,
) -> Dict[str, Any]:

    if solver_backend not in {"gurobi", "glpk", "cbc", "cplex"}:
        raise ValueError(f"Unsupported solver backend '{solver_backend}'. Only 'glpk' is currently implemented.")

    solve_started_at = time.perf_counter()
    progress_summary: Dict[str, Any] = {
        "progress_events": [],
        "first_feasible_time_seconds": None,
        "first_optimality_gap_percent": None,
        "explored_bnb_nodes": 0,
        "active_bnb_nodes": 0,
        "optimality_proven": False,
        "feasible_solution_found": False,
        "final_incumbent_value": None,
        "final_best_bound": None,
        "final_relative_gap_percent": None,
        "status_hint": None,
        "time_limit_reached": False,
        "log_path": "",
    }
    warm_start_application = apply_warm_start_to_model(cluster, builder)
    warm_start_requested = bool(warm_start_application.get("applied"))
    warm_start_provided_to_solver = False
    if warm_start_application.get("requested"):
        _update_cluster_warm_start_summary(
            cluster,
            provided_to_solver=False,
            accepted_by_solver=None,
            acceptance_source=warm_start_application.get("reason", ""),
        )

    if solver_backend == "glpk":
        log_path = _create_solver_log_path("glpk", cluster.id)
        env_keys = ("MVMTSP_GLPK_REAL_BIN", "MVMTSP_GLPK_LOG_PATH", "MVMTSP_GLPK_ECHO")
        previous_env = {key: os.environ.get(key) for key in env_keys}
        os.environ["MVMTSP_GLPK_REAL_BIN"] = shutil.which("glpsol") or "glpsol"
        os.environ["MVMTSP_GLPK_LOG_PATH"] = log_path
        os.environ["MVMTSP_GLPK_ECHO"] = "0"
        try:
            cluster.problem.solve(
                pl.GLPK_CMD(
                    path=_wrapper_path(),
                    timeLimit=time_limit_seconds,
                    msg=False,
                    options=["--mipgap", "0.0", "--seed", str(int(solver_seed))],
                )
            )
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    elif solver_backend == "gurobi":
        log_path = _create_solver_log_path("gurobi", cluster.id)
        solver_command, warm_start_provided_to_solver = _instantiate_solver_command(
            pl.GUROBI_CMD,
            warm_start_requested=warm_start_requested,
            timeLimit=time_limit_seconds,
            msg=False,
            gapRel=0.0,
            logPath=log_path,
            options=[("Seed", int(solver_seed))],
        )
        try:
            cluster.problem.solve(solver_command)
        except TypeError:
            # PuLP bug (gurobi_api.py): when Gurobi finds no feasible solution it
            # writes no .sol file; PuLP then calls assignVarsVals(None) which raises
            # TypeError.  cluster.problem.status is already LpStatusNotSolved (0)
            # at this point, so just let the rest of the function handle it.
            pass

    elif solver_backend == "cbc":
        log_path = _create_solver_log_path("cbc", cluster.id)
        solver_command, warm_start_provided_to_solver = _instantiate_solver_command(
            pl.PULP_CBC_CMD,
            warm_start_requested=warm_start_requested,
            timeLimit=time_limit_seconds,
            msg=False,
            gapRel=0.0,
            logPath=log_path,
            options=[f"randomSeed {int(solver_seed)}"],
        )
        cluster.problem.solve(solver_command)

    else:
        log_path = _create_solver_log_path("cplex", cluster.id)
        solver_command, warm_start_provided_to_solver = _instantiate_solver_command(
            pl.CPLEX_CMD,
            warm_start_requested=warm_start_requested,
            path="/Users/jimborg/Applications/CPLEX_Studio2212/cplex/bin/arm64_osx/cplex",
            timeLimit=time_limit_seconds,
            msg=False,
            gapRel=0.0,
            logPath=log_path,
            options=[f"set randomseed {int(solver_seed)}"],
        )
        cluster.problem.solve(solver_command)

    elapsed_time_seconds = time.perf_counter() - solve_started_at
    raw_status = pl.LpStatus.get(cluster.problem.status, "Unknown")
    pulp_objective_value = None
    try:
        pulp_objective_value = float(pl.value(cluster.problem.objective))
    except Exception:
        pulp_objective_value = None

    if solver_backend == "glpk":
        progress_summary = _parse_glpk_progress(log_path, pulp_objective_value, raw_status)
    elif solver_backend == "cbc":
        progress_summary = _parse_cbc_progress(log_path, pulp_objective_value, raw_status)
    elif solver_backend == "gurobi":
        progress_summary = _parse_gurobi_progress(log_path, pulp_objective_value, raw_status)
    else:
        progress_summary = _parse_cplex_progress(log_path, pulp_objective_value, raw_status)

    if warm_start_requested:
        acceptance = _parse_warm_start_acceptance_from_log(log_path, solver_backend)
        if acceptance.get("accepted") is None:
            acceptance = _infer_warm_start_acceptance_from_progress(
                progress_summary=progress_summary,
                expected_objective=dict(getattr(cluster, "warm_start_summary", {})).get("solver_objective_value"),
            )
        _update_cluster_warm_start_summary(
            cluster,
            provided_to_solver=bool(warm_start_provided_to_solver),
            accepted_by_solver=acceptance.get("accepted") if warm_start_provided_to_solver else None,
            acceptance_source=(
                acceptance.get("source")
                if warm_start_provided_to_solver
                else "solver_command_no_warm_start_support"
            ),
            accepted_objective_value=acceptance.get("objective_value"),
        )

    effective_raw_status = _resolve_effective_raw_status(raw_status, progress_summary)
    feasible_solution_found = bool(progress_summary.get("feasible_solution_found"))

    incumbent_value: Optional[float]
    if progress_summary.get("final_incumbent_value") is not None:
        incumbent_value = float(progress_summary["final_incumbent_value"])
    elif feasible_solution_found and pulp_objective_value is not None:
        incumbent_value = float(pulp_objective_value)
    else:
        incumbent_value = None

    objective_value = incumbent_value

    best_bound = progress_summary.get("final_best_bound")
    best_bound = float(best_bound) if best_bound is not None else None
    if progress_summary.get("optimality_proven") and incumbent_value is not None:
        best_bound = float(incumbent_value)

    relative_gap_percent = compute_relative_gap_percent(incumbent_value, best_bound)
    if relative_gap_percent is None and progress_summary.get("final_relative_gap_percent") is not None:
        relative_gap_percent = float(progress_summary["final_relative_gap_percent"])

    absolute_gap = compute_absolute_gap(incumbent_value, best_bound)
    relative_gap = compute_relative_gap(incumbent_value, best_bound)

    return {
        "status_code": int(cluster.problem.status),
        "raw_status": effective_raw_status,
        "normalized_status": normalize_solver_status(effective_raw_status),
        "objective_value": objective_value,
        "incumbent_value": incumbent_value,
        "best_bound": best_bound,
        "absolute_gap": absolute_gap,
        "relative_gap": relative_gap,
        "relative_gap_percent": relative_gap_percent,
        "time_limit_seconds": time_limit_seconds,
        "elapsed_time_seconds": float(elapsed_time_seconds),
        "termination_reason": infer_termination_reason(effective_raw_status, time_limit_seconds),
        "first_feasible_time_seconds": progress_summary.get("first_feasible_time_seconds"),
        "first_optimality_gap_percent": progress_summary.get("first_optimality_gap_percent"),
        "explored_bnb_nodes": progress_summary.get("explored_bnb_nodes"),
        "active_bnb_nodes": progress_summary.get("active_bnb_nodes"),
        "feasible_solution_found": feasible_solution_found,
        "time_limit_reached": bool(progress_summary.get("time_limit_reached")),
        "optimality_proven": progress_summary.get("optimality_proven", effective_raw_status == "Optimal"),
        "progress_events": progress_summary.get("progress_events", []),
        "solver_log_path": progress_summary.get("log_path", log_path),
        "solver_seed": int(solver_seed),
    }


def _solve_cluster_problem_with_iterative_dfj(
    cluster: Any,
    builder: Any,
    time_limit_seconds=None,
    solver_backend: str = "glpk",
    solver_seed: int = 42,
) -> Dict[str, Any]:
    max_dfj_rounds = max(1, int(getattr(builder, "max_dfj_rounds", 50) or 50))
    solve_started_at = time.perf_counter()
    round_history: List[Dict[str, Any]] = []
    total_cuts_added = 0
    last_solve_metadata: Dict[str, Any] | None = None
    last_subtours: List[tuple[int, tuple[int, ...]]] = []

    for round_id in range(max_dfj_rounds):
        round_time_limit = time_limit_seconds
        if time_limit_seconds is not None:
            elapsed_so_far = time.perf_counter() - solve_started_at
            remaining_time = float(time_limit_seconds) - float(elapsed_so_far)
            if remaining_time <= 0.0:
                cluster.problem.status = pl.LpStatusNotSolved
                timeout_metadata = dict(last_solve_metadata or {})
                timeout_metadata.update(
                    {
                        "status_code": int(cluster.problem.status),
                        "raw_status": "Not Solved",
                        "normalized_status": normalize_solver_status("Not Solved"),
                        "objective_value": None,
                        "incumbent_value": None,
                        "best_bound": None,
                        "absolute_gap": None,
                        "relative_gap": None,
                        "relative_gap_percent": None,
                        "elapsed_time_seconds": float(time_limit_seconds),
                        "time_limit_seconds": float(time_limit_seconds),
                        "termination_reason": "dfj_time_limit_reached",
                        "feasible_solution_found": False,
                        "time_limit_reached": True,
                        "optimality_proven": False,
                    }
                )
                return _annotate_dfj_metadata(
                    timeout_metadata,
                    dfj_rounds=round_id,
                    dfj_solve_passes=round_id,
                    dfj_cuts_added=total_cuts_added,
                    dfj_round_history=round_history,
                    violated_subtours=last_subtours,
                )
            round_time_limit = remaining_time

        solve_metadata = _solve_cluster_problem_once(
            cluster=cluster,
            builder=builder,
            time_limit_seconds=round_time_limit,
            solver_backend=solver_backend,
            solver_seed=solver_seed,
        )
        last_solve_metadata = solve_metadata

        round_entry = {
            "round_id": int(round_id),
            "raw_status": solve_metadata.get("raw_status"),
            "normalized_status": solve_metadata.get("normalized_status"),
            "elapsed_time_seconds": float(solve_metadata.get("elapsed_time_seconds", 0.0) or 0.0),
            "cuts_added_total": int(total_cuts_added),
        }

        if not bool(solve_metadata.get("feasible_solution_found")):
            round_history.append(round_entry)
            return _annotate_dfj_metadata(
                solve_metadata,
                dfj_rounds=round_id,
                dfj_solve_passes=round_id + 1,
                dfj_cuts_added=total_cuts_added,
                dfj_round_history=round_history,
            )

        subtours = find_disconnected_subtours(cluster)
        last_subtours = subtours
        round_entry["violated_subtours"] = [
            {"agent_id": int(agent_id), "nodes": list(subset)}
            for agent_id, subset in subtours
        ]

        if not subtours:
            round_history.append(round_entry)
            solve_metadata["elapsed_time_seconds"] = float(
                sum(float(entry.get("elapsed_time_seconds", 0.0) or 0.0) for entry in round_history)
            )
            return _annotate_dfj_metadata(
                solve_metadata,
                dfj_rounds=round_id,
                dfj_solve_passes=round_id + 1,
                dfj_cuts_added=total_cuts_added,
                dfj_round_history=round_history,
            )

        cuts_added_this_round = add_dfj_cuts(cluster, subtours)
        total_cuts_added += cuts_added_this_round
        round_entry["cuts_added_this_round"] = int(cuts_added_this_round)
        round_entry["cuts_added_total"] = int(total_cuts_added)
        round_history.append(round_entry)

        if cuts_added_this_round <= 0:
            return _mark_dfj_cut_limit_reached(
                cluster,
                solve_metadata,
                dfj_rounds=round_id + 1,
                dfj_solve_passes=round_id + 1,
                dfj_cuts_added=total_cuts_added,
                dfj_round_history=round_history,
                violated_subtours=subtours,
            )

    return _mark_dfj_cut_limit_reached(
        cluster,
        last_solve_metadata or {},
        dfj_rounds=max_dfj_rounds,
        dfj_solve_passes=max_dfj_rounds,
        dfj_cuts_added=total_cuts_added,
        dfj_round_history=round_history,
        violated_subtours=last_subtours,
    )


def solve_cluster_problem(
    cluster: Any,
    builder: Any,
    time_limit_seconds=None,
    solver_backend: str = "glpk",
    solver_seed: int = 42,
) -> Dict[str, Any]:
    subtour_mode = normalize_subtour_mode(
        getattr(builder, "subtour_mode", getattr(builder, "subtour_strategy", "mtz"))
    )
    if subtour_mode == "dfj_iter":
        return _solve_cluster_problem_with_iterative_dfj(
            cluster=cluster,
            builder=builder,
            time_limit_seconds=time_limit_seconds,
            solver_backend=solver_backend,
            solver_seed=solver_seed,
        )
    solve_metadata = _solve_cluster_problem_once(
        cluster=cluster,
        builder=builder,
        time_limit_seconds=time_limit_seconds,
        solver_backend=solver_backend,
        solver_seed=solver_seed,
    )
    return _annotate_dfj_metadata(
        solve_metadata,
        dfj_rounds=0,
        dfj_solve_passes=1,
        dfj_cuts_added=0,
        dfj_round_history=[],
    )
