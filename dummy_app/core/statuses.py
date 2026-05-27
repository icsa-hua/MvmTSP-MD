from __future__ import annotations

from typing import Optional


def compute_absolute_gap(
    incumbent: Optional[float],
    best_bound: Optional[float],
) -> Optional[float]:
    if incumbent is None or best_bound is None:
        return None

    incumbent = float(incumbent)
    best_bound = float(best_bound)
    return abs(incumbent - best_bound)


def compute_relative_gap(
    incumbent: Optional[float],
    best_bound: Optional[float],
    *,
    eps: float = 1e-9,
) -> Optional[float]:
    absolute_gap = compute_absolute_gap(incumbent, best_bound)
    if absolute_gap is None:
        return None

    incumbent = float(incumbent)
    if abs(incumbent) < eps:
        return None

    return absolute_gap / abs(incumbent)


def compute_relative_gap_percent(
    incumbent: Optional[float],
    best_bound: Optional[float],
    *,
    eps: float = 1e-9,
) -> Optional[float]:
    gap = compute_relative_gap(incumbent, best_bound, eps=eps)
    if gap is None:
        return None
    return 100.0 * gap


def normalize_solver_status(raw_status: str) -> str:
    normalized = (raw_status or "").strip().lower()
    if normalized == "optimal":
        return "optimal"
    if normalized in {"feasible", "integer feasible"}:
        return "feasible"
    if normalized in {"not solved", "undefined"}:
        return "feasible_time_limit"
    if normalized == "infeasible":
        return "infeasible"
    if normalized == "unbounded":
        return "unbounded"
    return "error"


def infer_termination_reason(raw_status: str, time_limit_seconds: Optional[float] = None) -> str:
    normalized = normalize_solver_status(raw_status)
    if normalized == "optimal":
        return "optimal_solution_found"
    if normalized == "feasible":
        return "feasible_solution_found"
    if normalized == "feasible_time_limit":
        return "time_limit_or_undefined_solver_stop" if time_limit_seconds else "solver_stopped_without_optimality"
    if normalized == "infeasible":
        return "problem_infeasible"
    if normalized == "unbounded":
        return "problem_unbounded"
    return "solver_error"
