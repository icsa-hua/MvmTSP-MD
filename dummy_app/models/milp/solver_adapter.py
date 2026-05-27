from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pulp as pl

from dummy_app.core.statuses import (
    compute_absolute_gap,
    compute_relative_gap,
    compute_relative_gap_percent,
    infer_termination_reason,
    normalize_solver_status,
)

_TIMESTAMPED_LINE_RE = re.compile(r"^\[(?P<elapsed>\d+(?:\.\d+)?)\]\s(?P<line>.*)$")
_MIP_PROGRESS_RE = re.compile(
    r"mip\s*=\s*(?P<incumbent>not found yet|[-+0-9.eE]+)\s*<=\s*(?P<bound>tree is empty|[-+0-9.eE]+|\+?inf)\s*(?:(?P<gap>[-+0-9.]+)%\s*)?\((?P<active>\d+);\s*(?P<processed>\d+)\)"
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


def _wrapper_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "tools" / "glpk_wrapper.py")


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
                    {
                        "elapsed_time_seconds": elapsed_seconds,
                        "incumbent_value": incumbent_value,
                        "best_bound": best_bound,
                        "relative_gap_percent": gap_value,
                        "active_bnb_nodes": int(progress_match.group("active")),
                        "explored_bnb_nodes": int(progress_match.group("processed")),
                        "raw_line": payload,
                    }
                )

    if objective_value is not None:
        final_elapsed = progress_events[-1]["elapsed_time_seconds"] if progress_events else 0.0
        final_bound = progress_events[-1]["best_bound"] if progress_events else None
        final_gap_percent = progress_events[-1]["relative_gap_percent"] if progress_events else None
        if raw_status == "Optimal":
            final_bound = objective_value
            final_gap_percent = 0.0
            optimality_proven = True
        progress_events.append(
            {
                "elapsed_time_seconds": float(final_elapsed),
                "incumbent_value": float(objective_value),
                "best_bound": float(final_bound) if final_bound is not None else None,
                "relative_gap_percent": float(final_gap_percent) if final_gap_percent is not None else None,
                "active_bnb_nodes": int(progress_events[-1]["active_bnb_nodes"]) if progress_events else 0,
                "explored_bnb_nodes": int(progress_events[-1]["explored_bnb_nodes"]) if progress_events else 0,
                "raw_line": "final_solution",
            }
        )

    feasible_events = [event for event in progress_events if event.get("incumbent_value") is not None]
    first_feasible_event = feasible_events[0] if feasible_events else None

    return {
        "progress_events": progress_events,
        "first_feasible_time_seconds": first_feasible_event.get("elapsed_time_seconds") if first_feasible_event else None,
        "first_optimality_gap_percent": first_feasible_event.get("relative_gap_percent") if first_feasible_event else None,
        "explored_bnb_nodes": progress_events[-1].get("explored_bnb_nodes", 0) if progress_events else 0,
        "active_bnb_nodes": progress_events[-1].get("active_bnb_nodes", 0) if progress_events else 0,
        "optimality_proven": bool(optimality_proven or raw_status == "Optimal"),
        "log_path": log_path,
    }


def solve_cluster_problem(
    cluster: Any,
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
        "log_path": "",
    }

    if solver_backend == "glpk":
        log_fd, log_path = tempfile.mkstemp(prefix=f"mvmtsp-glpk-cluster-{cluster.id}-", suffix=".log")
        os.close(log_fd)
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
        log_path = ""
        cluster.problem.solve(
            pl.GUROBI(
                timeLimit=time_limit_seconds,
                msg=False,
            )
        )
    elif solver_backend == "cbc":
        log_path = ""
        cluster.problem.solve(
            pl.PULP_CBC_CMD(
                timeLimit=time_limit_seconds,
                msg=False,
            )
        )
    else:
        log_path = ""
        cluster.problem.solve(
            pl.CPLEX_CMD(
                path="/Users/jimborg/Applications/CPLEX_Studio2212/cplex/bin/arm64_osx/cplex",
                timeLimit=time_limit_seconds,
                msg=False,
            )
        )

    elapsed_time_seconds = time.perf_counter() - solve_started_at
    raw_status = pl.LpStatus.get(cluster.problem.status, "Unknown")
    objective_value = None
    try:
        objective_value = float(pl.value(cluster.problem.objective))
    except Exception:
        objective_value = None

    incumbent_value = objective_value
    if raw_status == "Optimal" and incumbent_value is not None:
        best_bound = incumbent_value
        absolute_gap = 0.0
        relative_gap = 0.0
        relative_gap_percent = 0.0
    else:
        best_bound = None
        absolute_gap = None
        relative_gap = None
        relative_gap_percent = None

    if solver_backend == "glpk":
        progress_summary = _parse_glpk_progress(log_path, objective_value, raw_status)
        if best_bound is None:
            best_bound = progress_summary["progress_events"][-1]["best_bound"] if progress_summary["progress_events"] else None
        if raw_status != "Optimal":
            absolute_gap = compute_absolute_gap(incumbent_value, best_bound)
            relative_gap = compute_relative_gap(incumbent_value, best_bound)
            relative_gap_percent = compute_relative_gap_percent(incumbent_value, best_bound)
    elif raw_status != "Optimal":
        absolute_gap = compute_absolute_gap(incumbent_value, best_bound)
        relative_gap = compute_relative_gap(incumbent_value, best_bound)
        relative_gap_percent = compute_relative_gap_percent(incumbent_value, best_bound)

    return {
        "status_code": int(cluster.problem.status),
        "raw_status": raw_status,
        "normalized_status": normalize_solver_status(raw_status),
        "objective_value": objective_value,
        "incumbent_value": incumbent_value,
        "best_bound": best_bound,
        "absolute_gap": absolute_gap,
        "relative_gap": relative_gap,
        "relative_gap_percent": relative_gap_percent,
        "time_limit_seconds": time_limit_seconds,
        "elapsed_time_seconds": float(elapsed_time_seconds),
        "termination_reason": infer_termination_reason(raw_status, time_limit_seconds),
        "first_feasible_time_seconds": progress_summary.get("first_feasible_time_seconds"),
        "first_optimality_gap_percent": progress_summary.get("first_optimality_gap_percent"),
        "explored_bnb_nodes": progress_summary.get("explored_bnb_nodes"),
        "active_bnb_nodes": progress_summary.get("active_bnb_nodes"),
        "optimality_proven": progress_summary.get("optimality_proven", raw_status == "Optimal"),
        "progress_events": progress_summary.get("progress_events", []),
        "solver_log_path": progress_summary.get("log_path", log_path if solver_backend == "glpk" else ""),
        "solver_seed": int(solver_seed),
    }
