from __future__ import annotations

from typing import Any, Dict

import pulp as pl

from dummy_app.core.statuses import compute_absolute_gap, compute_relative_gap, infer_termination_reason, normalize_solver_status


def solve_cluster_problem(cluster: Any, time_limit_seconds=None, solver_backend: str = "glpk") -> Dict[str, Any]:
    if solver_backend != "glpk":
        raise ValueError(f"Unsupported solver backend '{solver_backend}'. Only 'glpk' is currently implemented.")

    cluster.problem.solve(
        pl.GLPK_CMD(
            timeLimit=time_limit_seconds,
            msg=False,
            options=["--mipgap", "0.0", "--seed", "42"],
        )
    )
    raw_status = pl.LpStatus.get(cluster.problem.status, "Unknown")
    objective_value = None
    try:
        objective_value = float(pl.value(cluster.problem.objective))
    except Exception:
        objective_value = None

    incumbent_value = objective_value
    best_bound = objective_value if raw_status == "Optimal" else None

    return {
        "status_code": int(cluster.problem.status),
        "raw_status": raw_status,
        "normalized_status": normalize_solver_status(raw_status),
        "objective_value": objective_value,
        "incumbent_value": incumbent_value,
        "best_bound": best_bound,
        "absolute_gap": compute_absolute_gap(incumbent_value, best_bound),
        "relative_gap": compute_relative_gap(incumbent_value, best_bound),
        "time_limit_seconds": time_limit_seconds,
        "termination_reason": infer_termination_reason(raw_status, time_limit_seconds),
    }
