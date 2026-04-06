from typing import Dict, Optional, Tuple


def compute_reward(run_metrics: Dict[str, float], baseline_objective: Optional[float]) -> Tuple[float, Dict[str, float]]:
    solve_time = float(run_metrics.get("solve_time_seconds", 0.0))
    time_limit = max(float(run_metrics.get("time_limit_seconds", 1.0)), 1.0)
    timeout_flag = float(run_metrics.get("timeout_flag", 0.0))
    infeasible_flag = 0.0 if run_metrics.get("feasible_flag", 0.0) else 1.0
    objective_value = float(run_metrics.get("objective_value", 0.0))
    full_coverage_flag = 1.0 if float(run_metrics.get("node_coverage_ratio", 0.0)) >= 0.999 else 0.0
    normalized_solve_time = min(solve_time / time_limit, 5.0)
    normalized_objective_gap = 0.0
    if baseline_objective is not None:
        normalized_objective_gap = (objective_value - baseline_objective) / max(abs(baseline_objective), 1e-6)

    reward = (
        -1.0 * normalized_solve_time
        -5.0 * timeout_flag
        -5.0 * infeasible_flag
        -0.5 * normalized_objective_gap
        +0.2 * full_coverage_flag
    )

    reward_terms = {
        "normalized_solve_time": normalized_solve_time,
        "timeout_penalty": -5.0 * timeout_flag,
        "infeasible_penalty": -5.0 * infeasible_flag,
        "normalized_objective_gap": normalized_objective_gap,
        "coverage_bonus": 0.2 * full_coverage_flag,
    }
    return reward, reward_terms
