from dataclasses import dataclass
from typing import Dict, List


OBJECTIVE_WEIGHT_PROFILES: Dict[str, Dict[str, float]] = {
    "energy_heavy": {"energy": 0.6, "distance": 0.2, "travel_time": 0.2},
    "balanced": {"energy": 0.33, "distance": 0.33, "travel_time": 0.34},
    "time_heavy": {"energy": 0.2, "distance": 0.2, "travel_time": 0.6},
    "distance_heavy": {"energy": 0.2, "distance": 0.6, "travel_time": 0.2},
}


CLUSTERING_BALANCE_PROFILES: Dict[str, Dict[str, float]] = {
    "balanced_priority": {"distance": 1.0, "energy": 1.0, "travel_time": 1.0},
    "energy_priority": {"distance": 0.8, "energy": 1.4, "travel_time": 0.8},
    "distance_priority": {"distance": 1.4, "energy": 0.8, "travel_time": 0.8},
}


@dataclass(frozen=True)
class SolverAction:
    action_id: str
    solver_mode: str
    stage_solution: int
    ga_generation_bucket: str
    ga_generations: int
    time_limit_bucket: str
    time_limit_seconds: int
    warm_start_mode: str
    objective_weight_profile: str
    clustering_balance_profile: str

    def to_runtime_config(self) -> Dict[str, object]:
        warm_start_mode = self.warm_start_mode
        return {
            "solver_mode": self.solver_mode,
            "stage_solution": self.stage_solution,
            "ga_generation_bucket": self.ga_generation_bucket,
            "ga_generations": self.ga_generations,
            "time_limit_bucket": self.time_limit_bucket,
            "time_limit_seconds": self.time_limit_seconds,
            "warm_start_mode": warm_start_mode,
            "objective_weight_profile": self.objective_weight_profile,
            "objective_weights": OBJECTIVE_WEIGHT_PROFILES[self.objective_weight_profile],
            "clustering_balance_profile": self.clustering_balance_profile,
            "clustering_feature_weights": CLUSTERING_BALANCE_PROFILES[self.clustering_balance_profile],
            "enable_ga": warm_start_mode != "none",
        }


def build_action_catalog() -> List[SolverAction]:
    return [
        SolverAction("A0", "single-stage", 1, "low", 20, "short", 60, "none", "balanced", "balanced_priority"),
        SolverAction("A1", "single-stage", 1, "medium", 50, "medium", 300, "ga_only", "balanced", "balanced_priority"),
        SolverAction("A2", "single-stage", 1, "high", 100, "medium", 300, "ga_plus_time_windows", "energy_heavy", "energy_priority"),
        SolverAction("A3", "single-stage", 1, "medium", 50, "long", 900, "ga_plus_time_windows", "time_heavy", "balanced_priority"),
        SolverAction("A4", "two-stage", 2, "low", 20, "medium", 300, "ga_only", "balanced", "balanced_priority"),
        SolverAction("A5", "two-stage", 2, "medium", 50, "long", 900, "ga_plus_time_windows", "energy_heavy", "energy_priority"),
        SolverAction("A6", "two-stage", 2, "high", 100, "long", 900, "ga_plus_time_windows", "time_heavy", "balanced_priority"),
        SolverAction("A7", "single-stage", 1, "low", 20, "short", 60, "ga_only", "distance_heavy", "distance_priority"),
    ]
