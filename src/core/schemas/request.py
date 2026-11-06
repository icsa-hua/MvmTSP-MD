from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelRunRequest:
    model_name: str
    solver_backend: str = "glpk"
    subtour_mode: str = "mtz"
    subtour_strategy: str = "mtz"
    scenario_constraint_set: str = "default"
    objective_strategy: str = "legacy_stage"
    warm_start_strategy: str = "ga"
    solver_time_limit_seconds: Optional[int] = None
    objective_weights: Dict[str, float] = field(default_factory=dict)
    clustering_feature_weights: Dict[str, float] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
