from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class PreparedClusterInput:
    cluster_id: int
    cluster_frame: pd.DataFrame
    assigned_agents: List[int]
    depot_id: int
    priority_rank: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProblemInstance:
    instance_id: str
    model_name: str
    scenario: str
    objective_function: str
    stage_solution: int
    environment_type: str
    number_of_agents: int
    number_of_areas: int
    number_of_users: int
    depots: List[int]
    user_points: Dict[int, List[Any]]
    prepared_data: pd.DataFrame
    distance_matrix: Any
    raw_cost_matrices: Dict[str, Any]
    normalized_cost_columns: Dict[str, List[str]]
    objective_weights: Dict[str, float]
    clustering_feature_weights: Dict[str, float]
    prepared_clusters: List[PreparedClusterInput] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
