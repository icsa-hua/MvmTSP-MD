from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ClusterSolveResult:
    cluster_id: int
    raw_status: str
    normalized_status: str
    status_code: int
    objective_value: Optional[float] = None
    incumbent_value: Optional[float] = None
    best_bound: Optional[float] = None
    absolute_gap: Optional[float] = None
    relative_gap: Optional[float] = None
    elapsed_time_seconds: float = 0.0
    time_limit_seconds: Optional[float] = None
    termination_reason: str = "unknown"
    agent_paths: Dict[str, Any] = field(default_factory=dict)
    agent_metrics: Dict[str, Any] = field(default_factory=dict)
    cluster_metrics: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRunResult:
    run_id: str
    instance_id: str
    model_name: str
    raw_status: str
    normalized_status: str
    objective_value: Optional[float] = None
    summary_objective_value: Optional[float] = None
    incumbent_value: Optional[float] = None
    best_bound: Optional[float] = None
    absolute_gap: Optional[float] = None
    relative_gap: Optional[float] = None
    elapsed_time_seconds: float = 0.0
    time_limit_seconds: Optional[float] = None
    termination_reason: str = "unknown"
    summary: Dict[str, Any] = field(default_factory=dict)
    cluster_results: List[ClusterSolveResult] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
