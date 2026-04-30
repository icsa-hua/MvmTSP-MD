from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dummy_app.core.schemas.instance import ProblemInstance
from dummy_app.core.schemas.request import ModelRunRequest
from dummy_app.core.schemas.result import ClusterSolveResult


class OptimizationModel(ABC):
    model_name = "base"

    @abstractmethod
    def solve_cluster(self, instance: ProblemInstance, cluster_input: Any, request: ModelRunRequest) -> ClusterSolveResult:
        raise NotImplementedError
