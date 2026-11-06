from __future__ import annotations

from typing import Any

from dummy_app.models.base import OptimizationModel


class MILPOptimizationModel(OptimizationModel):
    model_name = "milp"

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def solve_cluster(self, instance, cluster_input, request):
        return self.runtime.solve_cluster_instance(instance, cluster_input, request)
