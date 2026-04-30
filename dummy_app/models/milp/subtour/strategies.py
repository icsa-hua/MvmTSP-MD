from __future__ import annotations

from dummy_app.models.milp.subtour.dfj import apply_dfj_constraints
from dummy_app.models.milp.subtour.flow import apply_flow_constraints
from dummy_app.models.milp.subtour.mtz import apply_mtz_constraints


def apply_subtour_constraints(cluster, agents, nodes, depot_ind, strategy: str) -> None:
    strategy = (strategy or "mtz").lower()
    if strategy == "mtz":
        apply_mtz_constraints(cluster, agents, nodes, depot_ind)
        return
    if strategy == "dfj":
        apply_dfj_constraints(cluster, agents, nodes, depot_ind)
        return
    if strategy == "flow":
        apply_flow_constraints(cluster, agents, nodes, depot_ind)
        return
    raise ValueError(f"Unsupported subtour strategy '{strategy}'")
