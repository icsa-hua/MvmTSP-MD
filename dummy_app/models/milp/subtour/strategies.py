from __future__ import annotations

from dummy_app.models.milp.subtour.dfj import apply_dfj_constraints
from dummy_app.models.milp.subtour.flow import apply_flow_constraints
from dummy_app.models.milp.subtour.mtz import apply_mtz_constraints


def normalize_subtour_mode(mode: str | None) -> str:
    normalized = str(mode or "mtz").strip().lower()
    if normalized in {"", "mtz"}:
        return "mtz"
    if normalized in {"dfj", "dfj_iter", "iterative_dfj"}:
        return "dfj_iter"
    if normalized in {"flow", "single_commodity_flow", "scf"}:
        return "flow"
    if normalized in {"none", "off", "disabled"}:
        return "none"
    return normalized


def apply_subtour_constraints(cluster, agents, nodes, depot_ind, mode: str, subtour_cuts=None) -> None:
    subtour_mode = normalize_subtour_mode(mode)
    if subtour_mode == "none":
        return
    if subtour_mode == "mtz":
        apply_mtz_constraints(cluster, agents, nodes, depot_ind)
        return
    if subtour_mode == "dfj_iter":
        apply_dfj_constraints(
            cluster,
            agents,
            nodes,
            depot_ind,
            subtour_cuts=subtour_cuts if subtour_cuts is not None else getattr(cluster, "dfj_subtour_cuts", []),
        )
        return
    if subtour_mode == "flow":
        apply_flow_constraints(cluster, agents, nodes, depot_ind)
        return
    raise ValueError(f"Unsupported subtour mode '{mode}'")
