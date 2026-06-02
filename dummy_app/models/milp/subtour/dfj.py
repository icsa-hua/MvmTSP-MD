from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import pulp as pl


SubtourCut = Tuple[int, Tuple[int, ...]]


def _sorted_unique_nodes(nodes: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted({int(node) for node in nodes}))


def _iter_static_dfj_subsets(nodes: Sequence[int]) -> Iterable[Tuple[int, ...]]:
    ordered_nodes = _sorted_unique_nodes(nodes)
    for subset_size in range(2, len(ordered_nodes)):
        for subset in combinations(ordered_nodes, subset_size):
            yield subset


def add_dfj_cuts(cluster, subtour_cuts: Sequence[SubtourCut]) -> int:
    if not hasattr(cluster, "dfj_subtour_cuts"):
        cluster.dfj_subtour_cuts = []
    if not hasattr(cluster, "_dfj_cut_registry"):
        cluster._dfj_cut_registry = set()
    if not hasattr(cluster, "_dfj_cut_counter"):
        cluster._dfj_cut_counter = 0

    cuts_added = 0
    for agent_id, raw_subset in subtour_cuts:
        subset = _sorted_unique_nodes(raw_subset)
        if len(subset) <= 1:
            continue

        cut_key = (int(agent_id), subset)
        if cut_key in cluster._dfj_cut_registry:
            continue

        cluster.problem += (
            pl.lpSum(
                cluster.x[i, j, int(agent_id)]
                for i in subset
                for j in subset
                if i != j and (i, j, int(agent_id)) in cluster.x
            )
            <= len(subset) - 1,
            f"dfj_cut_k{int(agent_id)}_{cluster._dfj_cut_counter}",
        )
        cluster._dfj_cut_registry.add(cut_key)
        cluster.dfj_subtour_cuts.append(cut_key)
        cluster._dfj_cut_counter += 1
        cuts_added += 1

    return cuts_added


def _infer_depot_index(cluster, depot_ind: int | None) -> int:
    if depot_ind is not None:
        return int(depot_ind)
    nodes_dict = dict(getattr(cluster, "nodes_dict", {}))
    depot_id = getattr(cluster, "depot_id", None)
    for node_index, node_id in nodes_dict.items():
        if int(node_id) == int(depot_id):
            return int(node_index)
    raise ValueError("Unable to infer the depot index for DFJ subtour detection.")


def find_disconnected_subtours(cluster, depot_ind: int | None = None, threshold: float = 0.5) -> List[SubtourCut]:
    subtours: List[SubtourCut] = []
    resolved_depot_ind = _infer_depot_index(cluster, depot_ind)
    all_nodes = tuple(int(node) for node in getattr(cluster, "V_nodes", []))

    for agent_id in getattr(cluster, "employed_agents", []):
        graph = nx.DiGraph()
        if all_nodes:
            graph.add_nodes_from(all_nodes)

        for (source, target, assigned_agent), variable in cluster.x.items():
            if int(assigned_agent) != int(agent_id) or int(source) == int(target):
                continue
            value = getattr(variable, "varValue", None)
            if value is None or float(value) <= threshold:
                continue
            graph.add_edge(int(source), int(target))

        if graph.number_of_edges() == 0:
            continue

        for component in nx.weakly_connected_components(graph):
            if int(resolved_depot_ind) in component or len(component) <= 1:
                continue
            subtours.append((int(agent_id), _sorted_unique_nodes(component)))

    subtours.sort(key=lambda item: (item[0], item[1]))
    return subtours


def apply_dfj_constraints(
    cluster,
    agents,
    nodes,
    depot_ind,
    subtour_cuts: Sequence[SubtourCut] | None = None,
    *,
    enumerate_all_subsets: bool = False,
    max_static_nodes: int = 8,
) -> None:
    cuts_to_add: List[SubtourCut] = []

    if subtour_cuts:
        cuts_to_add.extend((int(agent_id), _sorted_unique_nodes(subset)) for agent_id, subset in subtour_cuts)

    if enumerate_all_subsets:
        if len(nodes) > max_static_nodes:
            raise ValueError(
                "Static DFJ subset enumeration is disabled for larger instances because it grows exponentially."
            )
        for agent_id in agents:
            for subset in _iter_static_dfj_subsets(nodes):
                cuts_to_add.append((int(agent_id), subset))

    add_dfj_cuts(cluster, cuts_to_add)
