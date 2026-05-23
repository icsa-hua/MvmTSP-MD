from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Union

import networkx as nx
import numpy as np

from dummy_app.models.central_hubs import CentralHub
from dummy_app.models.genetic_algorithm import GASolution, get_weights
from dummy_app.tools.common import add_virtual_nodes
from dummy_app.tools.graphs import is_eulerian_digraph


def normalize_warm_start_mode(mode: Any) -> str:
    normalized = str(mode or "none").strip().lower()
    if normalized in {"", "none", "off", "disabled", "false"}:
        return "none"
    if normalized in {"ga", "ga_only", "ga_plus_time_windows", "genetic_algorithm"}:
        return "ga"
    if normalized in {"alns", "alns_only", "alns_plus_time_windows"}:
        return "alns"
    return normalized


def prepare_cluster_cost_bundle(problem_builder: Any, extraction: Dict[str, Union[List[str], np.ndarray]], depot: int, employed_agents: List[int]):
    try:
        area_ids = np.array(extraction["area_ids"]).squeeze()
        dists = extraction["dists"]
        ees = extraction["ees"]
        travel_times = extraction["travel_times"]
    except KeyError as exc:
        raise ValueError(f"KeyError: {exc}") from exc

    cost_d = dict(zip(area_ids, dists))
    cost_e = dict(zip(area_ids, ees))
    cost_t = dict(zip(area_ids, travel_times))
    nodes_dict = {i: int(node) for i, node in enumerate(area_ids)}
    cost_bundle = {"distance": cost_d, "energy": cost_e, "travel_time": cost_t}

    graph = GASolution.create_model_graph(
        cost=cost_bundle,
        nodes=nodes_dict,
        weights=get_weights(getattr(problem_builder, "objective_weights", None)),
    )

    try:
        is_eulerian_digraph(graph)
        if not is_eulerian_digraph(graph):
            graph = nx.eulerian_circuit(graph)
    except Exception as exc:
        raise ValueError("Graph is not eulerian") from exc

    hub = CentralHub()
    bridge_nodes = hub.get_bridge_nodes(
        graph=graph,
        cluster_nodes=list(nodes_dict.keys()),
        cost_dist=cost_bundle["distance"],
        nodes_dict=nodes_dict,
        n_agents=len(employed_agents),
    )

    bridge_nodes = [nodes_dict[bridge_nodes[i]] for i in range(len(bridge_nodes))]
    reverse_nodes = {v: k for k, v in nodes_dict.items()}
    virtual_nodes = defaultdict(int)

    required_visits = []
    for i in nodes_dict:
        if nodes_dict[i] in bridge_nodes:
            allowed_visits = hub.number_allowed_visits[i]
        else:
            allowed_visits = 1
        required_visits.append(allowed_visits)

    if not all(required_visit == 1 for required_visit in required_visits) or len(bridge_nodes) > 1:
        for node in bridge_nodes:
            number_of_virtual_nodes = required_visits[reverse_nodes[node]]
            constant_length = len(cost_bundle["distance"][node])
            for clone_offset in range(number_of_virtual_nodes):
                virtual_nodes[clone_offset + constant_length] = node

        count = len(nodes_dict)
        cost_bundle = add_virtual_nodes(cost_bundle=cost_bundle, clones=virtual_nodes, add_epsilon=True)
        for virtual_node in virtual_nodes:
            nodes_dict[count] = virtual_node
            count += 1

    return cost_bundle, virtual_nodes, bridge_nodes, nodes_dict, {}
