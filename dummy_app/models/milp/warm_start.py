from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np

from collections import defaultdict

from dummy_app.models.central_hubs import CentralHub
from dummy_app.models.genetic_algorithm import GASolution, get_weights
from dummy_app.tools.common import add_virtual_nodes
from dummy_app.tools.graphs import is_eulerian_digraph


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
    initial_population = {}
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

    ga_nodes = nodes_dict.copy()
    for bridge_node in bridge_nodes:
        ga_nodes.pop(reverse_nodes[bridge_node])

    if hasattr(problem_builder, "enable_ga") and problem_builder.enable_ga:
        for agent in employed_agents:
            solution_path, solution_cost = problem_builder.call_genetic_algorithm(
                nodes_dict=ga_nodes,
                cost=cost_bundle,
                depot=depot,
                verbose=False,
                generations=getattr(problem_builder, "ga_generations", 100),
            )
            initial_population[agent] = (solution_path, solution_cost)

    return cost_bundle, virtual_nodes, bridge_nodes, nodes_dict, initial_population
