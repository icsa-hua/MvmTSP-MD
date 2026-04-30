from __future__ import annotations

from collections import defaultdict
from typing import Any

import math
import pulp as pl

from dummy_app.tools.logger import logger


def get_node_visits(cluster, builder: Any, node, dc) -> None:
    if dc[node] not in builder.visits_per_nodes:
        if dc[node] in cluster.virtual_nodes:
            builder.visits_per_nodes[cluster.virtual_nodes[dc[node]]] = 1
        else:
            builder.visits_per_nodes[dc[node]] = 1
    else:
        if dc[node] in cluster.virtual_nodes:
            builder.visits_per_nodes[cluster.virtual_nodes[dc[node]]] += 1
        else:
            builder.visits_per_nodes[dc[node]] += 1


def extract_cluster_solution(cluster, builder: Any):
    logger.debug(f"Cluster Time Frame is {cluster.timeframe}")
    if pl.LpStatus[cluster.problem.status] != "Optimal":
        logger.info("Problem did not terminate as Optimal; extracting structured failure state.")
        raise RuntimeError(f"Cluster {cluster.id} ended with status {pl.LpStatus[cluster.problem.status]}")

    V_nodes = list(cluster.nodes_dict.keys())
    reverse_dict = {v: k for k, v in cluster.nodes_dict.items()}
    depot_ind = reverse_dict[cluster.depot_id]
    NODES = [node for node in V_nodes if node != depot_ind]
    detailed_log = defaultdict(list)
    dc = cluster.nodes_dict

    for k in cluster.employed_agents:
        start_node = -1
        for j in NODES:
            if cluster.x[depot_ind, j, k].varValue > 0.5:
                start_node = j
                break

        if start_node == -1:
            continue

        real_start_node = cluster.virtual_nodes.get(dc[start_node], dc[start_node])
        arrival_at_start_node = cluster.t[start_node, k].varValue
        for t_step in range(round(0.0), round(arrival_at_start_node)):
            detailed_log[k].append((dc[depot_ind], real_start_node, t_step))

        get_node_visits(cluster, builder, start_node, dc)
        current_node = start_node
        while current_node != depot_ind:
            real_current_node = cluster.virtual_nodes.get(dc[current_node], dc[current_node])
            next_node_in_path = -1
            for next_node in V_nodes:
                if cluster.x[current_node, next_node, k].varValue > 0.5:
                    next_node_in_path = next_node
                    break

            if next_node_in_path == -1:
                logger.error(
                    f"Warning: Path broken for agent {k} at node {current_node}. "
                    f"Could not find a next step."
                )
                break

            real_next_node = cluster.virtual_nodes.get(dc[next_node_in_path], dc[next_node_in_path])
            arrival_at_current = cluster.t[current_node, k].varValue
            departure_from_current = arrival_at_current + builder.coverage_time

            for t_step in range(round(arrival_at_current), round(departure_from_current)):
                detailed_log[k].append((real_current_node, real_current_node, t_step))

            start_t_move = round(departure_from_current)
            if next_node_in_path == depot_ind:
                arrival_at_next = cluster.return_step[k].varValue
            else:
                arrival_at_next = cluster.t[next_node_in_path, k].varValue

            end_t_move = round(arrival_at_next)
            if start_t_move >= end_t_move and arrival_at_next >= departure_from_current:
                end_t_move = start_t_move + 1

            for t_step in range(start_t_move, end_t_move):
                detailed_log[k].append((real_current_node, real_next_node, t_step))

            get_node_visits(cluster, builder, next_node_in_path, dc)
            current_node = next_node_in_path

    unique_nodes_among_paths = set()
    for _, events in detailed_log.items():
        events.sort(key=lambda event: event[2])
        for source, target, _ in events:
            unique_nodes_among_paths.add(source)
            unique_nodes_among_paths.add(target)

    logger.debug(f"Solutions created for {len(cluster.employed_agents)} agents")
    builder.num_constraints += len(cluster.problem.constraints)
    builder.variables_count += len(cluster.problem.variables())
    builder.global_nodes_visited += len(unique_nodes_among_paths)
    builder.validate_paths(paths=detailed_log, nodes_dict=cluster.nodes_dict, cluster=cluster)
    return detailed_log
