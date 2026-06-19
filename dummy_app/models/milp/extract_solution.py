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
    solve_metadata = dict(getattr(cluster, "solve_metadata", {}))
    feasible_solution_found = bool(
        solve_metadata.get("accepted_solution")
        or solve_metadata.get("feasible_solution_found")
        or solve_metadata.get("incumbent_value") is not None
        or pl.LpStatus[cluster.problem.status] == "Optimal"
    )
    if not feasible_solution_found:
        logger.info("No feasible incumbent is available for route extraction.")
        raise RuntimeError(
            f"Cluster {cluster.id} ended without a feasible solution "
            f"(status={solve_metadata.get('raw_status', pl.LpStatus[cluster.problem.status])})"
        )

    V_nodes = list(cluster.nodes_dict.keys())
    reverse_dict = {v: k for k, v in cluster.nodes_dict.items()}
    depot_ind = reverse_dict[cluster.depot_id]
    NODES = [node for node in V_nodes if node != depot_ind]
    detailed_log = defaultdict(list)
    dc = cluster.nodes_dict

    for k in cluster.employed_agents:
        start_node = -1
        start_offset = float(getattr(cluster.start_step[k], "varValue", 0.0) or 0.0)
        start_candidates = [
            (float(cluster.x[depot_ind, j, k].varValue or 0.0), j)
            for j in NODES
            if float(cluster.x[depot_ind, j, k].varValue or 0.0) > 0.5
        ]
        if start_candidates:
            _, start_node = max(start_candidates)

        if start_node == -1:
            continue

        real_start_node = cluster.virtual_nodes.get(dc[start_node], dc[start_node])
        arrival_at_start_node = float(cluster.t[start_node, k].varValue) - start_offset
        for t_step in range(round(0.0), round(arrival_at_start_node)):
            detailed_log[k].append((dc[depot_ind], real_start_node, t_step))

        get_node_visits(cluster, builder, start_node, dc)
        current_node = start_node
        traversed_arcs = set()
        while current_node != depot_ind:
            real_current_node = cluster.virtual_nodes.get(dc[current_node], dc[current_node])
            successor_candidates = [
                (float(cluster.x[current_node, next_node, k].varValue or 0.0), next_node)
                for next_node in V_nodes
                if next_node != current_node
                and float(cluster.x[current_node, next_node, k].varValue or 0.0) > 0.5
            ]
            next_node_in_path = max(successor_candidates)[1] if successor_candidates else -1

            if next_node_in_path == -1:
                logger.error(
                    f"Warning: Path broken for agent {k} at node {current_node}. "
                    f"Could not find a next step."
                )
                break
            selected_arc = (current_node, next_node_in_path)
            if selected_arc in traversed_arcs:
                raise RuntimeError(
                    f"Cycle detected while extracting cluster {cluster.id}, agent {k}, arc {selected_arc}"
                )
            traversed_arcs.add(selected_arc)

            real_next_node = cluster.virtual_nodes.get(dc[next_node_in_path], dc[next_node_in_path])
            arrival_at_current = float(cluster.t[current_node, k].varValue) - start_offset
            departure_from_current = arrival_at_current + builder.coverage_time

            for t_step in range(round(arrival_at_current), round(departure_from_current)):
                detailed_log[k].append((real_current_node, real_current_node, t_step))

            start_t_move = round(departure_from_current)
            if next_node_in_path == depot_ind:
                arrival_at_next = float(cluster.return_step[k].varValue) - start_offset
            else:
                arrival_at_next = float(cluster.t[next_node_in_path, k].varValue) - start_offset

            # A selected discrete arc must always appear in the rendered route.
            # Solver tolerances and rounded continuous times can otherwise erase
            # the final return arc and create a false "does not return" failure.
            end_t_move = max(round(arrival_at_next), start_t_move + 1)

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
    builder.global_nodes_visited += len(unique_nodes_among_paths)
    builder.validate_paths(paths=detailed_log, nodes_dict=cluster.nodes_dict, cluster=cluster)
    return detailed_log
