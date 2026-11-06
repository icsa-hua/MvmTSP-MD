from __future__ import annotations

from typing import Any, Dict, List

from dummy_app.models.base import OptimizationModel
from dummy_app.models.heuristics.common import (
    HeuristicClusterSolution,
    build_data_reward_map,
    build_detailed_path,
    build_partition_diagnostics,
    build_static_partitions,
    build_uncovered_physical_nodes,
    is_candidate_feasible,
    list_service_nodes,
    resolve_task_node,
    select_nearest_candidate,
)


def _solve_partition_route(cluster: Any, builder: Any, candidate_nodes: List[int]) -> tuple[List[int], List[int]]:
    route_nodes: List[int] = []
    remaining_nodes = set(int(node_id) for node_id in candidate_nodes)
    current_node = int(cluster.depot_id)
    energy_used = 0.0

    while remaining_nodes:
        feasible_candidates = [
            node_id
            for node_id in remaining_nodes
            if is_candidate_feasible(builder, cluster, current_node, node_id, energy_used)
        ]
        if not feasible_candidates:
            break

        next_node = select_nearest_candidate(builder, cluster, current_node, feasible_candidates)
        if next_node is None:
            break

        route_nodes.append(int(next_node))
        energy_used += (
            builder.move_energy[resolve_task_node(cluster, current_node)][resolve_task_node(cluster, next_node)]
            + float(builder.average_coverage_energy) * float(builder.coverage_time)
        )
        current_node = int(next_node)
        remaining_nodes.remove(int(next_node))

    return route_nodes, sorted(int(node_id) for node_id in remaining_nodes)


def solve_static_partition_greedy_nn(cluster: Any, builder: Any) -> HeuristicClusterSolution:
    service_nodes = list_service_nodes(cluster)
    reward_by_task = build_data_reward_map(builder, cluster)
    partition_map = build_static_partitions(builder, cluster, cluster.employed_agents, service_nodes)

    agent_paths: Dict[int, List[tuple[int, int, int]]] = {}
    task_sequences: Dict[int, List[int]] = {}
    agent_finish_times: Dict[int, float] = {}
    uncovered_task_nodes: List[int] = []
    covered_nodes: List[int] = []
    total_data_transfer = 0.0

    for agent_id in sorted(cluster.employed_agents):
        route_nodes, leftovers = _solve_partition_route(cluster, builder, partition_map.get(int(agent_id), []))
        covered_nodes.extend(resolve_task_node(cluster, node_id) for node_id in route_nodes)
        uncovered_task_nodes.extend(leftovers)
        total_data_transfer += sum(reward_by_task.get(int(node_id), 0.0) for node_id in route_nodes)
        task_sequences[int(agent_id)] = list(route_nodes)

        path, finish_time = build_detailed_path(builder, cluster, route_nodes)
        agent_paths[int(agent_id)] = path
        agent_finish_times[int(agent_id)] = finish_time

    makespan = max(agent_finish_times.values(), default=0.0)
    unique_covered_nodes = sorted(set(int(node_id) for node_id in covered_nodes))
    unique_uncovered_task_nodes = sorted(set(int(node_id) for node_id in uncovered_task_nodes))
    raw_status = "Feasible" if unique_covered_nodes else "Infeasible"

    return HeuristicClusterSolution(
        raw_status=raw_status,
        status_code=1 if unique_covered_nodes else -1,
        agent_paths=agent_paths,
        task_sequences=task_sequences,
        agent_finish_times=agent_finish_times,
        total_data_transfer=float(total_data_transfer),
        makespan=float(makespan),
        uncovered_task_nodes=unique_uncovered_task_nodes,
        diagnostics={
            "heuristic_strategy": "static_partition_greedy_nn",
            "covered_nodes": unique_covered_nodes,
            "partition_by_agent": build_partition_diagnostics(cluster, partition_map),
            "uncovered_physical_nodes": build_uncovered_physical_nodes(cluster, unique_uncovered_task_nodes),
        },
    )


class StaticPartitionGreedyNNOptimizationModel(OptimizationModel):
    model_name = "greedy_partition_nn"

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def solve_cluster(self, instance, cluster_input, request):
        return self.runtime.solve_cluster_instance_heuristic(
            instance=instance,
            cluster_input=cluster_input,
            request=request,
            heuristic_solver=solve_static_partition_greedy_nn,
            heuristic_name=self.model_name,
        )
