from __future__ import annotations

import random
from typing import Any, Dict, List

import numpy as np

from dummy_app.models.base import OptimizationModel
from dummy_app.models.genetic_algorithm import GASolution
from dummy_app.models.heuristics.common import (
    apply_two_opt_local_search,
    build_partition_diagnostics,
    build_static_partitions,
    greedy_repair_unassigned_nodes,
    is_route_feasible,
    list_service_nodes,
    seeded_random,
    task_sequences_to_solution,
)


def _build_ga_nodes(cluster: Any, route_nodes: List[int]) -> Dict[int, int]:
    ga_nodes = {0: int(cluster.depot_id)}
    for offset, node_id in enumerate(route_nodes, start=1):
        ga_nodes[offset] = int(node_id)
    return ga_nodes


def _run_ga_route(
    builder: Any,
    cluster: Any,
    partition_nodes: List[int],
    seed: int,
) -> tuple[List[int], List[int]]:
    if not partition_nodes:
        return [], []

    random.seed(int(seed))
    np.random.seed(int(seed))

    ga_nodes = _build_ga_nodes(cluster, partition_nodes)
    ga_solver = GASolution(
        population=200,
        generations=int(getattr(builder, "ga_generations", 100)),
        nodes_dict=ga_nodes,
        depot=int(cluster.depot_id),
    )

    route_with_depot, _ = ga_solver.run(
        crossover_rate=0.7,
        mutation_rate=0.05,
        cost=cluster.cost,
        enable_indi_fitness=True,
        verbose=False,
    )

    ordered_nodes = [int(node_id) for node_id in route_with_depot if int(node_id) != int(cluster.depot_id)]
    feasible_prefix: List[int] = []
    leftovers: List[int] = []
    for node_id in ordered_nodes:
        candidate_route = feasible_prefix + [int(node_id)]
        if is_route_feasible(builder, cluster, candidate_route):
            feasible_prefix = candidate_route
        else:
            leftovers.append(int(node_id))

    feasible_prefix = apply_two_opt_local_search(builder, cluster, feasible_prefix)
    return feasible_prefix, leftovers


def solve_genetic_algorithm_baseline(cluster: Any, builder: Any) -> Any:
    base_seed = int(getattr(builder, "random_seed", 42))
    rng = seeded_random(base_seed + int(cluster.id))
    partition_map = build_static_partitions(builder, cluster, cluster.employed_agents, list_service_nodes(cluster))
    for agent_id, nodes in partition_map.items():
        shuffled_nodes = list(nodes)
        rng.shuffle(shuffled_nodes)
        partition_map[agent_id] = shuffled_nodes

    task_sequences: Dict[int, List[int]] = {int(agent_id): [] for agent_id in cluster.employed_agents}
    unassigned_nodes: List[int] = []

    for agent_id in sorted(cluster.employed_agents):
        route_nodes, leftovers = _run_ga_route(
            builder=builder,
            cluster=cluster,
            partition_nodes=partition_map.get(int(agent_id), []),
            seed=base_seed + int(cluster.id) + int(agent_id),
        )
        task_sequences[int(agent_id)] = route_nodes
        unassigned_nodes.extend(leftovers)

    unresolved_nodes = greedy_repair_unassigned_nodes(builder, cluster, task_sequences, unassigned_nodes)
    for agent_id, route_nodes in list(task_sequences.items()):
        task_sequences[int(agent_id)] = apply_two_opt_local_search(builder, cluster, route_nodes)

    return task_sequences_to_solution(
        builder=builder,
        cluster=cluster,
        task_sequences=task_sequences,
        uncovered_task_nodes=unresolved_nodes,
        strategy_name="genetic_algorithm_baseline",
        diagnostics={
            "partition_by_agent": build_partition_diagnostics(cluster, partition_map),
            "ga_unassigned_after_partition": sorted(set(int(node_id) for node_id in unassigned_nodes)),
        },
    )


class GeneticAlgorithmOptimizationModel(OptimizationModel):
    model_name = "ga"

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def solve_cluster(self, instance, cluster_input, request):
        return self.runtime.solve_cluster_instance_heuristic(
            instance=instance,
            cluster_input=cluster_input,
            request=request,
            heuristic_solver=solve_genetic_algorithm_baseline,
            heuristic_name=self.model_name,
        )
