from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass
class HeuristicClusterSolution:
    raw_status: str
    status_code: int
    agent_paths: Dict[int, List[Tuple[int, int, int]]] = field(default_factory=dict)
    task_sequences: Dict[int, List[int]] = field(default_factory=dict)
    agent_finish_times: Dict[int, float] = field(default_factory=dict)
    total_data_transfer: float = 0.0
    makespan: float = 0.0
    uncovered_task_nodes: List[int] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def resolve_task_node(cluster: Any, node_id: int) -> int:
    return int(cluster.virtual_nodes.get(int(node_id), int(node_id)))


def build_runtime_index(cluster: Any) -> Dict[int, int]:
    return {int(node_id): int(index) for index, node_id in cluster.nodes_dict.items()}


def list_service_nodes(cluster: Any) -> List[int]:
    removed_bridge_nodes = {int(node_id) for node_id in getattr(cluster, "virtual_nodes", {}).values()}
    return [
        int(node_id)
        for node_id in cluster.nodes_dict.values()
        if int(node_id) != int(cluster.depot_id) and int(node_id) not in removed_bridge_nodes
    ]


def get_node_coordinates(cluster: Any, node_id: int) -> Tuple[float, float]:
    physical_node = resolve_task_node(cluster, node_id)
    node_rows = cluster.cluster.loc[cluster.cluster["Area_id"] == physical_node, ["X_coords", "Y_coords"]]
    if node_rows.empty:
        raise ValueError(f"Unable to find coordinates for node {physical_node} in cluster {cluster.id}.")
    x_coord, y_coord = node_rows.iloc[0].tolist()
    return float(x_coord), float(y_coord)


def get_move_distance(builder: Any, cluster: Any, source_node: int, target_node: int) -> float:
    source = resolve_task_node(cluster, source_node)
    target = resolve_task_node(cluster, target_node)
    return float(builder.problem_cost_data["distance"][source][target])


def get_move_energy(builder: Any, cluster: Any, source_node: int, target_node: int) -> float:
    source = resolve_task_node(cluster, source_node)
    target = resolve_task_node(cluster, target_node)
    return float(builder.move_energy[source][target])


def get_travel_steps(builder: Any, cluster: Any, source_node: int, target_node: int) -> int:
    source = resolve_task_node(cluster, source_node)
    target = resolve_task_node(cluster, target_node)
    return int(math.ceil(max(float(builder.travel_cost[source][target]), 0.0)))


def get_service_energy(builder: Any) -> float:
    return float(builder.average_coverage_energy) * float(builder.coverage_time)


def get_route_nodes_with_depot(cluster: Any, route_nodes: Sequence[int]) -> List[int]:
    return [int(cluster.depot_id), *[int(node_id) for node_id in route_nodes], int(cluster.depot_id)]


def get_route_distance(builder: Any, cluster: Any, route_nodes: Sequence[int]) -> float:
    route = get_route_nodes_with_depot(cluster, route_nodes)
    total_distance = 0.0
    for index in range(len(route) - 1):
        total_distance += get_move_distance(builder, cluster, route[index], route[index + 1])
    return float(total_distance)


def get_route_travel_time(builder: Any, cluster: Any, route_nodes: Sequence[int]) -> float:
    route = get_route_nodes_with_depot(cluster, route_nodes)
    total_time = 0.0
    for index in range(len(route) - 1):
        total_time += float(get_travel_steps(builder, cluster, route[index], route[index + 1]))
    total_time += float(len(route_nodes)) * float(builder.coverage_time)
    return float(total_time)


def get_route_energy(builder: Any, cluster: Any, route_nodes: Sequence[int]) -> float:
    route = get_route_nodes_with_depot(cluster, route_nodes)
    total_energy = 0.0
    for index in range(len(route) - 1):
        total_energy += get_move_energy(builder, cluster, route[index], route[index + 1])
    total_energy += float(len(route_nodes)) * get_service_energy(builder)
    return float(total_energy)


def is_route_feasible(builder: Any, cluster: Any, route_nodes: Sequence[int]) -> bool:
    return get_route_energy(builder, cluster, route_nodes) <= float(builder.max_battery) + 1e-9


def get_balance_weight(builder: Any) -> float:
    return float(getattr(builder, "heuristic_balance_weight", 0.1))


def compute_task_sequence_objective(
    builder: Any,
    cluster: Any,
    task_sequences: Mapping[int, Sequence[int]],
    uncovered_nodes: Sequence[int] | None = None,
) -> Dict[str, float]:
    route_distances = [get_route_distance(builder, cluster, route_nodes) for route_nodes in task_sequences.values()]
    route_energies = [get_route_energy(builder, cluster, route_nodes) for route_nodes in task_sequences.values()]
    route_times = [get_route_travel_time(builder, cluster, route_nodes) for route_nodes in task_sequences.values()]

    total_distance = float(sum(route_distances))
    total_energy = float(sum(route_energies))
    total_time = float(sum(route_times))
    if route_energies:
        workload_imbalance = float(max(route_energies) - min(route_energies))
    else:
        workload_imbalance = 0.0

    penalty_uncovered = 1e6 * float(len(list(uncovered_nodes or [])))
    penalty_energy_violation = 0.0
    for route_nodes in task_sequences.values():
        energy_excess = max(get_route_energy(builder, cluster, route_nodes) - float(builder.max_battery), 0.0)
        penalty_energy_violation += 1e6 * float(energy_excess)

    weighted_cost = (
        float(builder.objective_weights["distance"]) * total_distance
        + float(builder.objective_weights["energy"]) * total_energy
        + float(builder.objective_weights["travel_time"]) * total_time
        + get_balance_weight(builder) * workload_imbalance
        + penalty_uncovered
        + penalty_energy_violation
    )
    return {
        "objective_value": float(weighted_cost),
        "total_distance": total_distance,
        "total_energy": total_energy,
        "total_time": total_time,
        "workload_imbalance": workload_imbalance,
        "penalty_uncovered": penalty_uncovered,
        "penalty_energy_violation": penalty_energy_violation,
    }


def is_candidate_feasible(
    builder: Any,
    cluster: Any,
    current_node: int,
    candidate_node: int,
    energy_used_so_far: float,
) -> bool:
    required_energy = (
        energy_used_so_far
        + get_move_energy(builder, cluster, current_node, candidate_node)
        + get_service_energy(builder)
        + get_move_energy(builder, cluster, candidate_node, cluster.depot_id)
    )
    return required_energy <= float(builder.max_battery) + 1e-9


def get_feasible_insertion_positions(
    builder: Any,
    cluster: Any,
    route_nodes: Sequence[int],
    candidate_node: int,
) -> List[int]:
    feasible_positions = []
    for insert_at in range(len(route_nodes) + 1):
        candidate_route = list(route_nodes[:insert_at]) + [int(candidate_node)] + list(route_nodes[insert_at:])
        if is_route_feasible(builder, cluster, candidate_route):
            feasible_positions.append(insert_at)
    return feasible_positions


def get_insertion_cost_delta(
    builder: Any,
    cluster: Any,
    route_nodes: Sequence[int],
    candidate_node: int,
    insert_at: int,
) -> float:
    base_metrics = compute_task_sequence_objective(builder, cluster, {0: list(route_nodes)})
    candidate_route = list(route_nodes[:insert_at]) + [int(candidate_node)] + list(route_nodes[insert_at:])
    candidate_metrics = compute_task_sequence_objective(builder, cluster, {0: candidate_route})
    return float(candidate_metrics["objective_value"] - base_metrics["objective_value"])


def select_nearest_candidate(
    builder: Any,
    cluster: Any,
    current_node: int,
    candidate_nodes: Iterable[int],
) -> int | None:
    candidates = list(candidate_nodes)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda node_id: (
            get_move_distance(builder, cluster, current_node, node_id),
            get_move_distance(builder, cluster, cluster.depot_id, node_id),
            resolve_task_node(cluster, node_id),
            int(node_id),
        ),
    )


def select_best_insertion(
    builder: Any,
    cluster: Any,
    task_sequences: Mapping[int, Sequence[int]],
    candidate_node: int,
) -> Tuple[int | None, int | None, float]:
    best_agent_id = None
    best_position = None
    best_delta = float("inf")
    for agent_id, route_nodes in task_sequences.items():
        feasible_positions = get_feasible_insertion_positions(builder, cluster, route_nodes, candidate_node)
        for insert_at in feasible_positions:
            delta = get_insertion_cost_delta(builder, cluster, route_nodes, candidate_node, insert_at)
            if delta < best_delta:
                best_agent_id = int(agent_id)
                best_position = int(insert_at)
                best_delta = float(delta)
    return best_agent_id, best_position, float(best_delta)


def greedy_repair_unassigned_nodes(
    builder: Any,
    cluster: Any,
    task_sequences: Dict[int, List[int]],
    unassigned_nodes: Sequence[int],
) -> List[int]:
    leftovers: List[int] = []
    for node_id in unassigned_nodes:
        agent_id, insert_at, _ = select_best_insertion(builder, cluster, task_sequences, int(node_id))
        if agent_id is None or insert_at is None:
            leftovers.append(int(node_id))
            continue
        task_sequences[int(agent_id)] = (
            list(task_sequences[int(agent_id)][:insert_at])
            + [int(node_id)]
            + list(task_sequences[int(agent_id)][insert_at:])
        )
    return leftovers


def task_sequences_to_solution(
    builder: Any,
    cluster: Any,
    task_sequences: Mapping[int, Sequence[int]],
    uncovered_task_nodes: Sequence[int] | None = None,
    strategy_name: str = "heuristic",
    diagnostics: Dict[str, Any] | None = None,
) -> HeuristicClusterSolution:
    agent_paths: Dict[int, List[Tuple[int, int, int]]] = {}
    agent_finish_times: Dict[int, float] = {}
    total_data_transfer = 0.0
    reward_by_task = build_data_reward_map(builder, cluster)

    for agent_id, route_nodes in task_sequences.items():
        path, finish_time = build_detailed_path(builder, cluster, route_nodes)
        agent_paths[int(agent_id)] = path
        agent_finish_times[int(agent_id)] = float(finish_time)
        total_data_transfer += sum(reward_by_task.get(int(node_id), 0.0) for node_id in route_nodes)

    makespan = max(agent_finish_times.values(), default=0.0)
    covered_nodes = sorted(
        {
            resolve_task_node(cluster, node_id)
            for route_nodes in task_sequences.values()
            for node_id in route_nodes
        }
    )
    uncovered_unique = sorted({int(node_id) for node_id in (uncovered_task_nodes or [])})
    objective_metrics = compute_task_sequence_objective(
        builder,
        cluster,
        task_sequences,
        uncovered_nodes=uncovered_unique,
    )
    raw_status = "Feasible" if covered_nodes else "Infeasible"
    payload_diagnostics = {
        "heuristic_strategy": strategy_name,
        "covered_nodes": covered_nodes,
        "uncovered_physical_nodes": build_uncovered_physical_nodes(cluster, uncovered_unique),
        **objective_metrics,
    }
    if diagnostics:
        payload_diagnostics.update(diagnostics)

    return HeuristicClusterSolution(
        raw_status=raw_status,
        status_code=1 if covered_nodes else -1,
        agent_paths=agent_paths,
        task_sequences={int(agent_id): list(route_nodes) for agent_id, route_nodes in task_sequences.items()},
        agent_finish_times=agent_finish_times,
        total_data_transfer=float(total_data_transfer),
        makespan=float(makespan),
        uncovered_task_nodes=uncovered_unique,
        diagnostics=payload_diagnostics,
    )


def apply_two_opt_local_search(
    builder: Any,
    cluster: Any,
    route_nodes: Sequence[int],
) -> List[int]:
    best_route = list(route_nodes)
    best_distance = get_route_distance(builder, cluster, best_route)
    improved = True

    while improved and len(best_route) >= 4:
        improved = False
        for start in range(len(best_route) - 2):
            for end in range(start + 2, len(best_route) + 1):
                candidate = best_route[:start] + list(reversed(best_route[start:end])) + best_route[end:]
                if not is_route_feasible(builder, cluster, candidate):
                    continue
                candidate_distance = get_route_distance(builder, cluster, candidate)
                if candidate_distance + 1e-9 < best_distance:
                    best_route = candidate
                    best_distance = candidate_distance
                    improved = True
                    break
            if improved:
                break

    return best_route


def seeded_random(seed: int = 42) -> random.Random:
    return random.Random(seed)


def build_detailed_path(
    builder: Any,
    cluster: Any,
    route_nodes: Sequence[int],
) -> Tuple[List[Tuple[int, int, int]], float]:
    if not route_nodes:
        return [], 0.0

    detailed_path: List[Tuple[int, int, int]] = []
    current_node = int(cluster.depot_id)
    current_time = 0

    for next_node in route_nodes:
        source_node = resolve_task_node(cluster, current_node)
        target_node = resolve_task_node(cluster, next_node)
        travel_steps = get_travel_steps(builder, cluster, current_node, next_node)
        for time_step in range(current_time, current_time + travel_steps):
            detailed_path.append((source_node, target_node, time_step))
        current_time += travel_steps

        for time_step in range(current_time, current_time + int(builder.coverage_time)):
            detailed_path.append((target_node, target_node, time_step))
        current_time += int(builder.coverage_time)
        current_node = int(next_node)

    source_node = resolve_task_node(cluster, current_node)
    depot_node = int(cluster.depot_id)
    return_steps = get_travel_steps(builder, cluster, current_node, depot_node)
    for time_step in range(current_time, current_time + return_steps):
        detailed_path.append((source_node, depot_node, time_step))
    current_time += return_steps
    return detailed_path, float(current_time)


def build_data_reward_map(builder: Any, cluster: Any) -> Dict[int, float]:
    runtime_index = build_runtime_index(cluster)
    reward_by_task: Dict[int, float] = {}
    for task_node in list_service_nodes(cluster):
        runtime_node = runtime_index.get(int(task_node))
        reward_by_task[int(task_node)] = float(cluster.R.get(runtime_node, 0.0)) * float(builder.coverage_time)
    return reward_by_task


def build_static_partitions(
    builder: Any,
    cluster: Any,
    agents: Sequence[int],
    service_nodes: Sequence[int],
) -> Dict[int, List[int]]:
    sorted_agents = sorted(int(agent_id) for agent_id in agents)
    partitions = {agent_id: [] for agent_id in sorted_agents}
    if not sorted_agents or not service_nodes:
        return partitions

    node_pool = [int(node_id) for node_id in service_nodes]
    seed_count = min(len(sorted_agents), len(node_pool))
    depot_node = int(cluster.depot_id)

    seeds = [
        max(
            node_pool,
            key=lambda node_id: (
                get_move_distance(builder, cluster, depot_node, node_id),
                resolve_task_node(cluster, node_id),
            ),
        )
    ]
    while len(seeds) < seed_count:
        remaining = [node_id for node_id in node_pool if node_id not in seeds]
        next_seed = max(
            remaining,
            key=lambda node_id: (
                min(get_move_distance(builder, cluster, seed, node_id) for seed in seeds),
                get_move_distance(builder, cluster, depot_node, node_id),
                resolve_task_node(cluster, node_id),
            ),
        )
        seeds.append(next_seed)

    partition_nodes = {seed: [] for seed in seeds}
    for node_id in node_pool:
        closest_seed = min(
            seeds,
            key=lambda seed: (
                get_move_distance(builder, cluster, seed, node_id),
                get_move_distance(builder, cluster, depot_node, node_id),
                resolve_task_node(cluster, node_id),
            ),
        )
        partition_nodes[closest_seed].append(node_id)

    for agent_id, seed in zip(sorted_agents, seeds):
        partitions[agent_id] = sorted(
            partition_nodes[seed],
            key=lambda node_id: (
                get_move_distance(builder, cluster, depot_node, node_id),
                resolve_task_node(cluster, node_id),
                int(node_id),
            ),
        )
    return partitions


def build_uncovered_physical_nodes(cluster: Any, uncovered_task_nodes: Sequence[int]) -> List[int]:
    return sorted({resolve_task_node(cluster, node_id) for node_id in uncovered_task_nodes})


def build_partition_diagnostics(cluster: Any, partition_map: Mapping[int, Sequence[int]]) -> Dict[str, List[int]]:
    diagnostics: Dict[str, List[int]] = {}
    for agent_id, nodes in partition_map.items():
        diagnostics[str(agent_id)] = [resolve_task_node(cluster, node_id) for node_id in nodes]
    return diagnostics
