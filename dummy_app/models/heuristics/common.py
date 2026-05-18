from __future__ import annotations

import math
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
    return [int(node_id) for node_id in cluster.nodes_dict.values() if int(node_id) != int(cluster.depot_id)]


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
