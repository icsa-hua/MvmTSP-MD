from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

from dummy_app.models.base import OptimizationModel
from dummy_app.models.heuristics.common import (
    apply_two_opt_local_search,
    build_partition_diagnostics,
    build_static_partitions,
    compute_task_sequence_objective,
    get_feasible_insertion_positions,
    get_insertion_cost_delta,
    get_move_distance,
    get_move_energy,
    get_service_energy,
    greedy_repair_unassigned_nodes,
    list_service_nodes,
    seeded_random,
    select_nearest_candidate,
    task_sequences_to_solution,
)


@dataclass
class _ALNSSolutionState:
    task_sequences: Dict[int, List[int]]
    uncovered_nodes: List[int]


def _copy_state(solution: _ALNSSolutionState) -> _ALNSSolutionState:
    return _ALNSSolutionState(
        task_sequences={int(agent_id): list(route_nodes) for agent_id, route_nodes in solution.task_sequences.items()},
        uncovered_nodes=list(solution.uncovered_nodes),
    )


def _build_initial_solution(cluster: Any, builder: Any) -> tuple[_ALNSSolutionState, Dict[int, List[int]]]:
    partition_map = build_static_partitions(builder, cluster, cluster.employed_agents, list_service_nodes(cluster))
    task_sequences: Dict[int, List[int]] = {int(agent_id): [] for agent_id in cluster.employed_agents}
    unassigned_nodes: List[int] = []

    for agent_id in sorted(cluster.employed_agents):
        route_nodes: List[int] = []
        remaining_nodes = set(int(node_id) for node_id in partition_map.get(int(agent_id), []))
        current_node = int(cluster.depot_id)
        energy_used = 0.0

        while remaining_nodes:
            feasible_candidates = [
                node_id
                for node_id in remaining_nodes
                if (
                    energy_used
                    + get_move_energy(builder, cluster, current_node, node_id)
                    + get_service_energy(builder)
                    + get_move_energy(builder, cluster, node_id, int(cluster.depot_id))
                )
                <= float(builder.max_battery) + 1e-9
            ]
            if not feasible_candidates:
                break

            next_node = select_nearest_candidate(builder, cluster, current_node, feasible_candidates)
            if next_node is None:
                break

            route_nodes.append(int(next_node))
            energy_used += (
                get_move_energy(builder, cluster, current_node, int(next_node))
                + get_service_energy(builder)
            )
            current_node = int(next_node)
            remaining_nodes.remove(int(next_node))

        task_sequences[int(agent_id)] = route_nodes
        unassigned_nodes.extend(sorted(int(node_id) for node_id in remaining_nodes))

    unresolved_nodes = greedy_repair_unassigned_nodes(builder, cluster, task_sequences, unassigned_nodes)
    for agent_id, route_nodes in list(task_sequences.items()):
        task_sequences[int(agent_id)] = apply_two_opt_local_search(builder, cluster, route_nodes)
    return _ALNSSolutionState(task_sequences=task_sequences, uncovered_nodes=unresolved_nodes), partition_map


def _solution_cost(builder: Any, cluster: Any, solution: _ALNSSolutionState) -> float:
    metrics = compute_task_sequence_objective(
        builder=builder,
        cluster=cluster,
        task_sequences=solution.task_sequences,
        uncovered_nodes=solution.uncovered_nodes,
    )
    return float(metrics["objective_value"])


def _random_removal(
    builder: Any,
    cluster: Any,
    solution: _ALNSSolutionState,
    rng,
    removal_rate: float,
) -> tuple[_ALNSSolutionState, List[int]]:
    partial = _copy_state(solution)
    assigned_nodes = [
        (int(agent_id), int(node_id))
        for agent_id, route_nodes in partial.task_sequences.items()
        for node_id in route_nodes
    ]
    removal_count = max(1, int(removal_rate * len(assigned_nodes))) if assigned_nodes else 0
    removed_pairs = rng.sample(assigned_nodes, k=min(removal_count, len(assigned_nodes))) if assigned_nodes else []
    removed_nodes: List[int] = list(partial.uncovered_nodes)
    partial.uncovered_nodes = []

    for agent_id, node_id in removed_pairs:
        if int(node_id) in partial.task_sequences[int(agent_id)]:
            partial.task_sequences[int(agent_id)].remove(int(node_id))
            removed_nodes.append(int(node_id))
    return partial, removed_nodes


def _node_marginal_cost(builder: Any, cluster: Any, route_nodes: Sequence[int], index: int) -> float:
    route = [int(cluster.depot_id), *[int(node_id) for node_id in route_nodes], int(cluster.depot_id)]
    prev_node = route[index]
    current_node = route[index + 1]
    next_node = route[index + 2]
    return float(
        get_move_distance(builder, cluster, prev_node, current_node)
        + get_move_distance(builder, cluster, current_node, next_node)
        - get_move_distance(builder, cluster, prev_node, next_node)
    )


def _worst_removal(
    builder: Any,
    cluster: Any,
    solution: _ALNSSolutionState,
    rng,
    removal_rate: float,
) -> tuple[_ALNSSolutionState, List[int]]:
    partial = _copy_state(solution)
    node_records: List[Tuple[float, int, int]] = []
    for agent_id, route_nodes in partial.task_sequences.items():
        for index, node_id in enumerate(route_nodes):
            node_records.append((_node_marginal_cost(builder, cluster, route_nodes, index), int(agent_id), int(node_id)))

    node_records.sort(key=lambda item: item[0], reverse=True)
    removal_count = max(1, int(removal_rate * len(node_records))) if node_records else 0
    removed_nodes: List[int] = list(partial.uncovered_nodes)
    partial.uncovered_nodes = []

    for _, agent_id, node_id in node_records[:removal_count]:
        if int(node_id) in partial.task_sequences[int(agent_id)]:
            partial.task_sequences[int(agent_id)].remove(int(node_id))
            removed_nodes.append(int(node_id))
    return partial, removed_nodes


def _route_segment_removal(
    builder: Any,
    cluster: Any,
    solution: _ALNSSolutionState,
    rng,
    removal_rate: float,
) -> tuple[_ALNSSolutionState, List[int]]:
    partial = _copy_state(solution)
    candidate_agents = [int(agent_id) for agent_id, route_nodes in partial.task_sequences.items() if route_nodes]
    removed_nodes: List[int] = list(partial.uncovered_nodes)
    partial.uncovered_nodes = []
    if not candidate_agents:
        return partial, removed_nodes

    selected_agent = int(rng.choice(candidate_agents))
    route_nodes = partial.task_sequences[selected_agent]
    segment_length = max(1, min(len(route_nodes), int(math.ceil(removal_rate * len(route_nodes)))))
    start_index = rng.randint(0, max(len(route_nodes) - segment_length, 0))
    removed_segment = route_nodes[start_index:start_index + segment_length]
    partial.task_sequences[selected_agent] = route_nodes[:start_index] + route_nodes[start_index + segment_length:]
    removed_nodes.extend(int(node_id) for node_id in removed_segment)
    return partial, removed_nodes


def _greedy_best_insertion(
    builder: Any,
    cluster: Any,
    partial: _ALNSSolutionState,
    removed_nodes: Sequence[int],
) -> _ALNSSolutionState:
    candidate = _copy_state(partial)
    unresolved_nodes = greedy_repair_unassigned_nodes(builder, cluster, candidate.task_sequences, removed_nodes)
    candidate.uncovered_nodes = unresolved_nodes
    return candidate


def _regret_two_insertion(
    builder: Any,
    cluster: Any,
    partial: _ALNSSolutionState,
    removed_nodes: Sequence[int],
) -> _ALNSSolutionState:
    candidate = _copy_state(partial)
    pending_nodes = [int(node_id) for node_id in removed_nodes]
    unresolved_nodes: List[int] = []

    while pending_nodes:
        best_choice = None
        best_regret = float("-inf")
        best_node = None

        for node_id in pending_nodes:
            insertion_options: List[Tuple[float, int, int]] = []
            for agent_id, route_nodes in candidate.task_sequences.items():
                feasible_positions = get_feasible_insertion_positions(builder, cluster, route_nodes, int(node_id))
                for insert_at in feasible_positions:
                    delta = get_insertion_cost_delta(builder, cluster, route_nodes, int(node_id), insert_at)
                    insertion_options.append((float(delta), int(agent_id), int(insert_at)))

            insertion_options.sort(key=lambda item: item[0])
            if not insertion_options:
                continue

            best_delta, best_agent_id, best_position = insertion_options[0]
            second_best_delta = insertion_options[1][0] if len(insertion_options) > 1 else best_delta
            regret = float(second_best_delta - best_delta)
            if regret > best_regret:
                best_regret = regret
                best_choice = (int(best_agent_id), int(best_position))
                best_node = int(node_id)

        if best_choice is None or best_node is None:
            unresolved_nodes.extend(int(node_id) for node_id in pending_nodes)
            break

        agent_id, insert_at = best_choice
        candidate.task_sequences[agent_id] = (
            list(candidate.task_sequences[agent_id][:insert_at])
            + [int(best_node)]
            + list(candidate.task_sequences[agent_id][insert_at:])
        )
        pending_nodes.remove(int(best_node))

    candidate.uncovered_nodes = unresolved_nodes
    return candidate


def _apply_local_search(builder: Any, cluster: Any, solution: _ALNSSolutionState) -> _ALNSSolutionState:
    improved = _copy_state(solution)
    for agent_id, route_nodes in list(improved.task_sequences.items()):
        improved.task_sequences[int(agent_id)] = apply_two_opt_local_search(builder, cluster, route_nodes)
    return improved


def _accept_candidate(candidate_cost: float, current_cost: float, temperature: float, rng) -> bool:
    if candidate_cost < current_cost:
        return True
    if temperature <= 1e-9:
        return False
    acceptance_probability = math.exp(-(candidate_cost - current_cost) / temperature)
    return rng.random() < acceptance_probability


def _update_weight(current_weight: float, score: float) -> float:
    return 0.8 * float(current_weight) + 0.2 * float(score)


def solve_alns_baseline(cluster: Any, builder: Any) -> Any:
    rng = seeded_random(int(getattr(builder, "random_seed", 42)) + int(cluster.id))
    current, partition_map = _build_initial_solution(cluster, builder)
    best = _copy_state(current)
    current_cost = _solution_cost(builder, cluster, current)
    best_cost = current_cost

    destroy_operators: Dict[str, Callable[..., tuple[_ALNSSolutionState, List[int]]]] = {
        "random_removal": _random_removal,
        "worst_removal": _worst_removal,
        "route_segment_removal": _route_segment_removal,
    }
    repair_operators: Dict[str, Callable[..., _ALNSSolutionState]] = {
        "greedy_best_insertion": _greedy_best_insertion,
        "regret_two_insertion": _regret_two_insertion,
    }
    destroy_weights = {name: 1.0 for name in destroy_operators}
    repair_weights = {name: 1.0 for name in repair_operators}

    temperature = 100.0
    cooling_rate = 0.995
    max_iter = 2000
    removal_rate = 0.15

    def weighted_choice(weight_map: Dict[str, float]) -> str:
        names = list(weight_map.keys())
        weights = [max(float(weight_map[name]), 1e-9) for name in names]
        total_weight = float(sum(weights))
        threshold = rng.random() * total_weight
        cumulative = 0.0
        for name, weight in zip(names, weights):
            cumulative += weight
            if threshold <= cumulative:
                return name
        return names[-1]

    for _ in range(max_iter):
        destroy_name = weighted_choice(destroy_weights)
        repair_name = weighted_choice(repair_weights)

        previous_current_cost = current_cost
        partial, removed_nodes = destroy_operators[destroy_name](builder, cluster, current, rng, removal_rate)
        candidate = repair_operators[repair_name](builder, cluster, partial, removed_nodes)
        candidate = _apply_local_search(builder, cluster, candidate)
        candidate_cost = _solution_cost(builder, cluster, candidate)

        accepted = _accept_candidate(candidate_cost, current_cost, temperature, rng)
        if accepted:
            current = _copy_state(candidate)
            current_cost = candidate_cost

        if candidate_cost + 1e-9 < best_cost:
            best = _copy_state(candidate)
            best_cost = candidate_cost
            score = 5.0
        elif candidate_cost + 1e-9 < previous_current_cost:
            score = 2.0
        elif accepted:
            score = 1.0
        else:
            score = 0.0

        destroy_weights[destroy_name] = _update_weight(destroy_weights[destroy_name], score)
        repair_weights[repair_name] = _update_weight(repair_weights[repair_name], score)
        temperature *= cooling_rate

    return task_sequences_to_solution(
        builder=builder,
        cluster=cluster,
        task_sequences=best.task_sequences,
        uncovered_task_nodes=best.uncovered_nodes,
        strategy_name="alns_baseline",
        diagnostics={
            "partition_by_agent": build_partition_diagnostics(cluster, partition_map),
            "destroy_weights": {name: float(weight) for name, weight in destroy_weights.items()},
            "repair_weights": {name: float(weight) for name, weight in repair_weights.items()},
            "alns_temperature_final": float(temperature),
            "alns_iterations": int(max_iter),
            "alns_best_objective": float(best_cost),
        },
    )


class ALNSOptimizationModel(OptimizationModel):
    model_name = "alns"

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def solve_cluster(self, instance, cluster_input, request):
        return self.runtime.solve_cluster_instance_heuristic(
            instance=instance,
            cluster_input=cluster_input,
            request=request,
            heuristic_solver=solve_alns_baseline,
            heuristic_name=self.model_name,
        )
