from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from dummy_app.models.base import OptimizationModel
from dummy_app.models.heuristics.common import (
    HeuristicClusterSolution,
    build_data_reward_map,
    build_detailed_path,
    build_uncovered_physical_nodes,
    is_candidate_feasible,
    list_service_nodes,
    resolve_task_node,
    select_nearest_candidate,
)


@dataclass
class _AgentState:
    current_node: int
    route_nodes: List[int] = field(default_factory=list)
    energy_used: float = 0.0
    distance_cost: float = 0.0
    closed: bool = False


def solve_global_greedy_nn(cluster: Any, builder: Any) -> HeuristicClusterSolution:
    service_nodes = list_service_nodes(cluster)
    remaining_nodes = set(service_nodes)
    reward_by_task = build_data_reward_map(builder, cluster)
    states = {
        int(agent_id): _AgentState(current_node=int(cluster.depot_id))
        for agent_id in sorted(cluster.employed_agents)
    }

    total_data_transfer = 0.0
    while remaining_nodes:
        open_agents = [agent_id for agent_id, state in states.items() if not state.closed]
        if not open_agents:
            break

        selected_agent = min(
            open_agents,
            key=lambda agent_id: (
                states[agent_id].distance_cost,
                len(states[agent_id].route_nodes),
                agent_id,
            ),
        )
        state = states[selected_agent]
        feasible_candidates = [
            node_id
            for node_id in remaining_nodes
            if is_candidate_feasible(builder, cluster, state.current_node, node_id, state.energy_used)
        ]
        if not feasible_candidates:
            state.closed = True
            continue

        next_node = select_nearest_candidate(builder, cluster, state.current_node, feasible_candidates)
        if next_node is None:
            state.closed = True
            continue

        state.route_nodes.append(int(next_node))
        state.energy_used += (
            builder.move_energy[resolve_task_node(cluster, state.current_node)][resolve_task_node(cluster, next_node)]
            + float(builder.average_coverage_energy) * float(builder.coverage_time)
        )
        state.distance_cost += builder.problem_cost_data["distance"][
            resolve_task_node(cluster, state.current_node)
        ][resolve_task_node(cluster, next_node)]
        state.current_node = int(next_node)
        remaining_nodes.remove(int(next_node))
        total_data_transfer += reward_by_task.get(int(next_node), 0.0)

    agent_paths: Dict[int, List[tuple[int, int, int]]] = {}
    agent_finish_times: Dict[int, float] = {}
    for agent_id, state in states.items():
        path, finish_time = build_detailed_path(builder, cluster, state.route_nodes)
        agent_paths[agent_id] = path
        agent_finish_times[agent_id] = finish_time

    makespan = max(agent_finish_times.values(), default=0.0)
    covered_nodes = sorted(
        {
            resolve_task_node(cluster, node_id)
            for state in states.values()
            for node_id in state.route_nodes
        }
    )
    uncovered_task_nodes = sorted(int(node_id) for node_id in remaining_nodes)
    raw_status = "Feasible" if covered_nodes else "Infeasible"

    return HeuristicClusterSolution(
        raw_status=raw_status,
        status_code=1 if covered_nodes else -1,
        agent_paths=agent_paths,
        task_sequences={agent_id: list(state.route_nodes) for agent_id, state in states.items()},
        agent_finish_times=agent_finish_times,
        total_data_transfer=float(total_data_transfer),
        makespan=float(makespan),
        uncovered_task_nodes=uncovered_task_nodes,
        diagnostics={
            "heuristic_strategy": "global_greedy_nn",
            "covered_nodes": covered_nodes,
            "uncovered_physical_nodes": build_uncovered_physical_nodes(cluster, uncovered_task_nodes),
        },
    )


class GlobalGreedyNNOptimizationModel(OptimizationModel):
    model_name = "greedy_nn"

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def solve_cluster(self, instance, cluster_input, request):
        return self.runtime.solve_cluster_instance_heuristic(
            instance=instance,
            cluster_input=cluster_input,
            request=request,
            heuristic_solver=solve_global_greedy_nn,
            heuristic_name=self.model_name,
        )
