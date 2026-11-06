from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Union

import networkx as nx
import numpy as np

from dummy_app.models.central_hubs import CentralHub
from dummy_app.models.genetic_algorithm import GASolution, get_weights
from dummy_app.tools.common import add_virtual_nodes
from dummy_app.tools.graphs import is_eulerian_digraph


def normalize_warm_start_mode(mode: Any) -> str:
    normalized = str(mode or "none").strip().lower()
    if normalized in {"auto", "automatic", "default"}:
        return "auto"
    if normalized in {"", "none", "off", "disabled", "false"}:
        return "none"
    if normalized in {"ga", "ga_only", "ga_plus_time_windows", "genetic_algorithm"}:
        return "ga"
    if normalized in {"alns", "alns_only", "alns_plus_time_windows"}:
        return "alns"
    if normalized in {"greedy_nn", "gnn_ntw", "global_greedy_nn"}:
        return "greedy_nn"
    if normalized in {"greedy_partition_nn", "static_partition_greedy_nn"}:
        return "greedy_partition_nn"
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
    bridge_visit_override = getattr(problem_builder, "bridge_node_required_visits_override", None)
    if bridge_visit_override is not None:
        bridge_visit_override = max(int(bridge_visit_override), 1)

    required_visits = []
    for i in nodes_dict:
        if nodes_dict[i] in bridge_nodes:
            if bridge_visit_override is not None:
                allowed_visits = int(bridge_visit_override)
            else:
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


def _resolve_runtime_node(cluster: Any, node_id: int) -> int:
    return int(cluster.virtual_nodes.get(int(node_id), int(node_id)))


def _set_initial_value(variable: Any, value: float | int) -> None:
    if hasattr(variable, "setInitialValue"):
        variable.setInitialValue(value)
        return
    variable.varValue = value


def compute_postprocessed_mission_objective(
    builder: Any,
    cluster: Any,
    task_sequences: Mapping[int, List[int]],
) -> Dict[str, float]:
    weights = get_weights(getattr(builder, "objective_weights", None))
    total_distance = 0.0
    total_energy = 0.0
    total_travel_time = 0.0

    for agent_id in cluster.employed_agents:
        route_nodes = [int(node_id) for node_id in task_sequences.get(int(agent_id), [])]
        route = [int(cluster.depot_id), *route_nodes, int(cluster.depot_id)]

        for source_node, target_node in zip(route[:-1], route[1:]):
            resolved_source = _resolve_runtime_node(cluster, source_node)
            resolved_target = _resolve_runtime_node(cluster, target_node)
            total_distance += float(builder.problem_cost_data["distance"][resolved_source][resolved_target])
            total_energy += float(builder.move_energy[resolved_source][resolved_target])
            total_travel_time += float(builder.travel_cost[resolved_source][resolved_target])

        total_travel_time += float(len(route_nodes)) * float(builder.coverage_time)
        total_energy += float(len(route_nodes)) * float(builder.average_coverage_energy) * float(builder.coverage_time)

    objective_value = (
        float(weights["distance"]) * float(total_distance)
        + float(weights["energy"]) * float(total_energy)
        + float(weights["travel_time"]) * float(total_travel_time)
    )
    return {
        "objective_value": float(objective_value),
        "total_distance": float(total_distance),
        "total_energy": float(total_energy),
        "total_travel_time": float(total_travel_time),
    }


def compute_solver_scale_warm_start_objective(
    builder: Any,
    *,
    mission_objective_value: float,
    makespan: float,
    total_data_transfer: float,
    absolute_agent_finish_times: Mapping[int, float],
) -> float:
    if getattr(builder, "objective_function", "") == "coverage":
        return float(-total_data_transfer)
    if int(getattr(builder, "stage_solution", 1)) == 1:
        return float(mission_objective_value) + 0.001 * float(sum(float(value) for value in absolute_agent_finish_times.values()))
    return float(makespan)


def build_warm_start_payload(
    builder: Any,
    cluster: Any,
    heuristic_solution: Any,
    *,
    strategy_name: str,
    generation_time_sec: float,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    task_sequences = {
        int(agent_id): [int(node_id) for node_id in route_nodes]
        for agent_id, route_nodes in dict(getattr(heuristic_solution, "task_sequences", {})).items()
    }
    local_agent_finish_times = {
        int(agent_id): float(finish_time)
        for agent_id, finish_time in dict(getattr(heuristic_solution, "agent_finish_times", {})).items()
    }
    absolute_agent_finish_times = {
        int(agent_id): float(cluster.agent_start_times.get(int(agent_id), 0.0)) + float(local_agent_finish_times.get(int(agent_id), 0.0))
        for agent_id in cluster.employed_agents
    }
    mission_metrics = compute_postprocessed_mission_objective(builder, cluster, task_sequences)
    solver_objective_value = compute_solver_scale_warm_start_objective(
        builder,
        mission_objective_value=float(mission_metrics["objective_value"]),
        makespan=float(getattr(heuristic_solution, "makespan", 0.0) or 0.0),
        total_data_transfer=float(getattr(heuristic_solution, "total_data_transfer", 0.0) or 0.0),
        absolute_agent_finish_times=absolute_agent_finish_times,
    )

    uncovered_nodes = sorted(
        int(node_id)
        for node_id in (
            dict(getattr(heuristic_solution, "diagnostics", {})).get("uncovered_physical_nodes")
            or getattr(heuristic_solution, "uncovered_task_nodes", [])
            or []
        )
    )
    all_agents_populated = all(bool(task_sequences.get(int(agent_id), [])) for agent_id in cluster.employed_agents)
    available = all_agents_populated and not uncovered_nodes
    summary = {
        "strategy": strategy_name,
        "available": bool(available),
        "provided_to_solver": False,
        "accepted_by_solver": None,
        "acceptance_source": "",
        "generation_time_sec": float(generation_time_sec),
        "objective_value": float(mission_metrics["objective_value"]),
        "solver_objective_value": float(solver_objective_value),
        "makespan": float(getattr(heuristic_solution, "makespan", 0.0) or 0.0),
        "timeframe_estimate": max(1, int(np.ceil(float(getattr(heuristic_solution, "makespan", 0.0) or 0.0))))
        if float(getattr(heuristic_solution, "makespan", 0.0) or 0.0) > 0.0
        else None,
        "covered_nodes": sorted(
            int(node_id)
            for node_id in dict(getattr(heuristic_solution, "diagnostics", {})).get("covered_nodes", [])
        ),
        "uncovered_nodes": uncovered_nodes,
        "total_distance": float(mission_metrics["total_distance"]),
        "total_energy": float(mission_metrics["total_energy"]),
        "total_travel_time": float(mission_metrics["total_travel_time"]),
        "local_return_time_sum": float(sum(float(value) for value in local_agent_finish_times.values())),
        "absolute_return_time_sum": float(sum(float(value) for value in absolute_agent_finish_times.values())),
    }
    payload = {
        "task_sequences": task_sequences,
        "local_agent_finish_times": local_agent_finish_times,
        "absolute_agent_finish_times": absolute_agent_finish_times,
        "covered_nodes": list(summary["covered_nodes"]),
        "uncovered_nodes": list(summary["uncovered_nodes"]),
        "total_data_transfer": float(getattr(heuristic_solution, "total_data_transfer", 0.0) or 0.0),
    }
    return summary, payload


def apply_warm_start_to_model(cluster: Any, builder: Any) -> Dict[str, Any]:
    summary = dict(getattr(cluster, "warm_start_summary", {}))
    payload = dict(getattr(cluster, "warm_start_solution", {}))
    task_sequences = {
        int(agent_id): [int(node_id) for node_id in route_nodes]
        for agent_id, route_nodes in dict(payload.get("task_sequences", {})).items()
    }
    if not summary or summary.get("strategy") in {"", None, "none"}:
        return {"requested": False, "applied": False, "reason": "warm_start_disabled"}
    if not summary.get("available"):
        return {"requested": True, "applied": False, "reason": "warm_start_unavailable"}
    if not task_sequences:
        return {"requested": True, "applied": False, "reason": "warm_start_missing_routes"}

    reverse_nodes = {int(node_id): int(index) for index, node_id in cluster.nodes_dict.items()}
    depot_index = reverse_nodes[int(cluster.depot_id)]
    service_time = float(getattr(builder, "coverage_time", 0.0))
    full_battery = float(getattr(builder, "max_battery", 0.0))

    for agent_id in cluster.employed_agents:
        for runtime_node in task_sequences.get(int(agent_id), []):
            if int(runtime_node) not in reverse_nodes:
                return {
                    "requested": True,
                    "applied": False,
                    "reason": f"warm_start_unknown_node_{int(runtime_node)}",
                }

    for _, variable in cluster.x.items():
        _set_initial_value(variable, 0)
    for _, variable in cluster.visit.items():
        _set_initial_value(variable, 0)
    for _, variable in cluster.p.items():
        _set_initial_value(variable, 0)
    for _, variable in cluster.u.items():
        _set_initial_value(variable, 0)
    for _, variable in cluster.t.items():
        _set_initial_value(variable, 0.0)
    for _, variable in cluster.e.items():
        _set_initial_value(variable, 0.0)
    for _, variable in cluster.service_time.items():
        _set_initial_value(variable, 0.0)

    max_return_step = 0.0
    for agent_id in cluster.employed_agents:
        route_nodes = [int(node_id) for node_id in task_sequences.get(int(agent_id), [])]
        if not route_nodes:
            return {"requested": True, "applied": False, "reason": f"agent_{int(agent_id)}_has_empty_route"}

        start_time = float(cluster.agent_start_times.get(int(agent_id), 0.0))
        current_index = depot_index
        current_runtime_node = int(cluster.depot_id)
        current_time = float(start_time)
        current_energy = float(full_battery)

        _set_initial_value(cluster.start_step[int(agent_id)], start_time)
        _set_initial_value(cluster.e[depot_index, int(agent_id)], full_battery)
        _set_initial_value(cluster.p[depot_index, int(agent_id)], 0)
        _set_initial_value(cluster.u[int(agent_id)], len(route_nodes))

        for position, runtime_node in enumerate(route_nodes, start=1):
            node_index = reverse_nodes[int(runtime_node)]
            resolved_source = _resolve_runtime_node(cluster, current_runtime_node)
            resolved_target = _resolve_runtime_node(cluster, int(runtime_node))
            travel_time = float(cluster.get_travel_times(current_index, node_index, cluster.nodes_dict, builder))
            travel_energy = float(builder.move_energy[resolved_source][resolved_target])

            current_time += travel_time
            current_energy -= travel_energy

            _set_initial_value(cluster.x[current_index, node_index, int(agent_id)], 1)
            _set_initial_value(cluster.visit[node_index, int(agent_id)], 1)
            _set_initial_value(cluster.p[node_index, int(agent_id)], position)
            _set_initial_value(cluster.t[node_index, int(agent_id)], current_time)
            _set_initial_value(cluster.service_time[node_index, int(agent_id)], service_time)
            _set_initial_value(cluster.e[node_index, int(agent_id)], max(current_energy, 0.0))

            current_time += service_time
            current_energy -= float(builder.average_coverage_energy) * service_time
            current_index = int(node_index)
            current_runtime_node = int(runtime_node)

        resolved_last_node = _resolve_runtime_node(cluster, current_runtime_node)
        return_travel_time = float(cluster.get_travel_times(current_index, depot_index, cluster.nodes_dict, builder))
        return_travel_energy = float(builder.move_energy[resolved_last_node][int(cluster.depot_id)])
        _set_initial_value(cluster.x[current_index, depot_index, int(agent_id)], 1)
        current_energy -= return_travel_energy
        return_step = current_time + return_travel_time
        _set_initial_value(cluster.return_step[int(agent_id)], return_step)
        max_return_step = max(max_return_step, float(return_step))

    _set_initial_value(cluster.makespan, max_return_step)
    return {
        "requested": True,
        "applied": True,
        "reason": "warm_start_applied",
        "strategy": summary.get("strategy", ""),
        "solver_objective_value": summary.get("solver_objective_value"),
        "mission_objective_value": summary.get("objective_value"),
    }
