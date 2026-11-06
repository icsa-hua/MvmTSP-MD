from __future__ import annotations

import pulp as pl

from dummy_app.models.genetic_algorithm import get_weights


def set_hybrid_objective(cluster, distance, energy, time):
    weights = get_weights(cluster.builder_objective_weights or None)
    energy_cost = pl.lpSum(
        cluster.x[i, j, k] * energy[source][target] * weights["energy"]
        for i, source in cluster.nodes_dict.items()
        for j, target in cluster.nodes_dict.items()
        if i != j and (source != cluster.depot_id and target != cluster.depot_id)
        for k in cluster.employed_agents
    )
    spatial_cost = pl.lpSum(
        cluster.x[i, j, k] * distance[source][target] * weights["distance"]
        for i, source in cluster.nodes_dict.items()
        for j, target in cluster.nodes_dict.items()
        if i != j and (source != cluster.depot_id and target != cluster.depot_id)
        for k in cluster.employed_agents
    )
    travel_time_cost = pl.lpSum(
        cluster.x[i, j, k] * time[source][target] * weights["travel_time"]
        for i, source in cluster.nodes_dict.items()
        for j, target in cluster.nodes_dict.items()
        if i != j and (source != cluster.depot_id and target != cluster.depot_id)
        for k in cluster.employed_agents
    )
    return_times_sum = pl.lpSum(cluster.return_step[k] for k in cluster.employed_agents)
    cluster.total_cost = energy_cost + spatial_cost + travel_time_cost
    cluster.problem.setObjective(1.0 * cluster.total_cost + 0.001 * return_times_sum)


def set_energy_objective(cluster, distance, energy, time):
    weights = get_weights(cluster.builder_objective_weights or None)
    energy_cost = pl.lpSum(
        cluster.x[i, j, k] * energy[source][target] * weights["energy"]
        for i, source in cluster.nodes_dict.items()
        for j, target in cluster.nodes_dict.items()
        if i != j and (source != cluster.depot_id and target != cluster.depot_id)
        for k in cluster.employed_agents
    )
    spatial_cost = pl.lpSum(
        cluster.x[i, j, k] * distance[source][target] * weights["distance"]
        for i, source in cluster.nodes_dict.items()
        for j, target in cluster.nodes_dict.items()
        if i != j and (source != cluster.depot_id and target != cluster.depot_id)
        for k in cluster.employed_agents
    )
    travel_time_cost = pl.lpSum(
        cluster.x[i, j, k] * time[source][target] * weights["travel_time"]
        for i, source in cluster.nodes_dict.items()
        for j, target in cluster.nodes_dict.items()
        if i != j and (source != cluster.depot_id and target != cluster.depot_id)
        for k in cluster.employed_agents
    )
    cluster.total_cost = energy_cost + spatial_cost + travel_time_cost
    cluster.problem.setObjective(cluster.total_cost)


def set_pareto_energy_objective(cluster, energy):
    cluster.spatial_cost = pl.lpSum(
        cluster.x[i, j, k] * energy[source][target]
        for i, source in cluster.nodes_dict.items()
        for j, target in cluster.nodes_dict.items()
        if i != j
        for k in cluster.employed_agents
    )
    cluster.problem.setObjective(cluster.spatial_cost)


def set_max_coverage_objective(cluster, builder):
    cluster.problem.setObjective(-cluster.total_data_collected_main)


def set_sum_return_times_objective(cluster):
    cluster.problem.setObjective(pl.lpSum(cluster.return_step[k] for k in cluster.employed_agents))


def set_makespan_objective(cluster, distance, energy, time):
    cluster.problem.setObjective(cluster.makespan)
