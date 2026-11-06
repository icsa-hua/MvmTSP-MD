from __future__ import annotations

import pulp as pl


def apply_flow_constraints(cluster, agents, nodes, depot_ind) -> None:
    non_depot_nodes = [int(node) for node in nodes if int(node) != int(depot_ind)]
    all_nodes = [int(depot_ind), *non_depot_nodes]
    flow_upper_bound = len(non_depot_nodes)

    cluster.f = {}
    if flow_upper_bound <= 0:
        return

    for agent_id in agents:
        for source in all_nodes:
            for target in all_nodes:
                if source == target or (source, target, int(agent_id)) not in cluster.x:
                    continue

                cluster.f[source, target, int(agent_id)] = pl.LpVariable(
                    f"flow_{source}_{target}_{int(agent_id)}",
                    lowBound=0,
                    upBound=flow_upper_bound,
                    cat="Continuous",
                )

                cluster.problem += (
                    cluster.f[source, target, int(agent_id)] <= flow_upper_bound * cluster.x[source, target, int(agent_id)],
                    f"flow_link_{source}_{target}_{int(agent_id)}",
                )

    for agent_id in agents:
        cluster.problem += (
            pl.lpSum(
                cluster.f[depot_ind, target, int(agent_id)]
                for target in non_depot_nodes
                if (depot_ind, target, int(agent_id)) in cluster.f
            )
            == pl.lpSum(
                cluster.x[source, target, int(agent_id)]
                for target in non_depot_nodes
                for source in all_nodes
                if source != target and (source, target, int(agent_id)) in cluster.x
            ),
            f"flow_depot_supply_{int(agent_id)}",
        )

        for target in non_depot_nodes:
            inflow = pl.lpSum(
                cluster.f[source, target, int(agent_id)]
                for source in all_nodes
                if source != target and (source, target, int(agent_id)) in cluster.f
            )
            outflow = pl.lpSum(
                cluster.f[target, sink, int(agent_id)]
                for sink in all_nodes
                if sink != target and (target, sink, int(agent_id)) in cluster.f
            )
            visited = pl.lpSum(
                cluster.x[source, target, int(agent_id)]
                for source in all_nodes
                if source != target and (source, target, int(agent_id)) in cluster.x
            )
            cluster.problem += (
                inflow - outflow == visited,
                f"flow_conservation_{target}_{int(agent_id)}",
            )
