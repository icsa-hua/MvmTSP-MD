from __future__ import annotations


def apply_mtz_constraints(cluster, agents, nodes, depot_ind) -> None:
    n = len(nodes)
    for k in agents:
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                cluster.problem += cluster.p[i, k] - cluster.p[j, k] + n * cluster.x[i, j, k] <= n - 1
    for k in agents:
        cluster.problem += cluster.p[depot_ind, k] == 0
