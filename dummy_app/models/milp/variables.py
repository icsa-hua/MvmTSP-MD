from __future__ import annotations

import pulp as pl


def create_problem_variables(cluster, scenario: str = "cooperative") -> None:
    V = cluster.V_nodes
    NODES = cluster.NODES

    cluster.x = pl.LpVariable.dicts("x", ((i, j, v) for i in V for j in V for v in cluster.employed_agents), cat="Binary")
    cluster.t = pl.LpVariable.dicts("t", ((i, v) for i in V for v in cluster.employed_agents), lowBound=0, cat="Continuous")
    cluster.visit = pl.LpVariable.dicts("v", ((j, v) for j in NODES for v in cluster.employed_agents), lowBound=0, upBound=1, cat="Binary")
    cluster.p = pl.LpVariable.dicts("p", ((j, v) for j in V for v in cluster.employed_agents), lowBound=0, upBound=len(V) - 1, cat="Integer")
    cluster.u = pl.LpVariable.dicts("u", (v for v in cluster.employed_agents), lowBound=0, cat="Integer")
    cluster.return_step = pl.LpVariable.dicts("return", (k for k in cluster.employed_agents), lowBound=cluster.timeframe[0], cat="Continuous")
    cluster.e = pl.LpVariable.dicts("e", ((i, v) for i in V for v in cluster.employed_agents), lowBound=0, upBound=cluster.max_battery, cat="Continuous")
    cluster.makespan = pl.LpVariable("makespan", lowBound=0, cat="Continuous")
    cluster.service_time = pl.LpVariable.dicts("service_time", ((i, v) for i in V for v in cluster.employed_agents), lowBound=0, cat="Continuous")
    cluster.problem = pl.LpProblem(name=f"MVMTSP_Cluster_{cluster.id}", sense=pl.LpMinimize)
    cluster.total_data_collected_main = pl.LpVariable("total_data_collected_main", lowBound=0, cat="Continuous")

    if scenario == "individual":
        cluster.precedes = pl.LpVariable.dicts(
            "precedes",
            ((j, k1, k2) for j in V for k1 in cluster.employed_agents for k2 in cluster.employed_agents if k1 < k2),
            cat="Binary",
        )
