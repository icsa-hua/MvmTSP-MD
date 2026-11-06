from __future__ import annotations

import unittest
from types import SimpleNamespace

import pulp as pl

from dummy_app.models.milp.solver_adapter import solve_cluster_problem
from dummy_app.models.milp.subtour.dfj import find_disconnected_subtours
from dummy_app.models.milp.subtour.strategies import apply_subtour_constraints


def _build_synthetic_cluster(subtour_mode: str):
    depot = 0
    nodes = [0, 1, 2, 3, 4, 5]
    non_depot_nodes = [1, 2, 3, 4, 5]
    agent_id = 0

    cluster = SimpleNamespace()
    cluster.id = 999
    cluster.depot_id = depot
    cluster.employed_agents = [agent_id]
    cluster.V_nodes = list(nodes)
    cluster.NODES = list(non_depot_nodes)
    cluster.nodes_dict = {node: node for node in nodes}
    cluster.problem = pl.LpProblem(name=f"synthetic_{subtour_mode}", sense=pl.LpMinimize)
    cluster.x = pl.LpVariable.dicts("x", ((i, j, agent_id) for i in nodes for j in nodes), cat="Binary")
    cluster.visit = pl.LpVariable.dicts(
        "visit",
        ((node, agent_id) for node in non_depot_nodes),
        lowBound=0,
        upBound=1,
        cat="Binary",
    )
    cluster.p = pl.LpVariable.dicts(
        "order",
        ((node, agent_id) for node in nodes),
        lowBound=0,
        upBound=len(nodes) - 1,
        cat="Integer",
    )
    cluster.warm_start_summary = {}
    cluster.warm_start_solution = {}

    for node in non_depot_nodes:
        cluster.problem += pl.lpSum(cluster.x[source, node, agent_id] for source in nodes) == cluster.visit[node, agent_id]
        cluster.problem += pl.lpSum(cluster.x[node, target, agent_id] for target in nodes) == cluster.visit[node, agent_id]
        cluster.problem += cluster.visit[node, agent_id] == 1

    cluster.problem += pl.lpSum(cluster.x[depot, node, agent_id] for node in non_depot_nodes) == 1
    cluster.problem += pl.lpSum(cluster.x[node, depot, agent_id] for node in non_depot_nodes) == 1

    for source in nodes:
        for target in nodes:
            cluster.problem += cluster.x[source, target, agent_id] + cluster.x[target, source, agent_id] <= 1

    for node in non_depot_nodes:
        cluster.problem += (
            pl.lpSum(cluster.x[node, target, agent_id] for target in nodes)
            == pl.lpSum(cluster.x[source, node, agent_id] for source in nodes)
        )

    cheap_arcs = {
        (0, 1): 1.0,
        (1, 2): 1.0,
        (2, 0): 1.0,
        (3, 4): 1.0,
        (4, 5): 1.0,
        (5, 3): 1.0,
    }
    cluster.problem += pl.lpSum(
        cheap_arcs.get((source, target), 100.0) * cluster.x[source, target, agent_id]
        for source in nodes
        for target in nodes
    )

    apply_subtour_constraints(cluster, cluster.employed_agents, cluster.NODES, depot, subtour_mode)
    return cluster


@unittest.skipUnless(pl.PULP_CBC_CMD().available(), "CBC solver is not available")
class SubtourModeTests(unittest.TestCase):
    def _solve_direct(self, subtour_mode: str):
        cluster = _build_synthetic_cluster(subtour_mode)
        status_code = cluster.problem.solve(pl.PULP_CBC_CMD(msg=False))
        self.assertEqual(pl.LpStatus[status_code], "Optimal")
        return cluster

    def test_disabled_subtour_constraints_allow_disconnected_cycle(self):
        cluster = self._solve_direct("none")
        self.assertAlmostEqual(float(pl.value(cluster.problem.objective)), 6.0, places=5)
        self.assertEqual(find_disconnected_subtours(cluster), [(0, (3, 4, 5))])

    def test_mtz_eliminates_disconnected_cycle(self):
        cluster = self._solve_direct("mtz")
        self.assertGreater(float(pl.value(cluster.problem.objective)), 6.0)
        self.assertEqual(find_disconnected_subtours(cluster), [])

    def test_flow_eliminates_disconnected_cycle(self):
        cluster = self._solve_direct("flow")
        self.assertGreater(float(pl.value(cluster.problem.objective)), 6.0)
        self.assertEqual(find_disconnected_subtours(cluster), [])

    def test_iterative_dfj_eliminates_disconnected_cycle(self):
        cluster = _build_synthetic_cluster("dfj_iter")
        solve_metadata = solve_cluster_problem(
            cluster=cluster,
            builder=SimpleNamespace(subtour_mode="dfj_iter", max_dfj_rounds=5),
            time_limit_seconds=30,
            solver_backend="cbc",
            solver_seed=7,
        )
        self.assertEqual(solve_metadata["raw_status"], "Optimal")
        self.assertGreaterEqual(int(solve_metadata["dfj_cuts_added"]), 1)
        self.assertGreater(float(pl.value(cluster.problem.objective)), 6.0)
        self.assertEqual(find_disconnected_subtours(cluster), [])


if __name__ == "__main__":
    unittest.main()
