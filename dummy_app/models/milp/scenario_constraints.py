from __future__ import annotations

from dummy_app.designs.constraint import cooperative_scenario_constraints, individual_scenario_constraints


def apply_scenario_constraints(cluster, builder, scenario: str, list_of_agents, subtour_mode: str) -> None:
    if scenario == "cooperative":
        cooperative_scenario_constraints(
            cluster=cluster,
            builder=builder,
            V_nodes=cluster.V_nodes,
            list_of_agents=list_of_agents,
            subtour_mode=subtour_mode,
        )
        return

    if scenario == "individual":
        individual_scenario_constraints(
            cluster=cluster,
            builder=builder,
            V_nodes=cluster.V_nodes,
            list_of_agents=list_of_agents,
            subtour_mode=subtour_mode,
        )
        return

    raise ValueError(f"Scenario {scenario} not implemented")
