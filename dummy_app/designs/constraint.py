from dummy_app.tools.logger import logger 
from dummy_app.models.milp.subtour.strategies import apply_subtour_constraints

import pulp as pl 
import numpy as np 

from collections import defaultdict
from typing import Any


def get_arcs(V_nodes, depot_ind): 

    valid_arcs = [(i,j) for i in V_nodes for j in V_nodes if i != j and i != depot_ind and j != depot_ind]
    in_arcs = defaultdict(list)
    out_arcs = defaultdict(list)

    for i, j in valid_arcs:
        out_arcs[i].append(j)
        in_arcs[j].append(i)
    return out_arcs, in_arcs


def get_depot_node(depot_id, nodes_dict):
    reverse = {v: k for k, v in nodes_dict.items()}
    return reverse[depot_id]


def cooperative_scenario_constraints(cluster:Any, builder: Any, V_nodes:list, list_of_agents:dict, subtour_strategy: str = "mtz"): 
    """
    This function sets all the constraints (Spatial flow, Energy constraints, Time progression, synchronization and path completion) 
    NOTE: No objective function is enforced here. Constraints are mathematical inequalities and are linear. This scenario 
          forces that the agents, collctively visit all areas inside the cluster graph. 
    """
    depot_ind, NODES, agents, model, D, TF, MANDATORY_WAIT_TIME, wait_energy_consumption, data_per_visit = configuration_set_up(cluster=cluster, builder=builder, V_nodes=V_nodes, list_of_agents=list_of_agents)

    # Collective visits for the group of agents. 
    for j in NODES:
        model += pl.lpSum(cluster.visit[j,k] for k in agents) == 1

    # Each agents traverses through all nodes the appropriate amount of times. 
    for j in NODES:
        for k in agents: 
            model += pl.lpSum(cluster.x[i,j,k] for i in V_nodes) == cluster.visit[j,k]
            model += pl.lpSum(cluster.x[j,i,k] for i in V_nodes) == cluster.visit[j,k]

    # Start & End at depot 
    for k in agents: 
        model += pl.lpSum(cluster.x[depot_ind,i,k] for i in NODES) == 1
        model += pl.lpSum(cluster.x[i,depot_ind,k] for i in NODES) == 1 

    # Flow conservation (No loops)  
    for i in V_nodes: 
        for j in V_nodes: 
            for k in agents: 
                model += cluster.x[i,j,k] + cluster.x[j,i,k] <= 1 
    
    # Minimum visits bound and count the total number of nodes visited
    for k in agents: 
        # this constraint won't function for larger agent population cases where only one location is feaisble. >=2 change to >= 1
        model += pl.lpSum(cluster.visit[j,k] for j in NODES) >= 1
        model += cluster.u[k] == pl.lpSum(cluster.visit[j,k] for j in NODES)

    # Path continuity flow 
    for i in NODES: 
        for k in agents: 
            model += pl.lpSum(cluster.x[i,j,k] for j in V_nodes) == pl.lpSum(cluster.x[j,i,k] for j in V_nodes)

    apply_subtour_constraints(cluster, agents, NODES, depot_ind, subtour_strategy)
    
    M = TF[-1]
    original_depot_ind = get_depot_node(cluster.depot_id, cluster.original_nodes_dict)
    dept = cluster.original_nodes_dict[original_depot_ind]

    # Depot-side operational release time for the current cluster solve.
    for k in agents:
        agent_release_time = float(getattr(cluster, "agent_start_times", {}).get(k, 0.0))
        model += cluster.start_step[k] >= agent_release_time, f"depot_release_lb_{cluster.id}_{k}"
        model += cluster.start_step[k] <= agent_release_time, f"depot_release_ub_{cluster.id}_{k}"

    # Time progression specific to match the depot case (No coverage time added) 
    for k in agents:
        for j in NODES:
            target = cluster.original_nodes_dict[j]
            # Arrival at first node >= mission start at depot + travel time.
            model += cluster.t[j, k] >= (cluster.start_step[k] + D[(dept, target)]) - M * (1 - cluster.x[depot_ind, j, k])
    
    # Time progression for the rest of the nodes in the cluster set (service time is dynamically allocated to find the optimal value) 
    for k in agents:
        for i in NODES:
            for j in NODES:
                if i == j: continue
                source = cluster.original_nodes_dict[i]
                trgt = cluster.original_nodes_dict[j]
                # Arrival at j >= (Arrival at i + Wait at i) + Travel Time from i to j
                model += cluster.t[j, k] >= (cluster.t[i, k] + cluster.service_time[i,k]) + D[(source, trgt)] - M * (1 - cluster.x[i, j, k])

    # Get the return step when the agent completes its journey  and recharges at the depot. 
    for k in agents:
        for i in NODES:
            source = cluster.original_nodes_dict[i]

            # Return to depot >= (Arrival at last node i + Wait at i) + Travel Time to depot
            model += cluster.return_step[k] >= (cluster.t[i, k] + cluster.service_time[i, k]) + D[(source, dept)] - M * (1 - cluster.x[i, depot_ind, k])
    
    # Each agent starts its journey at full battery capacity 
    M_energy = builder.max_battery
    for k in agents: 
        model += cluster.e[depot_ind, k] == builder.max_battery 


    # Energy Update after every arc traveled (uses the appropriate energy at every time step)  
    for k in agents: 
        for i in V_nodes: 
            for j in NODES: 
                if i == j : continue 
                
                source = cluster.original_nodes_dict[i]
                target = cluster.original_nodes_dict[j]
                if source in cluster.virtual_nodes: 
                    source = cluster.virtual_nodes[source] 
                if target in cluster.virtual_nodes: 
                    target = cluster.virtual_nodes[target]

                energy_cost = builder.move_energy[source][target]
                total_energy_cost = energy_cost
                
                if i != depot_ind: 
                    total_energy_cost += wait_energy_consumption * cluster.service_time[i, k]
                
                model += cluster.e[j, k] <= cluster.e[i, k] - total_energy_cost + M_energy * (1 - cluster.x[i, j, k])

    # The energy required for the next travel has to be available at the current node. Otherwise no travel is allowed 
    for k in agents:
        for i in V_nodes:
            for j in V_nodes:
                if i == j: continue
                
                source = cluster.original_nodes_dict[i]
                target = cluster.original_nodes_dict[j] 
                if source in cluster.virtual_nodes: 
                    source = cluster.virtual_nodes[source] 
                if target in cluster.virtual_nodes: 
                    target = cluster.virtual_nodes[target]

                energy_cost = builder.move_energy[source][target]
                
                total_energy_cost = energy_cost
                if i != depot_ind:
                    total_energy_cost += wait_energy_consumption * cluster.service_time[i, k]
                
                # Energy at i must be >= the energy needed for the next leg
                model += cluster.e[i, k] >= total_energy_cost - M_energy * (1 - cluster.x[i, j, k])

    # Work balancing constraint (similar amount of nodes visited per agent) 
    visit_counts = [pl.lpSum(cluster.visit[i,k] for i in NODES) for k in agents]
    fairness_tolerance = int(getattr(builder, "fairness_tolerance", 2))
    for i in range(len(visit_counts)):
        for j in range(i +1, len(visit_counts)):
            model += visit_counts[i] - visit_counts[j] <= fairness_tolerance
            model += visit_counts[j] - visit_counts[i] <= fairness_tolerance

    # Get the makespan for this problem formulation 
    for k in agents:
        model += cluster.makespan >= cluster.return_step[k], f"makespan_constraint_{k}"
    
    # Set the bound of service time, which have to consider that the agent has to at least 
    # provide the mandatory wait time but also not overexceed the time frame
    for k in agents: 
        for j in NODES: 
            model += cluster.service_time[j,k] >= MANDATORY_WAIT_TIME * cluster.visit[j,k]
            model += cluster.service_time[j,k] <= MANDATORY_WAIT_TIME * 2 * cluster.visit[j,k]

    # Calculate the total data rate that all agents can achieve through all areas 
    cluster.total_data_collected_main = pl.lpSum(
            cluster.visit[j, k] * data_per_visit.get(j, 0)
            for j in cluster.NODES 
            for k in cluster.employed_agents
        )


    
def individual_scenario_constraints(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict, subtour_strategy: str = "mtz"):
    """
    This function sets all the constraints (Spatial flow, Energy constraints, Time progression, synchronization and path completion) 
    NOTE: No objective function is enforced here. Constraints are mathematical inequalities and are linear. This scenario 
          forces that the agents, visit all nodes individually and considers collision avoidance for their routes.  
    """
 
    depot_ind, NODES, agents, model, D, TF, MANDATORY_WAIT_TIME, wait_energy_consumption, data_per_visit, bridge_nodes, reverse_nodes = configuration_set_up(cluster=cluster, builder=builder, V_nodes=V_nodes, list_of_agents=list_of_agents)
    
    # For indivual paths the agents have to pass through the bridge nodes a certain amount of times 
    # NOTE: These visits are determined by the central hubs mechanism as are the hub/bridges. 
    for k in agents: 
        for j in NODES: 
            if cluster.nodes_dict[j] in bridge_nodes: continue 
            model += cluster.visit[j,k] == 1 

    # Pass through the bridge node the appropriate amount of times 
    hubs = set(cluster.virtual_nodes.values())
    for k in agents: 
        for hub in hubs: 
            virtual_copies = [vn for vn ,h in cluster.virtual_nodes.items() if h == hub] 
            model += pl.lpSum(cluster.visit[reverse_nodes[v_node],k] for v_node in virtual_copies) == cluster.allowed_visits[hub]
    
    # Each agent must pass through all nodes the appropriate amount of times 
    for j in NODES:
        for k in agents: 
            model += pl.lpSum(cluster.x[i,j,k] for i in V_nodes) == cluster.visit[j,k]
            model += pl.lpSum(cluster.x[j,i,k] for i in V_nodes) == cluster.visit[j,k]
   
    # Start & End the journey at the depot node 
    for k in agents: 
        model += pl.lpSum(cluster.x[depot_ind,i,k] for i in NODES) == 1
        model += pl.lpSum(cluster.x[i,depot_ind,k] for i in NODES) == 1 

    # Spatial flow conservation (No loops) 
    for i in V_nodes: 
        for j in V_nodes: 
            for k in agents: 
                model += cluster.x[i,j,k] + cluster.x[j,i,k] <= 1 

    # Path continuity flow 
    for i in NODES: 
        for k in agents: 
            model += pl.lpSum(cluster.x[i,j,k] for j in V_nodes) == pl.lpSum(cluster.x[j,i,k] for j in V_nodes)

    apply_subtour_constraints(cluster, agents, NODES, depot_ind, subtour_strategy)
   
    # Collision avoidance defined by precedence (to visit a node, an agent must ensure that the other agent has entered, serviced and left the next node) 
    M = TF[-1]
    for j in NODES: 

        if cluster.nodes_dict[j] in bridge_nodes: continue 
        
        for k1 in agents: 
            for k2 in agents: 
                if k1 >= k2 : continue 

                model += cluster.t[j,k2] >= (cluster.t[j,k1] + cluster.service_time[j,k1]) - M * (1 - cluster.precedes[j,k1,k2]) 

                model += cluster.t[j,k1] >= (cluster.t[j,k2] + cluster.service_time[j,k2]) - M * (cluster.precedes[j,k1,k2])
    
    # NOTE: The solver has to choose a precesed value for each node conflict, which in turn forces the arrival time variables to be spaced-out, thus 
    # preventing collisions.  

    # Time progression depot exclusion 
    original_depot_ind = get_depot_node(cluster.depot_id, cluster.original_nodes_dict)
    dept = cluster.original_nodes_dict[original_depot_ind]
    for k in agents:
        agent_release_time = float(getattr(cluster, "agent_start_times", {}).get(k, 0.0))
        model += cluster.start_step[k] >= agent_release_time, f"depot_release_lb_{cluster.id}_{k}"
        model += cluster.start_step[k] <= agent_release_time, f"depot_release_ub_{cluster.id}_{k}"

    for k in agents:
        for j in NODES:
            target = cluster.original_nodes_dict[j]
            # Arrival at first node >= mission start at depot + travel time.
            model += cluster.t[j, k] >= (cluster.start_step[k] + D[(dept, target)]) - M * (1 - cluster.x[depot_ind, j, k])

    # Time progression for the rest of the nodes inside the cluster set
    for k in agents:
        for i in NODES:
            for j in NODES:
                if i == j: continue
                source = cluster.original_nodes_dict[i]
                trgt = cluster.original_nodes_dict[j]
                # Arrival at j >= (Arrival at i + Wait at i) + Travel Time from i to j
                model += cluster.t[j, k] >= (cluster.t[i, k] + cluster.service_time[i,k]) + D[(source, trgt)] - M * (1 - cluster.x[i, j, k])

    # Get the return_step when the agent completes its assigned mission 
    for k in agents:
        for i in NODES:
            source = cluster.original_nodes_dict[i]

            # Return to depot >= (Arrival at last node i + Wait at i) + Travel Time to depot
            model += cluster.return_step[k] >= (cluster.t[i, k] + cluster.service_time[i, k]) + D[(source, dept)] - M * (1 - cluster.x[i, depot_ind, k])

    # All agents start their journey with full battery capacity
    M_energy = builder.max_battery
    for k in agents: 
        model += cluster.e[depot_ind, k] == builder.max_battery 

    # Energy Update after every travel for each agent
    for k in agents: 
        for i in V_nodes: 
            for j in NODES: 
                if i == j : continue 
                
                source = cluster.original_nodes_dict[i]
                target = cluster.original_nodes_dict[j]
                if source in cluster.virtual_nodes: 
                    source = cluster.virtual_nodes[source] 
                if target in cluster.virtual_nodes: 
                    target = cluster.virtual_nodes[target]

                energy_cost = builder.move_energy[source][target]
                total_energy_cost = energy_cost
                
                if i != depot_ind: 
                    total_energy_cost += wait_energy_consumption * cluster.service_time[i,k]
                
                model += cluster.e[j, k] <= cluster.e[i, k] - total_energy_cost + M_energy * (1 - cluster.x[i, j, k])


    # Energy to move and service the next node must be less than the energy available to agent at current node
    for k in agents:
        for i in V_nodes:
            for j in V_nodes:
                if i == j: continue
                
                source = cluster.original_nodes_dict[i]
                target = cluster.original_nodes_dict[j]
                if source in cluster.virtual_nodes: 
                    source = cluster.virtual_nodes[source] 
                if target in cluster.virtual_nodes: 
                    target = cluster.virtual_nodes[target]

                energy_cost = builder.move_energy[source][target]
                
                total_energy_cost = energy_cost
                if i != depot_ind:
                    total_energy_cost += wait_energy_consumption * cluster.service_time[i,k]
                
                # Energy at i must be >= the energy needed for the next leg
                model += cluster.e[i, k] >= total_energy_cost - M_energy * (1 - cluster.x[i, j, k])
    
    # Calculate the makespan of the cluster
    for k in agents: 
        model += cluster.makespan >= cluster.return_step[k] 

    # Bound the service time to include the mandatory visit time and not exceed the time frame for cluster completion
    for k in agents: 
        for j in NODES: 
            model += cluster.service_time[j,k] >= MANDATORY_WAIT_TIME * cluster.visit[j,k]
            model += cluster.service_time[j,k] <= MANDATORY_WAIT_TIME * 2 * cluster.visit[j,k]
    
    # Calculate the total achievable data for every agent
    cluster.total_data_collected_main = pl.lpSum(
            cluster.visit[j, k] * data_per_visit.get(j, 0)
            for j in cluster.NODES 
            for k in cluster.employed_agents
        )
    

def configuration_set_up(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    
    NODES = cluster.NODES
    MANDATORY_WAIT_TIME = builder.coverage_time
    D = cluster.tr_times
    TF = cluster.timeframe
    
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    agents = cluster.employed_agents
    model = cluster.problem 
    wait_energy_consumption = builder.average_coverage_energy
    data_per_visit = {node:rate * builder.coverage_time for node, rate in cluster.R.items()}

    if builder.scenario == 'cooperative':
        return depot_ind, NODES, agents, model, D, TF, MANDATORY_WAIT_TIME, wait_energy_consumption, data_per_visit

    elif builder.scenario == 'individual':
        bridge_nodes = set(cluster.virtual_nodes.keys()) 
        reverse_nodes = {v: k for k, v in cluster.nodes_dict.items()}

        return depot_ind, NODES, agents, model, D, TF, MANDATORY_WAIT_TIME, wait_energy_consumption, data_per_visit, bridge_nodes, reverse_nodes

    else: 
        return depot_ind, NODES, agents, model, D, TF, MANDATORY_WAIT_TIME, wait_energy_consumption, data_per_visit
