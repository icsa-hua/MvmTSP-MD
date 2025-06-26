from dummy_app.tools.logger import logger 

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


def cooperative_scenario_constraints(cluster:Any, builder: Any, V_nodes:list, list_of_agents:dict): 

    depot_ind, NODES, agents, model, D, TF, MANDATORY_WAIT_TIME, wait_energy_consumption, data_per_visit = configuration_set_up(cluster=cluster, builder=builder, V_nodes=V_nodes, list_of_agents=list_of_agents)

    for j in NODES:
        model += pl.lpSum(cluster.visit[j,k] for k in agents) == 1

    for j in NODES:
        for k in agents: 
            model += pl.lpSum(cluster.x[i,j,k] for i in V_nodes) == cluster.visit[j,k]
            model += pl.lpSum(cluster.x[j,i,k] for i in V_nodes) == cluster.visit[j,k]

    for k in agents: 
        model += pl.lpSum(cluster.x[depot_ind,i,k] for i in NODES) == 1
        model += pl.lpSum(cluster.x[i,depot_ind,k] for i in NODES) == 1 

    for i in V_nodes: 
        for j in V_nodes: 
            for k in agents: 
                model += cluster.x[i,j,k] + cluster.x[j,i,k] <= 1 

    for k in agents: 
        # this constraint won't function for larger agent population cases where only one location is feaisble. >=2 change to >= 1
        model += pl.lpSum(cluster.visit[j,k] for j in NODES) >= 1
        model += cluster.u[k] == pl.lpSum(cluster.visit[j,k] for j in NODES)
    
    # Path continuity flow 
    for i in NODES: 
        for k in agents: 
            model += pl.lpSum(cluster.x[i,j,k] for j in V_nodes) == pl.lpSum(cluster.x[j,i,k] for j in V_nodes)

    # At any given time agent k can only be on 1 travel . No overlap 
      
    # MTZ 
    n = len(NODES)
    for k in agents:
        for i in NODES: 
            for j in NODES: 
                if i == j: continue 
                model += cluster.p[i,k] - cluster.p[j,k] + n*cluster.x[i,j,k] <= n - 1

    for k in agents: 
        model += cluster.p[depot_ind,k] == 0

    M = TF[-1]
    original_depot_ind = get_depot_node(cluster.depot_id, cluster.original_nodes_dict)
    dept = cluster.original_nodes_dict[original_depot_ind]

    for k in agents:
        for j in NODES:
            target = cluster.original_nodes_dict[j]
            # Arrival at first node >= (Time at Depot + Wait at Depot) + Travel Time
            # Assuming no wait time at the depot itself before starting the tour.
            model += cluster.t[j, k] >= (0 + D[(dept, target)]) - M * (1 - cluster.x[depot_ind, j, k])

    for k in agents:
        for i in NODES:
            for j in NODES:
                if i == j: continue
                source = cluster.original_nodes_dict[i]
                trgt = cluster.original_nodes_dict[j]
                # Arrival at j >= (Arrival at i + Wait at i) + Travel Time from i to j
                model += cluster.t[j, k] >= (cluster.t[i, k] + cluster.service_time[i,k]) + D[(source, trgt)] - M * (1 - cluster.x[i, j, k])

    for k in agents:
        for i in NODES:
            source = cluster.original_nodes_dict[i]

            # Return to depot >= (Arrival at last node i + Wait at i) + Travel Time to depot
            model += cluster.return_step[k] >= (cluster.t[i, k] + cluster.service_time[i, k]) + D[(source, dept)] - M * (1 - cluster.x[i, depot_ind, k])


    M_energy = builder.max_battery
    for k in agents: 
        model += cluster.e[depot_ind, k] == builder.max_battery 

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
                    total_energy_cost += wait_energy_consumption 
                
                model += cluster.e[j, k] <= cluster.e[i, k] - total_energy_cost + M_energy * (1 - cluster.x[i, j, k])

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
                    total_energy_cost += wait_energy_consumption
                
                # Energy at i must be >= the energy needed for the next leg
                model += cluster.e[i, k] >= total_energy_cost - M_energy * (1 - cluster.x[i, j, k])


    visit_counts = [pl.lpSum(cluster.visit[i,k] for i in NODES) for k in agents]
    for i in range(len(visit_counts)):
        for j in range(i +1, len(visit_counts)):
            model += visit_counts[i] - visit_counts[j] <= 2 
            model += visit_counts[j] - visit_counts[i] <= 2

    for k in agents:
        model += cluster.makespan >= cluster.return_step[k], f"makespan_constraint_{k}"
         
    for k in agents: 
        for j in NODES: 
            model += cluster.service_time[j,k] >= MANDATORY_WAIT_TIME * cluster.visit[j,k]
            model += cluster.service_time[j,k] <= M * cluster.visit[j,k]

    cluster.total_data_collected_main = pl.lpSum(
            cluster.visit[j, k] * data_per_visit.get(j, 0)
            for j in cluster.NODES 
            for k in cluster.employed_agents
        )
    
    
def individual_scenario_constraints(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    
    depot_ind, NODES, agents, model, D, TF, MANDATORY_WAIT_TIME, wait_energy_consumption, data_per_visit, bridge_nodes, reverse_nodes = configuration_set_up(cluster=cluster, builder=builder, V_nodes=V_nodes, list_of_agents=list_of_agents)
    
    for k in agents: 
        for j in NODES: 
            if cluster.nodes_dict[j] in bridge_nodes: continue 
            model += cluster.visit[j,k] == 1 

    hubs = set(cluster.virtual_nodes.values())
    for k in agents: 
        for hub in hubs: 
            virtual_copies = [vn for vn ,h in cluster.virtual_nodes.items() if h == hub] 
            model += pl.lpSum(cluster.visit[reverse_nodes[v_node],k] for v_node in virtual_copies) == cluster.allowed_visits[hub]

    for j in NODES:
        for k in agents: 
            model += pl.lpSum(cluster.x[i,j,k] for i in V_nodes) == cluster.visit[j,k]
            model += pl.lpSum(cluster.x[j,i,k] for i in V_nodes) == cluster.visit[j,k]

    for k in agents: 
        model += pl.lpSum(cluster.x[depot_ind,i,k] for i in NODES) == 1
        model += pl.lpSum(cluster.x[i,depot_ind,k] for i in NODES) == 1 

    for i in V_nodes: 
        for j in V_nodes: 
            for k in agents: 
                model += cluster.x[i,j,k] + cluster.x[j,i,k] <= 1 

    # Redundant since all nodes visited is enforced. 
    # for k in agents: 
    #     # this constraint won't function for larger agent population cases where only one location is feaisble. >=2 change to >= 1
    #     model += pl.lpSum(cluster.visit[j,k] for j in NODES) >= 1
    #     model += cluster.u[k] == pl.lpSum(cluster.visit[j,k] for j in NODES)

    # Path continuity flow 
    for i in NODES: 
        for k in agents: 
            model += pl.lpSum(cluster.x[i,j,k] for j in V_nodes) == pl.lpSum(cluster.x[j,i,k] for j in V_nodes)
    # At any given time agent k can only be on 1 travel . No overlap 

    # MTZ 
    n = len(NODES)
    for k in agents:
        for i in NODES: 
            for j in NODES: 
                if i == j: continue 
                model += cluster.p[i,k] - cluster.p[j,k] + n*cluster.x[i,j,k] <= n - 1

    for k in agents: 
        model += cluster.p[depot_ind,k] == 0

    M = TF[-1]

    for j in NODES: 

        if cluster.nodes_dict[j] in bridge_nodes: continue 
        for k1 in agents: 
            for k2 in agents: 
                if k1 >= k2 : continue 

                model += cluster.t[j,k2] >= (cluster.t[j,k1] + cluster.service_time[j,k1]) - M * (1 - cluster.precedes[j,k1,k2]) 

                model += cluster.t[j,k1] >= (cluster.t[j,k2] + cluster.service_time[j,k2]) - M * (cluster.precedes[j,k1,k2])

    # Solver has to choose a precesed value for each node conflict, which in turn forces the arrival time variables to be spaced-out, thus 
    # preventing collisions.  

    original_depot_ind = get_depot_node(cluster.depot_id, cluster.original_nodes_dict)
    dept = cluster.original_nodes_dict[original_depot_ind]

    for k in agents:
        for j in NODES:
            target = cluster.original_nodes_dict[j]
            # Arrival at first node >= (Time at Depot + Wait at Depot) + Travel Time
            # Assuming no wait time at the depot itself before starting the tour.
            model += cluster.t[j, k] >= (0 + D[(dept, target)]) - M * (1 - cluster.x[depot_ind, j, k])


    for k in agents:
        for i in NODES:
            for j in NODES:
                if i == j: continue
                source = cluster.original_nodes_dict[i]
                trgt = cluster.original_nodes_dict[j]
                # Arrival at j >= (Arrival at i + Wait at i) + Travel Time from i to j
                model += cluster.t[j, k] >= (cluster.t[i, k] + cluster.service_time[i,k]) + D[(source, trgt)] - M * (1 - cluster.x[i, j, k])

    for k in agents:
        for i in NODES:
            source = cluster.original_nodes_dict[i]

            # Return to depot >= (Arrival at last node i + Wait at i) + Travel Time to depot
            model += cluster.return_step[k] >= (cluster.t[i, k] + cluster.service_time[i, k]) + D[(source, dept)] - M * (1 - cluster.x[i, depot_ind, k])


    M_energy = builder.max_battery
    for k in agents: 
        model += cluster.e[depot_ind, k] == builder.max_battery 

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
                    total_energy_cost += wait_energy_consumption 
                
                model += cluster.e[j, k] <= cluster.e[i, k] - total_energy_cost + M_energy * (1 - cluster.x[i, j, k])

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
                    total_energy_cost += wait_energy_consumption
                
                # Energy at i must be >= the energy needed for the next leg
                model += cluster.e[i, k] >= total_energy_cost - M_energy * (1 - cluster.x[i, j, k])
    
    for k in agents: 
        model += cluster.makespan >= cluster.return_step[k] 

    for k in agents: 
        for j in NODES: 
            model += cluster.service_time[j,k] >= MANDATORY_WAIT_TIME * cluster.visit[j,k]
            model += cluster.service_time[j,k] <= M * cluster.visit[j,k]
    
    cluster.total_data_collected_main = pl.lpSum(
            cluster.visit[j, k] * data_per_visit.get(j, 0)
            for j in cluster.NODES 
            for k in cluster.employed_agents
        )
    

def configuration_set_up(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    NODES = cluster.NODES
    agents = cluster.employed_agents
    model = cluster.problem 
    D = cluster.tr_times
    TF = cluster.timeframe
    MANDATORY_WAIT_TIME = builder.coverage_time
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