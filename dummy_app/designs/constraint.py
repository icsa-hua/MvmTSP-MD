import pulp as pl 
import numpy as np 
from collections import defaultdict
from typing import Any
from dummy_app.tools.logger import logger 



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


def constraint_0(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Allow multiple visits (essential for MV-TSP)
    # """

    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    # Correct - But try to match == 
    for k, v in list_of_agents.items(): 
        for j in V_nodes: 
            if cluster.nodes_dict[j] not in cluster.bridge_nodes: continue
            cluster.problem.addConstraint(
                name=f"Allowed_visits_for_each_agent_{k}_for_node_{j}", 
                constraint = pl.lpSum(cluster.x[i,j,v] for i in V_nodes if i != j) <= cluster.R_points[j],
            )

    # Correct but may lack clarification due to duration 
    for j in V_nodes: 
        cluster.problem += pl.lpSum(
            cluster.t[i,j,v,t]
            for i in V_nodes if i != j and i!=depot_ind
            for v in list_of_agents.values() 
            for t in cluster.timeframe
        ) >= cluster.R_points[j]*len(cluster.employed_agents), f"Allowed_visits_for_each_agent_for_node_{j}"

    logger.debug(f"Constraint 0: {len(cluster.problem.constraints)}")


def constraint_1(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Only allow a single travel from depot to all nodes and the reverse as well.
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    for k, v in list_of_agents.items(): 

        # Correct
        cluster.problem.addConstraint(
            name=f"{k}_enters_single_area_from_depot_{depot_ind}",
            constraint = pl.lpSum(cluster.x[depot_ind, j, v] for j in V_nodes if j != depot_ind ) == 1,
        )

        # Correct
        cluster.problem.addConstraint(
            name=f"{k}_leaves_single_area_to_depot_{depot_ind}", 
            constraint = pl.lpSum(cluster.x[i, depot_ind, v] for i in V_nodes if i != depot_ind ) == 1
        )

    # Correct 
    for j in V_nodes: 
        if j == depot_ind: continue 
        cluster.problem.addConstraint(
            name=f"Only_one_depot_to_j_for_node_{j}",
            constraint= pl.lpSum(cluster.x[depot_ind,j,v] for v in list_of_agents.values() ) <= 1,
        )

    # Correct 
    # cluster.problem.addConstraint(
    #     name=f"exactly_{len(cluster.employed_agents)}_depart_from_depot",
    #     constraint=pl.lpSum(cluster.x[depot_ind,i,v] for i in V_nodes for v in list_of_agents.values()) == len(cluster.employed_agents)
    # )

    # # Correct 
    # cluster.problem.addConstraint(
    #     name=f"exactly_{len(cluster.employed_agents)}_return_to_depot",
    #     constraint=pl.lpSum(cluster.x[i,depot_ind,v] for i in V_nodes for v in list_of_agents.values()) == len(cluster.employed_agents)
    # )

    # # Correct 
    # for j in V_nodes: 
    #     if j == depot_ind: continue
    #     cluster.problem.addConstraint(
    #         name=f"Only_a_single_j_to_depot_for_node_{j}",
    #         constraint= pl.lpSum(cluster.x[j,depot_ind,v] for v in list_of_agents.values() ) <= 1,
    #     )
    # Correct 
    cluster.problem.addConstraint(
        name=f"exactly_{len(cluster.employed_agents)}_return_to_depot",
        constraint=pl.lpSum(cluster.x[i,depot_ind,v] for i in V_nodes for v in list_of_agents.values()) == len(cluster.employed_agents)
    )

    # Correct 
    for j in V_nodes: 
        if j == depot_ind: continue
        cluster.problem.addConstraint(
            name=f"Only_a_single_j_to_depot_for_node_{j}",
            constraint= pl.lpSum(cluster.x[j,depot_ind,v] for v in list_of_agents.values() ) <= 1,
        )

    logger.debug(f"Constraint 1: {len(cluster.problem.constraints)}")        


def constraint_2(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Allow an agent to start his journey whenever it fits best and finish as well at a different time that before (not time frame [-1])
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    for k, v in list_of_agents.items(): 

        # # # Correct
        cluster.problem.addConstraint(
            name=f"{k}_leaves_depot_{depot_ind}_at_specific_interval", 
            constraint=pl.lpSum(cluster.t[depot_ind, j, v, cluster.timeframe[0]] for j in V_nodes if depot_ind != j ) == 1,
        )
        
        # # # Correct 
        cluster.problem.addConstraint(
            name=f"{k}_return_depot_{depot_ind}_at_dynamic_interval",
            # constraint = pl.lpSum(cluster.y_depart[i, depot_ind, v, t]for i in V_nodes if i != depot_ind for t in cluster.timeframe if t + cluster.tr_times[(i,depot_ind)] <= cluster.timeframe[-1]) == 1,
            constraint = pl.lpSum(cluster.t[i, depot_ind, v, cluster.timeframe[-1]] for i in V_nodes if i != depot_ind ) == 1, 
        )

        # for i in V_nodes: 
        #     for j in V_nodes: 
        #         if i == j : continue 
        #         cluster.problem += pl.lpSum(cluster.t[i,j,v,t] for t in cluster.timeframe) <= cluster.max_durations[(i,j)]
        #         cluster.problem += pl.lpSum(cluster.t[i,j,v,t] for t in cluster.timeframe) >= 1 
        
        # for t in cluster.timeframe: 
        #     cluster.problem += pl.lpSum(
        #         cluster.t[i,depot_ind,v,t] for i in V_nodes if i != depot_ind 
        #     ) <= 1 
  
    logger.debug(f"Constraint 2: {len(cluster.problem.constraints)}")
    

def constraint_3(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Enforce that only a single journey from i -> j exists so that the reverse is not possible (j -> i) 
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    # Correct
    for k, v in list_of_agents.items(): 
        cluster.problem.addConstraint(
            name=f"{k}_start_&_finishes_at_depot_{depot_ind}", 
            constraint = pl.lpSum(cluster.x[depot_ind, j, v] for j in V_nodes if j != depot_ind ) + \
                         pl.lpSum(cluster.x[i, depot_ind, v] for i in V_nodes if i != depot_ind ) == 2
        )

    logger.debug(f"Constraint 3: {len(cluster.problem.constraints)}")
            

def constraint_4(cluster:Any, builder:Any, V_nodes:list, list_of_agents:dict):
    """
    Prohibit Depot looping for each agent.
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    # Correct 
    for k, v in list_of_agents.items():
        cluster.problem.addConstraint(
            name=f"No_loop_at depot_{depot_ind}_for_{k}_at_any_timepoint",
            constraint=cluster.x[depot_ind, depot_ind, v] == 0,  
        )

        for j in V_nodes: 
            if j == depot_ind: continue 
            cluster.problem.addConstraint(
                name = f"No_travel_with_a single_node_{j}_allowed_{k}",
                constraint = cluster.x[depot_ind,j,v] + cluster.x[j,depot_ind,v] <= 1
            )

    logger.debug(f"Constraint 4: {len(cluster.problem.constraints)}")


def constraint_5(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Only one visit from i to j and from j to i except for bridge nodes. We don't exclude the depots here as they also are a 1 out 1 in node. 
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    for k, v in list_of_agents.items(): 
        for i in V_nodes:

            # Correct
            cluster.problem.addConstraint(
                    name=f"Only_one_visit_from_i_to_j_for_agent_{k}_for_node_{i}",
                    constraint= pl.lpSum(cluster.x[i,j,v] for j in V_nodes if i != j) == 1,
                )
            
            # Correct 
            if cluster.nodes_dict[i] in cluster.bridge_nodes :
                cluster.problem.addConstraint(
                    name=f"Bridge_NODES_from_i_to_j_for_agent_{k}_for_node_{i}",
                    constraint=pl.lpSum(cluster.x[i,j,v] for j in V_nodes if i != j) <= cluster.R_points[i],
                )

        for j in V_nodes:
            cluster.problem.addConstraint(
                name=f"Only_one_visit_from_j_to_i_for_agent_{k}_for_node_{j}",
                constraint= pl.lpSum(cluster.x[i,j,v] for i in V_nodes if i != j ) == 1,
            )

            if cluster.nodes_dict[j] in cluster.bridge_nodes :
                cluster.problem.addConstraint(
                    name=f"Bridge_NODES_from_j_to_i_for_agent_{k}_for_node_{j}",
                    constraint= pl.lpSum(cluster.x[i,j,v] for i in V_nodes if i != j) <= cluster.R_points[j],
                )
                

    # Enforce Full Cluster Coverage (Coolectively)
    # if builder.scenario == "cooperative": 
    #     for j in V_nodes:
    #         if cluster.nodes_dict[j] in cluster.bridge_nodes : continue 
    #         cluster.problem += (
    #             pl.lpSum(cluster.x[i, j, v] for i in V_nodes for v in list_of_agents.values() if i != j) >= len(list_of_agents),
    #             f"At_least_one_agent_visit{j}"
    #         )

    #     for j in V_nodes:
    #         if cluster.nodes_dict[j] in cluster.bridge_nodes : continue 
    #         cluster.problem += (
    #             pl.lpSum(cluster.x[j,i,v] for i in V_nodes for v in list_of_agents.values() if i != j) >= len(list_of_agents),
    #             f"At_least_one_agent_exit_{j}"
    #         )


    logger.debug(f"Constraint 5: {len(cluster.problem.constraints)}")


def constraint_6(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Collision avoidance / Unique agent per node 
    """    
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    
    
    for i in V_nodes: 
        for j in V_nodes: 
            if j == depot_ind or cluster.nodes_dict[j] in cluster.bridge_nodes: continue
            if j == depot_ind or cluster.nodes_dict[j] in cluster.bridge_nodes: continue
            for step in cluster.timeframe: 
               cluster.problem.addConstraint(
                   name=f"Unique_Time_visits_constraint_at_travel_{i}_{j}_at_time_{step}",
                   constraint= pl.lpSum(cluster.t[i,j,v,step] for _,v in list_of_agents.items()) <= 1
               ) 

    logger.debug(f"Constraint 6: {len(cluster.problem.constraints)}")


def constraint_7(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    
    # NOTE: This constraint is used to count the number of visits to each node by each agent.
    for k, v in list_of_agents.items():
        for j in V_nodes:
            cluster.problem += (
                cluster.visit_miss[j, v] >= 1 - pl.lpSum(cluster.x[i, j, v] for i in V_nodes if i != j),
                f"Miss_indicator_for_agent_{k}_node_{j}"
            )
    logger.debug(f"Constraint 7: {len(cluster.problem.constraints)}")


def constraint_8(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Synchronization between the time and space 
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    # Suggestion don't use the T_MAX as it prevents later steps from being checked while it should check feasibility over the full timeline, not against the max travel time to the depot. 
    # Missing depot transitions which may be intentional - but it leaves depot timing uncontrolled unless differenct synchronization constraint is used. 
    # for k, v in list_of_agents.items(): 
    #     for i in V_nodes: 
    #         for j in V_nodes: 
    #             if i == j : continue 
    #             # if i == depot_ind or j == depot_ind: continue 
    #             d = cluster.tr_times[(i,j)]
    #             step = 0
    #             while step < len(cluster.timeframe)-d+1: 
    #                     cluster.problem.addConstraint(
    #                         name=f"Synchronization_time_and_space_{i,j}_for_agent_{k}_for_{step}",
    #                         constraint= pl.lpSum(
    #                                         cluster.t[i,j,v,cluster.timeframe[t]] for t in range(step, step + d)
    #                                     ) == d*cluster.x[i,j,v],
    #                     )
    #                     step += d
            
    # BIG_M = len(V_nodes)
    # for k, v in list_of_agents.items(): 
    #     for i in V_nodes: 
    #         for j in V_nodes: 
    #             if i == j : continue 
                
    #             cluster.problem += pl.lpSum(cluster.t[i,j,v,t] for t in cluster.timeframe) == cluster.x[i,j,v]

    #             for t in cluster.timeframe:
    #                 cluster.problem += (
    #                     cluster.arrival_time[j,v] >= cluster.arrival_time[i,v] + cluster.tr_times[(i,j)] - BIG_M*(1 - cluster.t[i,j,v,t])
    #                 )
    # T = cluster.timeframe 
    # for v in list_of_agents.values(): 
    #     for t in T : 
            
    #         if t == 0 : 
    #             for i in V_nodes: 
    #                 cluster.problem += (cluster.t[depot_ind,i,v,t] ) == cluster.x[depot_ind, i, v]
    #         if t == T[-1]: 
    #             for i in V_nodes: 
    #                 cluster.problem += (cluster.t[i, depot_ind,v,t] ) == cluster.x[i, depot_ind, v]
            

    
    for i in V_nodes: 
        for j in V_nodes: 
            if i == j: continue 
            d = cluster.tr_times[(i,j)]
            for k, v in list_of_agents.items(): 
                cluster.problem += pl.lpSum(
                     cluster.t[i,j,v, step]
                     for step in cluster.timeframe[cluster.tr_times[(depot_ind,i)]: -(cluster.tr_times[(i,j)] + cluster.tr_times[(j,depot_ind)])]
                ) == cluster.x[i,j,v] * d 
    
    logger.debug(f"Constraint 8: {len(cluster.problem.constraints)}")



def constraint_9(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Only one travel from i to j for the entirety of the time frame. This cannotbe used if we opt to align all time steps individually. 
    """

    for k, v in list_of_agents.items(): 

        if builder.scenario == 'coverage': 
            cluster.problem += (
                cluster.route_time[v] == pl.lpSum(
                    cluster.tr_times[(i,j)] * cluster.x[i,j,v]
                    for i in V_nodes for j in V_nodes if i!=j 
                ) + builder.coverage_time * pl.lpSum(
                    cluster.wait[v,t] 
                    for t in cluster.timeframe)
            )

            cluster.problem += cluster.route_time[v] <= cluster.T_MAX
        elif builder.scenario == 'cooperative': 
            for j in V_nodes: 
                cluster.problem += pl.lpSum(
                    cluster.arrive[j,v,t] for t in cluster.timeframe
                ) -1 <= cluster.y[j,v]

                cluster.problem += cluster.visit_miss[j,v] >= 1 - pl.lpSum(cluster.x[i,j,v] for i in V_nodes if i != j)
    
    logger.debug(f"Constraint 9: {len(cluster.problem.constraints)}")


def constraint_10(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):

    BIG_M = sum(cluster.R_points)

    # Correct For individual case 
    if builder.scenario == "coverage": 

        for k,v in list_of_agents.items():
            for i in V_nodes:
                cluster.problem.addConstraint(
                    name=f"Agent_{v}_must_visit_node_{i}",
                    constraint=cluster.visit[i,v] == 1 
                )

    # For cooperative case 
    elif builder.scenario == "cooperative": 
        for i in V_nodes: 
            cluster.problem += pl.lpSum(cluster.visit[i,v] for v in list_of_agents.values()) >= 1 

    
    for k, v in list_of_agents.items() : 
        for i in V_nodes: 
            cluster.problem += (
                cluster.visit[i,v] <= pl.lpSum(cluster.x[i,j,v] for j in V_nodes if i != j)
            )
            
    # for k, v in list_of_agents.items():
    #     for i in V_nodes:
    #         # Correct (including j != depot breaks the solution)
    #         cluster.problem.addConstraint(
    #             name=f"visit_upper_{i}_{k}",
    #             constraint=cluster.visit[i, v] <= pl.lpSum(cluster.x[i, j, v] for j in V_nodes if i != j )
    #         )

    #         # Correct (the same as above)
    #         cluster.problem.addConstraint(
    #             name=f"visit_lower_{i}_{k}",
    #             constraint=cluster.visit[i, v] >= (1 / BIG_M) * pl.lpSum(cluster.x[i, j, v] for j in V_nodes if i != j ), 
    #         )

    logger.debug(f"Constraint 10: {len(cluster.problem.constraints)}")


def constraint_11(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
   
    visit_counts = [pl.lpSum(cluster.visit[i, v] for i in V_nodes if i != depot_ind and cluster.nodes_dict[i]!=cluster.bridge_nodes) for k, v in list_of_agents.items()]
    for a in visit_counts:
        for b in visit_counts:
            cluster.problem += a <= b + 1  # Keep agent loads balanced
    
    logger.debug(f"Constraint 11: {len(cluster.problem.constraints)}")

    # The != 0 is not linear hence it might not be correctly interpreted by the solver. It is reformulated using binary auxiliary, big-M or indicator constraints. 
    # this also overconstrains the system. 


def constraint_12(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    M = builder.max_battery 
    NODES = V_nodes[:-1]
    agents = cluster.employed_agents
    model = cluster.problem 
    TF = cluster.timeframe 

    for k in agents:
        for i in V_nodes:
            for j in V_nodes:
                if i == j: continue 
                source = cluster.nodes_dict[i]
                source = cluster.nodes_dict[i]
                target = cluster.nodes_dict[j]-1
                energy_cost = builder.move_energy[source][target] 
                
                model += cluster.e[j,k] >= cluster.e[i,k] - energy_cost - M * (1-cluster.x[i,j,k])
                model += cluster.e[i,k] >= energy_cost - M * (1-cluster.x[i,j,k])
            model += cluster.e[i,k] >= builder.move_energy[cluster.nodes_dict[i]][cluster.depot_id] * cluster.x[i,depot_ind,k]


        cov_cost = builder.average_coverage_energy 
        for t in TF: 
            for i in NODES: 

                model += cluster.e[i,k] >= cluster.e[i,k] - cov_cost * cluster.wait[k,t]
                model += cluster.e[i,k] >= cov_cost * cluster.wait[k, t]                

        cov_cost = builder.average_coverage_energy 
        for t in cluster.timeframe: 
            for i in V_nodes: 
                if i == depot_ind: continue 
                # Subtract coverage energy from remaining battery
                cluster.problem.addConstraint(
                    name=f"Update_remaining_energy_comm_{i}_at_{t}_for_{k}",
                    constraint=cluster.e[i, k] >= cluster.e[i, k] - cov_cost * cluster.wait[k, t],
                )

                # Optional: ensure energy is enough before waiting
                cluster.problem.addConstraint(
                    name=f"No_wait_if_low_energy_{i}_at_{t}_for_{k}",
                    constraint=cluster.e[i, k] >= cov_cost * cluster.wait[k, t],
                )

               
    logger.debug(f"Constraint 12: {len(cluster.problem.constraints)}")


def constraint_13(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    M = builder.max_battery 
    NODES = V_nodes[:-1]
    agents = cluster.employed_agents
    model = cluster.problem 
    for k in agents:
        for i in NODES:
            desc_cost = builder.descend_energy.loc[cluster.depot_id, cluster.nodes_dict[i]]

            model += cluster.e[i,k] >= 0
            model += cluster.e[i,k] >= desc_cost *cluster.x[i,depot_ind,k]
            model += cluster.e[depot_ind,k] - M * (1-cluster.x[i,depot_ind,k])

        model += cluster.e[depot_ind,k] == builder.max_battery


    M = builder.max_battery 

    for k, v in list_of_agents.items():
        for i in V_nodes:
            if i == depot_ind: continue
            desc_cost = builder.descend_energy.loc[cluster.depot_id, cluster.nodes_dict[i]]

            cluster.problem.addConstraint(
                name=f"Energy_cannot_be_negative_{i}_for_{k}",
                constraint=cluster.e[i,v] >= 0,
            )

            cluster.problem.addConstraint(
                name=f"Enough_energy_to_descend_at_depot_from_{i}_for_{k}",
                constraint=cluster.e[i,v] >= desc_cost * cluster.x[i, depot_ind, v]
            )

            cluster.problem.addConstraint(
                name=f"No_descend_if_low_energy_{i}_for_{k}",
                constraint=cluster.e[depot_ind, v] - M * (1 - cluster.x[i,depot_ind,v]) >= 0
            )

        cluster.problem.addConstraint(
            name=f"Every_agent_starts_with_full_battery_{k}",
            constraint= cluster.e[depot_ind, v] == builder.max_battery 
        )


    logger.debug(f"Constraint 13: {len(cluster.problem.constraints)}")


def constraint_14(cluster:Any, builder:Any, V_nodes:list, list_of_agents:dict): 
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    M = len(V_nodes)

    for k, v in list_of_agents.items():
        for i in V_nodes: 
            for j in V_nodes: 
                cluster.problem += cluster.p[j,v] >= cluster.p[i,v] + 1 - M*(1 - cluster.x[i,j,v])
        
        # for t in cluster.timeframe:
        #     cluster.problem += cluster.p[v, t] == depot_ind + (1 - (t == cluster.return_step[v])) * M, \
        #         f"Return_alignment_at_time_{t}_for_agent_{k}"
        # cluster.problem.addConstraint(
        #     name=f"Positional_variable_at_start_of_journey_for_{k}",
        #     constraint=cluster.p[v,cluster.timeframe[0]] == depot_ind, 
        # )
        # cluster.problem += cluster.p[v, t] == depot_ind + (1 - pl.lpSum(cluster.t[depot_ind, j, v, t] for j in V_nodes if j != depot_ind)) * M  # Big-M allows flexibility before departure
        
    logger.debug(f"Constraint 14: {len(cluster.problem.constraints)}")


def constraint_15(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    valid_pairs = [(i, j) for i in V_nodes for j in V_nodes if i != j and i != depot_ind and j != depot_ind]
    
    for k, v in list_of_agents.items():

        for i, j in valid_pairs:
            travel_duration = cluster.tr_times[(i, j)]
            max_start = len(cluster.timeframe) - travel_duration 
            for t_start in cluster.timeframe[:max_start]:
                    t_end = t_start + travel_duration 
                    for dt in range(travel_duration):
                        t = t_start + dt 
                        cluster.problem.addConstraint(
                            constraint=cluster.busy[v,t] <= cluster.t[i,j,v,t_start],
                        )
                    if t_end < cluster.timeframe[-1]:
                        cluster.problem.addConstraint(
                            constraint=cluster.wait[v, t_end] >= cluster.t[i, j, v, t_start]
                        )
                    

        # for t in cluster.timeframe:
        #     cluster.problem += cluster.busy[v, t] <= (t <= cluster.return_step[v]), \
        #         f"No_travel_post_return_for_agent_{k}_at_{t}"


    logger.debug(f"Constraint 15: {len(cluster.problem.constraints)}")


def constraint_16(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    for k,v in list_of_agents.items():
        for t in cluster.timeframe:
            cluster.problem.addConstraint(
                name=f"{k}_can_only_be_busy_or_idle_at_{t}",
                constraint=cluster.busy[v, t] + cluster.wait[v, t] == 1
            )
            
            #Suggestion: With the current formulation, the constraint limits co-activation, but does not guarantee at least one is active. 
            # So the agent might be inactive (neither busy nor waiting) unless we add the == instead of <=. 

    logger.debug(f"Constraint 16: {len(cluster.problem.constraints)}")


def constraint_17(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    #Per agent     #Per agent 
    for k, v in list_of_agents.items():
        for j in V_nodes: 
            if j != depot_ind:
                # For departure at zero 
                for step in cluster.timeframe[:cluster.tr_times[(depot_ind, j)]]: 
                    # # cluster.problem.addConstraint(
                    # #     name=f"Dynamic_time_enforcement_{depot_ind}_{j}_for_{k}_time_{step}",
                    # #     constraint=cluster.t[depot_ind, j, v, step] == cluster.t[depot_ind, j, v, cluster.timeframe[0]], 
                    # # )  
                    cluster.problem.addConstraint(
                        name=f"Enforce synchronization_between_t_and_x_{depot_ind}_{j}_for_{k}_time_{step}",
                        constraint=cluster.t[depot_ind, j, v, step] == cluster.x[depot_ind, j, v], 
                    )

        for i in V_nodes: 
            if i != depot_ind:
                # For departure at zero 
                valid_return_window = cluster.timeframe[-(cluster.tr_times[(i, depot_ind)]+1):]

                for step in valid_return_window: 
                    # cluster.problem.addConstraint(
                #         name=f"Dynamic_time_enforcement_{i,depot_ind}__for_{k}_time_{step}",
                #         constraint=cluster.t[i,depot_ind, v, step] == cluster.t[i,depot_ind, v, cluster.timeframe[-1]], 
                #     )  
                    cluster.problem.addConstraint(
                        name=f"Enforce synchronization_between_t_and_x_{i,depot_ind}_for_{k}_time_{step}",
                        constraint=cluster.t[i, depot_ind, v, step] <= cluster.x[i, depot_ind, v], 
                    )
   
    logger.debug(f"Constraint 17: {len(cluster.problem.constraints)}")


def constraint_18(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    # Symmetry-breaking constraint 
    if builder.scenario == "cooperative": 
        for i in range(len(cluster.employed_agents) - 1):
            v1 = cluster.employed_agents[i]
            v2 = cluster.employed_agents[i + 1]
            cluster.problem += pl.lpSum(
                cluster.x[i, j, v1] for i in V_nodes for j in V_nodes if i != j
            ) != pl.lpSum(
                cluster.x[i, j, v2] for i in V_nodes for j in V_nodes if i != j
            ), f"Symmetry_break_by_edge_count_{v1}_vs_{v2}"

    logger.debug(f"Constraint 18: {len(cluster.problem.constraints)}")


def constraint_19(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    cluster.returned = pl.LpVariable.dicts("returned", ((v, t) for v in cluster.employed_agents for t in cluster.timeframe), cat="Binary")


    for k, v in list_of_agents.items():
        for t in cluster.timeframe:
            cluster.problem += cluster.returned[v, t] <= pl.lpSum(
                cluster.t[i, depot_ind, v, tt]
                for i in V_nodes if i != depot_ind
                for tt in cluster.timeframe if tt <= t - cluster.tr_times[(i, depot_ind)]
            )

    cluster.returned = pl.LpVariable.dicts("returned", ((v, t) for v in cluster.employed_agents for t in cluster.timeframe), cat="Binary")


    for k, v in list_of_agents.items():
        for t in cluster.timeframe:
            cluster.problem += cluster.returned[v, t] <= pl.lpSum(
                cluster.t[i, depot_ind, v, tt]
                for i in V_nodes if i != depot_ind
                for tt in cluster.timeframe if tt <= t - cluster.tr_times[(i, depot_ind)]
            )

        for i in V_nodes:
            for j in V_nodes:
                if i == j: continue
                if i == j: continue
                travel_duration = cluster.tr_times[(i, j)]
                for t in cluster.timeframe:
                    if t + travel_duration <= cluster.timeframe[-1]:
                        cluster.problem += cluster.t[i, j, v, t] <= 1 - cluster.returned[v, t]
                        cluster.problem += cluster.depart[i,j,v,t] <= 1 - cluster.returned[v, t]

        for t in cluster.timeframe:
            cluster.problem += cluster.busy[v, t] <= 1 - cluster.returned[v, t]
            cluster.problem += cluster.wait[v, t] <= 1 - cluster.returned[v, t]

    logger.debug(f"Constraint 19: {len(cluster.problem.constraints)}")


def constraint_20(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    # NOTE: May be redundant since 15 also handles wait after busy
    logger.debug(f"Constraint 20: {len(cluster.problem.constraints)}")
    
    
def constraint_21(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    # This is a good approach however a better approach would be to use an integer decision variable 
    # that penalizes every repeat visit not just presence. 
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    for k, v in list_of_agents.items(): 
        for i in V_nodes: 
            for t in cluster.timeframe: 
                if t - 1 < 0 :continue 
                cluster.problem += cluster.arrive[i,v,t] >= cluster.atNode[i,v,t] - cluster.atNode[i,v,t-1]
                # cluster.problem += cluster.arrive[i,v,t] >= cluster.atNode[i,v,cluster.timeframe[0]] 

            # AT node 
            cluster.problem += cluster.atNode[depot_ind, v, cluster.timeframe[0]] == 1 
            for i in V_nodes: 
                if i == depot_ind: continue
                cluster.problem += cluster.atNode[i, v, cluster.timeframe[0]] == 0 

            for j in V_nodes: 
                if j == depot_ind : continue
                cluster.problem += pl.lpSum(cluster.atNode[j,v,t] for t in cluster.timeframe) >= 1


def constraint_22(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    # suggestion : change t to use x so to get rid of time. 
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    for k, v in list_of_agents.items():
        for j in V_nodes: 
            # if j == cluster.depot_id: continue 
            # if cluster.nodes_dict[j] in cluster.bridge_nodes: continue 

            cluster.problem.addConstraint(
                name=f"Flow_conservation_Not_Time_expanded_{k}_{j}", 
                constraint= pl.lpSum(cluster.x[i, j, v] for i in V_nodes if i != j ) == \
                            pl.lpSum(cluster.x[j, l, v] for l in V_nodes if l != j )
                
            )

    logger.debug(f"Constraint 22 : {len(cluster.problem.constraints)}")


def constraint_23(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    out_arcs, _ = get_arcs(V_nodes, cluster.depot_id)
 
    for i in out_arcs: 
        for j in out_arcs[i]: 
            # if cluster.nodes_dict[j] not in cluster.bridge_nodes:
                for k, v in list_of_agents.items(): 
                    cluster.problem.addConstraint(
                        name=f"No_loops_in_path_for_{k}_at_{i}_{j}", 
                        constraint=cluster.x[i,j,v] + cluster.x[j,i,v] <= 1, 
                    )

    logger.debug(f"Constraint 23 : {len(cluster.problem.constraints)}")
      

def constraint_24(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    MIN_VISITS = np.round(len(V_nodes) / len(cluster.employed_agents))

    for k, v in list_of_agents.items(): 
        if builder.scenario == "coverage":
            MIN_VISITS = len(cluster.nodes_dict)-1
            cluster.problem += pl.lpSum(
                cluster.visit[i,v] for i in V_nodes if i != depot_ind
            ) >= MIN_VISITS, f"Min_coverage_for_agent_{k}" 

        elif builder.scenario == "cooperative": 
            cluster.problem += pl.lpSum(
                cluster.visit[i,v] for i in V_nodes if i != depot_ind
            ) >= MIN_VISITS, f"Min_coverage_for_agent_{k}"

        for i in V_nodes:
            if i == depot_ind: continue
            if cluster.nodes_dict[i] in cluster.bridge_nodes: continue
            cluster.problem += cluster.visit[i, v] >= pl.lpSum(cluster.x[i, j, v] for j in V_nodes if j != i), \
                f"Visit_tracking_{i}_agent_{k}"
            
    # for j in V_nodes: 
    #     cluster.problem += pl.lpSum(cluster.x[i,j,v] for i in V_nodes for v in list_of_agents.values()) >= len(cluster.nodes_dict)
            
    logger.debug(f"Constraint 24: {len(cluster.problem.constraints)}")


def constraint_25(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    for k, v in list_of_agents.items():
        cluster.problem += pl.lpSum(
            cluster.depart[depot_ind, j, v, cluster.timeframe[0]]
            for j in V_nodes if j != depot_ind
        ) == 1 

        cluster.problem += pl.lpSum(
            cluster.depart[i, depot_ind, v, t]
            for i in V_nodes if i != depot_ind 
            for t in cluster.timeframe
            if t + cluster.tr_times[(i,depot_ind)] <= cluster.timeframe[-1]
        ) == 1

        for i in V_nodes: 
            for j in V_nodes: 
                if i == j : continue
                cluster.problem += pl.lpSum(
                    cluster.depart[i,j,v,t] for t in cluster.timeframe
                    if t + cluster.tr_times[(i,j)] <= cluster.timeframe[-1] 
                ) == cluster.x[i,j,v]

        for i in V_nodes: 
            for j in V_nodes: 
                if i == j : continue 
                cluster.problem += pl.lpSum(cluster.depart[i,j,v,t] for t in cluster.timeframe) <= 1
        
        cluster.problem += pl.lpSum(
            cluster.depart[depot_ind, j, v, cluster.timeframe[0]]
            for j in V_nodes if j != depot_ind
        ) == 1 

        cluster.problem += pl.lpSum(
            cluster.depart[i, depot_ind, v, t]
            for i in V_nodes if i != depot_ind 
            for t in cluster.timeframe
            if t + cluster.tr_times[(i,depot_ind)] <= cluster.timeframe[-1]
        ) == 1

        for i in V_nodes: 
            for j in V_nodes: 
                if i == j : continue
                cluster.problem += pl.lpSum(
                    cluster.depart[i,j,v,t] for t in cluster.timeframe
                    if t + cluster.tr_times[(i,j)] <= cluster.timeframe[-1] 
                ) == cluster.x[i,j,v]

        for i in V_nodes: 
            for j in V_nodes: 
                if i == j : continue 
                cluster.problem += pl.lpSum(cluster.depart[i,j,v,t] for t in cluster.timeframe) <= 1
        
    logger.debug(f"Constraint 25: {len(cluster.problem.constraints)}")


def constraint_26(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    
    
    # Only set visit when agent arrives at node j after trip i → j completes
    for i in V_nodes:
        for j in V_nodes:
            if i == j or i == depot_ind or j == depot_ind: continue
            trip_time = cluster.tr_times[(i, j)]
            for t_start in cluster.timeframe[:-trip_time]:
                t_arrival = t_start + trip_time
                for k, v in list_of_agents.items():
                    cluster.problem.addConstraint(
                        name=f"Visit_trigger_at_{j}_after_{i}_{t_arrival}_for_{k}",
                        constraint=cluster.visit[j, v] >= cluster.t[i, j, v, t_start]
                    )

    logger.debug(f"Constraint 26: {len(cluster.problem.constraints)}")


def constraint_27(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    for k, v in list_of_agents.items():
        for i in V_nodes:
            for j in V_nodes:
                if i != j:
                    for step in cluster.timeframe:
                        arrival_time = step + cluster.tr_times[(i, j)]
                        # ensure time index exists
                        if arrival_time + 1 in cluster.timeframe:
                            cluster.problem += cluster.t[i, j, v, arrival_time] <= cluster.t[j, j, v, arrival_time + 1], \
                                f"TimeProgress_{i}_{j}_at_{step}_agent_{k}"  
    logger.debug(f"Constraint 27 : {len(cluster.problem.constraints)}")


def constraint_28(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)


    for k, v in list_of_agents.items():

        u = pl.LpVariable.dicts(f"u{k}", ((i) for i in V_nodes if i != depot_ind), lowBound=1, upBound=len(V_nodes)-1, cat='Integer')

        n = len(V_nodes)-1
        for i in V_nodes: 
            if i == depot_ind: continue 
            for j in V_nodes: 
                if j == depot_ind: continue 
                if i == j: continue 
                cluster.problem += u[i] - u[j] + n*cluster.x[i,j,v] <= n - 1

    logger.debug(f"Constraint 28: {len(cluster.problem.constraints)}")


def constraint_29(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    cluster.route_time = pl.LpVariable.dicts("route_time", ((v) for v in cluster.employed_agents), lowBound=0, cat='Continuous')
    for v in list_of_agents.values(): 
        cluster.problem += (
            cluster.route_time[v] == pl.lpSum(
                cluster.tr_times[(i,j)] * cluster.x[i,j,v]
                for i in V_nodes for j in V_nodes if i != j
            ) + 1 * pl.lpSum(cluster.wait[v,t] for t in cluster.timeframe)
        )

    for v in list_of_agents.values(): 
        cluster.problem += cluster.route_time[v] <= cluster.T_MAX

    logger.debug(f"Constraint 29: {len(cluster.problem.constraints)}")


def constraint_30(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    logger.debug(f"Constraint 30: {len(cluster.problem.constraints)}")


def all_constraints(cluster:Any, builder:Any, V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    reversed_original = {v:k for k,v in cluster.original_nodes_dict.items()}
    NODES = V_nodes[:depot_ind] + V_nodes[depot_ind+1:]
    agents = cluster.employed_agents
    model = cluster.problem 
    D = cluster.tr_times
    TF = cluster.timeframe
    MANDATORY_WAIT_TIME = builder.coverage_time
    wait_energy_consumption = builder.average_coverage_energy
    
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

    # for k in agents: 
    #     for i in V_nodes: 
    #         for j in NODES: 
    #             arrival_t = cluster.t.get((i,k), 0)
    #             model += cluster.t[j,k] >= arrival_t + D[(i,j)] - M *(1 - cluster.x[i,j,k])

    
    # for k in agents: 
    #     for i in NODES: 
    #         model += cluster.return_step[k] >= (cluster.t[i,k]) + D[(i,depot_ind)] - M * (1-cluster.x[i,depot_ind,k])
 
    for k in agents:
        for j in NODES:
            import pdb;pdb.set_trace()
            # Arrival at first node >= (Time at Depot + Wait at Depot) + Travel Time
            # Assuming no wait time at the depot itself before starting the tour.
            model += cluster.t[j, k] >= (0 + D[(depot_ind, j)]) - M * (1 - cluster.x[depot_ind, j, k])

    for k in agents:
        for i in NODES:
            for j in NODES:
                if i == j: continue
                # Arrival at j >= (Arrival at i + Wait at i) + Travel Time from i to j
                model += cluster.t[j, k] >= (cluster.t[i, k] + MANDATORY_WAIT_TIME) + D[(i, j)] - M * (1 - cluster.x[i, j, k])

    for k in agents:
        for i in NODES:
            # Return to depot >= (Arrival at last node i + Wait at i) + Travel Time to depot
            model += cluster.return_step[k] >= (cluster.t[i, k] + MANDATORY_WAIT_TIME) + D[(i, depot_ind)] - M * (1 - cluster.x[i, depot_ind, k])


    M_energy = builder.max_battery
    for k in agents: 
        model += cluster.e[depot_ind, k] == builder.max_battery 

    for k in agents: 
        for i in V_nodes: 
            for j in NODES: 
                if i == j : continue 
                
                source = cluster.nodes_dict[i]
                target = cluster.nodes_dict[j]
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
                
                source = cluster.nodes_dict[i]
                target = cluster.nodes_dict[j] - 1
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
            model += visit_counts[i] - visit_counts[j] <= 1 
            model += visit_counts[j] - visit_counts[i] <= 1

    
