import pulp as pl 
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
    for k, v in list_of_agents.items():
        for j in V_nodes: 
            cluster.problem += pl.lpSum(
                cluster.x[i,j,v] for i in V_nodes if i != j
            ) <= cluster.R_points[j], f"Allowed_visits_for_each_agent_{k}_for_node_{j}"

    # for j in V_nodes: 
    #     cluster.problem += pl.lpSum(
    #         cluster.t[i,j,v,t]
    #         for i in V_nodes if i != j 
    #         for v in list_of_agents.values() 
    #         for t in cluster.timeframe
    #     ) == cluster.R_points[j], f"Allowed_visits_for_each_agent_for_node_{j}"

    logger.debug(f"Constraint 0: {len(cluster.problem.constraints)}")


def constraint_1(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Only allow a single travel from depot to all nodes and the reverse as well.
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    for k, v in list_of_agents.items():
        cluster.problem += pl.lpSum(
            cluster.x[depot_ind, j, v] 
            for j in V_nodes if j != depot_ind 
        ) == 1, f"{k}_enters_single_area_from_depot_{depot_ind}"

        cluster.problem += pl.lpSum(
            cluster.x[i, depot_ind, v]
            for i in V_nodes if i != depot_ind 
        ) == 1, f"{k}_leaves_single_area_to_depot_{depot_ind}"
    logger.debug(f"Constraint 1: {len(cluster.problem.constraints)}")        


def constraint_2(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Allow an agent to start his journey whenever it fits best and finish as well at a different time that before (not time frame [-1])
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    for k, v in list_of_agents.items():
        # for j in V_nodes:
        #     if j == depot_ind: continue 
        #     valid_departure_window = cluster.timeframe[:-(cluster.tr_times[(depot_ind, j)] + cluster.tr_times[(j, depot_ind)])]
        #     cluster.problem += pl.lpSum(cluster.t[depot_ind, j, v, t] for t in valid_departure_window) >= 1, \
        #         f"{k}_leaves_depot_{depot_ind}_for_node_{j}_within_valid_time"
            
        # For dynamic departure (start at any time)
        # cluster.problem += pl.lpSum(
        #     cluster.t[depot_ind, j, v, t]
        #     for j in V_nodes if j != depot_ind
        #     for t in cluster.timeframe[:-(cluster.tr_times[(depot_ind, j)] + 1)]
        # # ) <= 1, f"{k}_leaves_depot_once"
        # ) == 1, f"{k}_departs_depot_once"

        # Departure immediately from the base station 
        cluster.problem += pl.lpSum(cluster.t[depot_ind, j, v, cluster.timeframe[0]] for j in V_nodes if depot_ind != j ) == 1, f"{k}_leaves_depot_{depot_ind}_at_specific_interval"
                    
            # valid_return_window = cluster.timeframe[-(cluster.tr_times[(j, depot_ind)] + 1):]
            
            # # cluster.problem += pl.lpSum(cluster.t[j, depot_ind, v, t] for t in valid_return_window) >= 1, \
            # cluster.problem += pl.lpSum(cluster.t[j, depot_ind, v, t] for t in valid_return_window) == 1, \
            #     f"{k}_returns_to_depot_{depot_ind}_from_node_{j}_within_valid_time"

        # For dynamic return (return at feasible time)
        cluster.problem += pl.lpSum(
            cluster.t[i, depot_ind, v, t]
            for i in V_nodes if i != depot_ind
            for t in cluster.timeframe[-(max(cluster.tr_times[(i, depot_ind)] + 1, 1)):]
        # ) <= 1, f"{k}_returns_to_depot_once"
        ) == 1, f"{k}_returns_to_depot_once"
  
    logger.debug(f"Constraint 2: {len(cluster.problem.constraints)}")

    # for k, v in list_of_agents.items():
    #     cluster.problem += cluster.p[v,cluster.timeframe[0]] == depot_ind, f"Positional_variable_at_start_of_journey_for_{k}" 

    
def constraint_3(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Enforce that only a single journey from i -> j exists so that the reverse is not possible (j -> i) 
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    for k, v in list_of_agents.items():
        cluster.problem += pl.lpSum(cluster.x[depot_ind, j, v] for j in V_nodes if j != depot_ind ) + \
                        pl.lpSum(cluster.x[i, depot_ind, v] for i in V_nodes if i != depot_ind ) == 2, f"{k}_start_&_finishes_at_depot_{depot_ind}"  
    logger.debug(f"Constraint 3: {len(cluster.problem.constraints)}")
            

def constraint_4(cluster:Any, builder:Any, V_nodes:list, list_of_agents:dict):
    """
    Prohibit Depot looping for each agent.
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    for k, v in list_of_agents.items():
        cluster.problem += cluster.x[depot_ind, depot_ind, v] == 0,  f"No_loop_at depot_{depot_ind}_for_{k}_at_any_timepoint"
    logger.debug(f"Constraint 4: {len(cluster.problem.constraints)}")


def constraint_5(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Only one visit from i to j and from j to i except for bridge nodes. We don't exclude the depots here as they also are a 1 out 1 in node. 
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    # PER AGENT 
    for k, v in list_of_agents.items(): 
        for i in V_nodes: 
            if cluster.nodes_dict[i] not in cluster.bridge_nodes and i != depot_ind:
                cluster.problem += pl.lpSum(cluster.x[i,j,v] for j in V_nodes if i != j) == 1, f"Only_one_visit_from_i_to_j_for_agent_{k}_for_node_{i}"
            elif cluster.nodes_dict[i] in cluster.bridge_nodes and i != depot_ind:
                cluster.problem += pl.lpSum(cluster.x[i,j,v] for j in V_nodes if i != j) <= cluster.R_points[i], f"Only_one_visit_from_i_to_j_for_agent_{k}_for_node_{i}"
        # Suggestion, use the <= inequality for bridge nodes so as to not overconstrain the problem. 
        for j in V_nodes:
            if cluster.nodes_dict[j] not in cluster.bridge_nodes and j != depot_ind:
                cluster.problem += pl.lpSum(cluster.x[i,j,v] for i in V_nodes if i != j) == 1, f"Only_one_visit_from_j_to_i_for_agent_{k}_for_node_{j}"
            elif cluster.nodes_dict[j] in cluster.bridge_nodes and j != depot_ind:
                cluster.problem += pl.lpSum(cluster.x[i,j,v] for i in V_nodes if i != j) <= cluster.R_points[j], f"Only_one_visit_from_j_to_i_for_agent_{k}_for_node_{j}"

    # PER TEAM 
    # for i in V_nodes:  
    #     if cluster.nodes_dict[i] not in cluster.bridge_nodes and i != depot_ind:
    #     #     cluster.problem += pl.lpSum(cluster.x[i,j,v] for j in V_nodes if i != j for v in list_of_agents.values()) == 1, f"Only_one_visit_from_i_to_j_for_agents_for_node_{i}"
    #     # elif cluster.nodes_dict[i] in cluster.bridge_nodes and i != depot_ind:
    #         cluster.problem += pl.lpSum(cluster.x[i,j,v] for j in V_nodes if i != j for v in list_of_agents.values()) <= cluster.R_points[i], f"Only_one_visit_from_i_to_j_for_agents_for_node_{i}"

    # for j in V_nodes: 
    #     if cluster.nodes_dict[j] not in cluster.bridge_nodes and j != depot_ind:
    #     #     cluster.problem += pl.lpSum(cluster.x[i,j,v] for i in V_nodes if i != j for v in list_of_agents.values()) == 1, f"Only_one_visit_from_j_to_i_for_agents_for_node_{j}"
    #     # elif cluster.nodes_dict[j] in cluster.bridge_nodes and j != depot_ind:
    #         cluster.problem += pl.lpSum(cluster.x[i,j,v] for i in V_nodes if i != j for v in list_of_agents.values()) <= cluster.R_points[j], f"Only_one_visit_from_j_to_i_for_agents_for_node_{j}"
    logger.debug(f"Constraint 5: {len(cluster.problem.constraints)}")


def constraint_6(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Collision avoidance / Unique agent per node 
    """    
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    out_arcs, _ = get_arcs(V_nodes, depot_ind)
    for i in V_nodes: #out_arcs 
        for j in V_nodes: #in_arcs 
            if i == j: continue 
            # if nodes_dict[j] not in self.bridge_nodes and nodes_dict[i] not in self.bridge_nodes:
            for step in cluster.timeframe:
                cluster.problem += pl.lpSum(cluster.t[i,j,v,step] for _,v in list_of_agents.items()) <= 1, f"Unique_Time_visits_constraint_at_travel_{i}_{j}_at_time_{step}" 
    logger.debug(f"Constraint 6: {len(cluster.problem.constraints)}")


def constraint_7(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Assign the appropriate values to each time step to cover for the duration 
    """
    # depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    # for k, v in list_of_agents.items():
    #         for j in V_nodes:
    #             if j == depot_ind: continue 
    #             cluster.problem += pl.lpSum(
    #                 cluster.t[i,j,v,t]
    #                 for i in V_nodes if i != j
    #                 for t in cluster.timeframe) <= cluster.R_points[j], f"Agent_{k}_max_arrivals_at_{j}"
    logger.debug(f"Constraint 7: {len(cluster.problem.constraints)}")


def constraint_8(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Synchronization between the time and space 
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    #Performance Changes 
    for k, v in list_of_agents.items(): 
        for i in V_nodes: 
            for j in V_nodes:
                if i == j : continue 
                if i == depot_ind or j == depot_ind: continue 
                duration = cluster.tr_times[(i,j)]
                for step in cluster.timeframe: 
                    if step + duration <= cluster.timeframe[-1]:    
                        cluster.problem += pl.lpSum(
                            cluster.t[i,j,v,t] for t in range(step, step + duration)
                        ) <= duration*cluster.x[i,j,v]

                        cluster.problem += pl.lpSum(
                            cluster.t[i,j,v,t] for t in range(step, step + duration)
                        ) >= duration*cluster.x[i,j,v]
    
    # Suggestion don't use the T_MAX as it prevents later steps from being checked while it should check feasibility over the full timeline, not against the max travel time to the depot. 
    # Missing depot transitions which may be intentional - but it leaves depot timing uncontrolled unless differenct synchronization constraint is used. 
    # for k, v in list_of_agents.items(): 
    #     for i in V_nodes: 
    #         for j in V_nodes:
    #             if i == j : continue 
    #             if i == depot_ind or j == depot_ind: continue 

    #             for step in cluster.timeframe: 
    #                 if step + cluster.tr_times[(i,j)] <= cluster.timeframe[-1]:    
    #                     cluster.problem += pl.lpSum(
    #                         cluster.t[i,j,v,t] for t in range(step, step + cluster.tr_times[(i,j)])
    #                     ) == cluster.tr_times[(i,j)]*cluster.x[i,j,v]
    
    #PER TEAM 
    # for k, v in list_of_agents.items(): 
    #     for i in V_nodes: 
    #         for j in V_nodes:
    #             if i == j : continue 
    #             if i == depot_ind or j == depot_ind: continue 
    #             duration = cluster.tr_times[(i,j)]

    #             for step in cluster.timeframe: 
    #                 if step + cluster.tr_times[(i,j)] > cluster.timeframe[-1]:continue     

    #                 lhs = pl.lpSum(
    #                     cluster.t[i,j,v,t] for t in range(step, step + duration)
    #                 )
    #                 cluster.problem += lhs <= duration * cluster.x[i,j,v], f"Duration_upper_{i}_{j}_{v}_{step}"
    #                 cluster.problem += lhs >= cluster.x[i,j,v], f"Duration_lower_{i}_{j}_{v}_{step}"

    logger.debug(f"Constraint 8: {len(cluster.problem.constraints)}")



def constraint_9(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Only one travel from i to j for the entirety of the time frame. This cannotbe used if we opt to align all time steps individually. 
    """
    
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    for k, v in list_of_agents.items(): 
                
        cluster.problem += pl.lpSum(cluster.t[depot_ind, j, v, t]
            for j in V_nodes if j != depot_ind 
            for t in cluster.timeframe[:-(cluster.tr_times[(j, depot_ind)] + 1)]
        ) >= 1, f"Only_one_travel_from_depot_to_j_for_agent_{k}_based_on_time"
        
        cluster.problem += pl.lpSum(cluster.t[j,depot_ind,v,t]
            for j in V_nodes if j != depot_ind 
            for t in cluster.timeframe[-(cluster.tr_times[(j, depot_ind)] + 1):]
        ) >= 1, f"Only_one_travel_from_j_to_depot_for_agent_{k}_based_on_time"

    logger.debug(f"Constraint 9: {len(cluster.problem.constraints)}")


def constraint_10(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    for k, v in list_of_agents.items():
        for i in V_nodes:
            for j in V_nodes:
                if i == j : continue 
                if i == cluster.depot_id or j == cluster.depot_id: continue
                if cluster.nodes_dict[i] in cluster.bridge_nodes or cluster.nodes_dict[j] in cluster.bridge_nodes: continue 
                for t in cluster.timeframe:
                    if t + cluster.tr_times[(i,j)] in cluster.timeframe:
                        # cluster.problem += cluster.t[i, j, v, t] + cluster.t[j, i, v, t + 1] <= 1, f"No_immediate_loop_{i}_{j}_time_{t}_agent_{k}"
                        cluster.problem += cluster.t[i, j, v, t] + cluster.t[j, i, v, t + cluster.tr_times[(i,j)]] <= 1, f"No_immediate_loop_{i}_{j}_time_{t}_agent_{k}"
    logger.debug(f"Constraint 10: {len(cluster.problem.constraints)}")


def constraint_11(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    out_arcs, _ = get_arcs(V_nodes, cluster.depot_id)

    for k1, v1 in list_of_agents.items() : 
        for k2, v2 in list_of_agents.items() : 
            if k1 != k2 : 
                for step in cluster.timeframe: 
                    cluster.problem += pl.lpSum(
                        cluster.t[i, j, v1, step] - cluster.t[i,j, v2, step]
                        for i in out_arcs
                        for j in out_arcs[i]
                        # if cluster.nodes_dict[i] != cluster.bridge_nodes and 
                        #  cluster.nodes_dict[j] != cluster.bridge_nodes
                        ) != 0, f"Agent_unique_paths_for_{k1}_and_{k2}_at_time_{step}"
    logger.debug(f"Constraint 11: {len(cluster.problem.constraints)}")

    # The != 0 is not linear hence it might not be correctly interpreted by the solver. It is reformulated using binary auxiliary, big-M or indicator constraints. 
    # this also overconstrains the system. 


def constraint_12(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    out_arcs, in_arcs = get_arcs(V_nodes, depot_ind)
    for k, v in list_of_agents.items():
        for i in out_arcs:
            source = cluster.nodes_dict[i]-1
            for j in out_arcs[i]:
                target = cluster.nodes_dict[j]-1
                cluster.problem += cluster.e[j,v] >= cluster.e[i,v] - builder.normalized_battery[source][target] * cluster.x[i,j,v], f"Update_remaining_energy_{i}_{j}_for_{k}"
                cluster.problem += cluster.e[i,v] >= builder.normalized_battery[source][target] * cluster.x[i, j, v],f"No_travel_if_low_energy_{i}_{j}_for_{k}"
            cluster.problem += cluster.e[i,v] >= builder.normalized_battery[source][cluster.nodes_dict[depot_ind]-1] * cluster.x[i, depot_ind, v],f"Enough_energy_to_return_to_depot_from_{i}_for_{k}"
    logger.debug(f"Constraint 12: {len(cluster.problem.constraints)}")
    
        # for i in out_arcs: 
        #     source = cluster.nodes_dict[i]-1
        #     for j in out_arcs[i]: 
        #         target =  cluster.nodes_dict[j]-1
        #         # for step in self.timeFrame_per_cluster[self.tr_times[(depot_ind,i)]:-(self.tr_times[(i,j)]+self.tr_times[(j,depot_ind)])]: 
        #         for step in cluster.timeframe: 
        #             cluster.problem += cluster.e[j,v] >= cluster.e[i,v] - builder.normalized_battery[source][target] * pl.lpSum(cluster.t[i, j, v, step]), f"Energy_update_{i}_{j}_at_time_{step}_for_{k}"

            # cluster.problem += cluster.e[i,v] >= builder.normalized_battery[source][target] * \
            #     pl.lpSum(
            #         cluster.t[i,depot_ind,v,t] for t in cluster.timeframe[-cluster.tr_times[(i,depot_ind)]:]
            #     ), f"Ensure_depot_return_from{i}_for_{k}_for_correct_time_Steps"
            

def constraint_13(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    for k, v in list_of_agents.items():
        for i in V_nodes:
            if i == depot_ind: continue
            cluster.problem += cluster.e[i,v] >= 0, f"Energy_cannot_be_negative_{i}_for_{k}"

        cluster.problem += cluster.e[depot_ind, v] == builder.max_battery_norm, f"Every_agent_starts_with_full_battery_{k}"
    logger.debug(f"Constraint 13: {len(cluster.problem.constraints)}")


def constraint_14(cluster:Any, builder:Any, V_nodes:list, list_of_agents:dict): 
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    M = len(V_nodes)+ len(cluster.bridge_nodes) -1 
    for k, v in list_of_agents.items():
        for t in cluster.timeframe[1:]:
            # The agent is at the depot until they begin their first travel
            # cluster.problem += cluster.p[v, t] == depot_ind + (1 - pl.lpSum(cluster.t[depot_ind, j, v, t] for j in V_nodes if j != depot_ind)) * M  # Big-M allows flexibility before departure
            cluster.problem += cluster.p[v ,t] == depot_ind + (1 - pl.lpSum(cluster.t[i, depot_ind, v, t] for i in V_nodes if i != depot_ind)) * M
        cluster.problem += cluster.p[v,cluster.timeframe[0]] == depot_ind, f"Positional_variable_at_start_of_journey_for_{k}"     
    logger.debug(f"Constraint 14: {len(cluster.problem.constraints)}")


def constraint_15(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    for k, v in list_of_agents.items():
        for i in V_nodes:
            for j in V_nodes:
                if i == j : continue 
                travel_duration = cluster.tr_times[(i, j)]
                for t_start in cluster.timeframe[:-travel_duration]:
                    for dt in range(travel_duration):
                        t = t_start + dt
                        cluster.problem += cluster.busy[v, t] >= cluster.t[i, j, v, t_start], f"Busy_if_travel_{i}_{j}_starts_at_{t_start}_for_{k}_covers_{t}"
    logger.debug(f"Constraint 15: {len(cluster.problem.constraints)}")


def constraint_16(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    for k,v in list_of_agents.items():
        for t in cluster.timeframe:
            cluster.problem += cluster.busy[v, t] + cluster.wait[v, t] == 1

            #Suggestion: With the current formulation, the constraint limits co-activation, but does not guarantee at least one is active. 
            # So the agent might be inactive (neither busy nor waiting) unless we add the == instead of <=. 

            # the wait variable can also be defined as a wait[v,t] >= 1-busy[v,t] or a separate activity tracking layer 
    logger.debug(f"Constraint 16: {len(cluster.problem.constraints)}")


def constraint_17(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    #Per agent 
    for k, v in list_of_agents.items():
        # for i in V_nodes:
        #     if i == depot_ind: continue
            # cluster.problem += pl.lpSum(cluster.t[depot_ind,i,v,t] for t in cluster.timeframe[0:cluster.tr_times[(depot_ind,i)]]) >= cluster.x[depot_ind,i,v]
            # valid_departure_window = cluster.timeframe[:-(cluster.tr_times[(depot_ind, i)] + cluster.tr_times[(i, depot_ind)])]
            # cluster.problem += pl.lpSum(cluster.t[depot_ind, i, v, t] for t in valid_departure_window
            # ) >= cluster.x[depot_ind,i,v] , f"Dynamic_time_enforcement_{depot_ind}_{i}_for_{k}_time"
        for j in V_nodes: 
            if j != depot_ind: 
                for step in cluster.timeframe[1:cluster.tr_times[(depot_ind, j)]]: 
                    cluster.problem += (
                        cluster.t[depot_ind, j, v, step] == cluster.t[depot_ind, j, v, cluster.timeframe[0]], 
                        f"Dynamic_time_enforcement_{depot_ind}_{j}_for_{k}_time_{step}"
                    )   
                    cluster.problem += (
                        cluster.t[depot_ind, j, v, step] == cluster.x[depot_ind, j, v], 
                        f"Enforce synchronization_between_t_and_x_{depot_ind}_{j}_for_{k}_time_{step}"
                    )   

        for j in V_nodes:
            if j == depot_ind: continue
            
            valid_return_window = cluster.timeframe[-(cluster.tr_times[(j, depot_ind)] + 1):]
            cluster.problem += pl.lpSum(cluster.t[j, depot_ind, v, t] for t in valid_return_window
            ) >= cluster.x[j, depot_ind, v], f"Dynamic_time_enforcement_{j}_{depot_ind}_for_{k}_time"

    # PER TEAM 
    # for k, v in list_of_agents.items(): 
    #     for i in V_nodes: 
    #         if i == depot_ind: continue 
    #         valid_departure_window = cluster.timeframe[:-(cluster.tr_times[(depot_ind, i)] + cluster.tr_times[(i, depot_ind)])]
    #         cluster.problem += pl.lpSum(
    #             cluster.t[depot_ind, i, v, t] for t in valid_departure_window
    #         ) <= len(valid_departure_window) * cluster.x[depot_ind, i, v], f"Valid_departure_time_enforced_{v}_{i}"

    #     for j in V_nodes:
    #         if j == depot_ind: continue

    #         valid_return_window = cluster.timeframe[-(cluster.tr_times[(j, depot_ind)] + 1):]
    #         cluster.problem += pl.lpSum(
    #             cluster.t[j, depot_ind, v, t] for t in valid_return_window
    #         ) <= len(valid_return_window) * cluster.x[j, depot_ind, v], f"Valid_return_time_enforced_{v}_{j}"

    logger.debug(f"Constraint 17: {len(cluster.problem.constraints)}")



def constraint_18(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    constraint_counter = 0
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    out_arcs, in_arcs = get_arcs(V_nodes, depot_ind)
    M = len(V_nodes) * len(cluster.timeframe)
    for k, v in list_of_agents.items(): 
        for i in out_arcs: 
            for j in out_arcs[i]:
                if cluster.nodes_dict[i] in cluster.bridge_nodes or cluster.nodes_dict[j] in cluster.bridge_nodes: continue
                trip_time = cluster.tr_times[(i,j)]
                for step in cluster.timeframe[:-trip_time-1]:
                    if (i, j, v, step) in cluster.t and cluster.t[i,j,v,step].name in cluster.problem.variablesDict():
                        cluster.problem += cluster.p[v, step + trip_time-1] <= j + (1 - cluster.t[i, j, v, step]) * M, f"Positional_alignment_with_step_{step}_for_{k}_at_{j}{i}"
                        cluster.problem += cluster.p[v, step + trip_time-1] >= j - (1 - cluster.t[i, j, v, step]) * M, f"Positional_alignment_with_step_{step}_for_{k}_at_-{j}{i}"
                        constraint_counter += 1

    logger.debug(f"Constraint 18: {len(cluster.problem.constraints)}")


def constraint_19(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    for k,v in list_of_agents.items(): 
        for i in V_nodes: 
            for j in V_nodes: 
                if i == j: continue 
                if i == cluster.depot_id or j == cluster.depot_id: continue
                for t in cluster.timeframe: 
                    if t + cluster.tr_times[(i,j)] in cluster.timeframe:
                        cluster.problem += cluster.t[i, j, v, t] <= cluster.busy[v, t + cluster.tr_times[(i,j)]], f"Busy_constraint_{i}_{j}_for_{k}_at_time_{t}"
    logger.debug(f"Constraint 19: {len(cluster.problem.constraints)}")


def constraint_20(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    
    for v in list_of_agents.values():
        for t in cluster.timeframe:
            cluster.problem += pl.lpSum(
                cluster.t[i, j, v, tau]
                for i in V_nodes
                for j in V_nodes
                if i != j 
                for tau in range(t, min(t + cluster.tr_times[(i,j)], cluster.timeframe[-1] + 1))
                if tau in cluster.timeframe
            ) == 1, f"exclusive_window_agent_{v}_time_{t}"
    logger.debug(f"Constraint 19: {len(cluster.problem.constraints)}")
    

def constraint_21(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
   # This does not work for the team but shoulf work per agent. 
   # visit at most once
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    out_arcs, in_arcs = get_arcs(V_nodes, cluster.depot_id)
    for k, v in list_of_agents.items():
        for j in V_nodes: 
            cluster.problem += pl.lpSum(
                cluster.t[i, j, v, t]
                for i in V_nodes if i != j 
                for t in cluster.timeframe
            ) >= 1, f"Constraint_21_{k}_for_node_{j}"

    logger.debug(f"Constraint 21: {len(cluster.problem.constraints)}")


def constraint_22(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    # suggestion : change t to use x so to get rid of time. 
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    for k, v in list_of_agents.items():
        for j in V_nodes: 
            if j == cluster.depot_id: continue 
            cluster.problem += pl.lpSum(
                cluster.x[i, j, v] for i in V_nodes if i != j
                
                # for t in cluster.timeframe
            ) == pl.lpSum(
                cluster.x[j, l, v] for l in V_nodes if l != j
                # for t in cluster.timeframe
            ) 
    logger.debug(f"Constraint 22 : {len(cluster.problem.constraints)}")


def constraint_23(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    #  
    logger.debug(f"Constraint 23: {len(cluster.problem.constraints)}")


def constraint_24(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    # Constraint : travel exclusivity per agent per timestep. 
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    for k, v in list_of_agents.items():
        for t in cluster.timeframe: 
            cluster.problem += pl.lpSum(
                cluster.t[i, j, v, t]
                for i in V_nodes 
                for j in V_nodes
                if i != j
            ) == 1, f"Only_one_travel_from_i_to_j_for_agent_{k}_at_time_{t}"
    logger.debug(f"Constraint 24: {len(cluster.problem.constraints)}")


def constraint_25(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    for k, v in list_of_agents.items(): 
        for i in V_nodes: 
            for j in V_nodes: 
                if i == j: continue 
                if cluster.nodes_dict[i] in cluster.bridge_nodes or cluster.nodes_dict[j] in cluster.bridge_nodes: continue
                cluster.problem += pl.lpSum(cluster.t[i,j,v,t] for t in cluster.timeframe) <=1 
    logger.debug(f"Constraint 25: {len(cluster.problem.constraints)}")


def constraint_26(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    # This is a good approach however a better approach would be to use an integer decision variable 
    # that penalizes every repeat visit not just presence. 
    # M = len(V_nodes) * len(cluster.timeframe)

    # for k, v in list_of_agents.items(): 
    #     for j in V_nodes: 
    #         cluster.problem += pl.lpSum(
    #             cluster.t[i,j,v,t]
    #             for i in V_nodes if i != j 
    #             for t in cluster.timeframe
    #         ) <= M * cluster.y[j,v] 

    for k, v in list_of_agents.items(): 
        for j in V_nodes: 
            cluster.problem += pl.lpSum(
                cluster.t[i,j,v,t]
                for i in V_nodes if i != j 
                for t in cluster.timeframe
            ) <= cluster.y[j,v]
    logger.debug(f"Constraint 26: {len(cluster.problem.constraints)}")


def constraint_27(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    for k, v in list_of_agents.items(): 
        cluster.problem += pl.lpSum(
            cluster.t[i,j,v] 
            for i in V_nodes
            for j in cluster.bridge_nodes
            if i != j 
            for t in cluster.timeframe
        ) >=1 

    for k, v in list_of_agents.items(): 
        cluster.problem += pl.lpSum(
            cluster.t[i,j,v] 
            for i in cluster.bridge_nodes
            for j in V_nodes
            if i != j 
            for t in cluster.timeframe
        ) >=1
    logger.debug(f"Constraint 27: {len(cluster.problem.constraints)}")


def constraint_28(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    for k, v in list_of_agents.items(): 
        cluster.problem += pl.lpSum(
            cluster.t[i,j,v,t]
            for i in V_nodes
            for j in V_nodes
            if i != j
            for t in cluster.timeframe

        ) == len(V_nodes) + len(cluster.bridge_nodes) + 1
    logger.debug(f"Constraint 28: {len(cluster.problem.constraints)}")


def constraint_29(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    for k, v in list_of_agents.items(): 
        cluster.problem += pl.lpSum(
            cluster.t[depot_ind,j,v,t]
            for j in V_nodes
            if depot_ind != j
            for t in cluster.timeframe
        ) <= len(V_nodes) * cluster.agent_active[v], f"Agent_{k}_active_at_depot_{depot_ind}"
    logger.debug(f"Constraint 29: {len(cluster.problem.constraints)}")



def constraint_30(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    import numpy as np
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    MIN_VISITS = np.round(len(V_nodes) / len(list_of_agents))
    for k, v in list_of_agents.items(): 
        cluster.problem += pl.lpSum(

            cluster.x[i,j,v]
            for i in V_nodes
            for j in V_nodes
            if i != j and i != cluster.depot_id and j != cluster.depot_id
        ) >= MIN_VISITS * cluster.agent_active[v], f"Agent_{k}_is_active_at_depot_{depot_ind}"
    logger.debug(f"Constraint 30: {len(cluster.problem.constraints)}")





def constraint_31(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    for v in list_of_agents.values():
        cluster.problem += pl.lpSum(
            cluster.t[depot_ind, j, v, 0]
            for j in V_nodes if j != depot_ind
        ) <= cluster.agent_active[v]

    logger.debug(f"Constraint 31: {len(cluster.problem.constraints)}")


def constraint_32(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    out_arcs, in_arcs = get_arcs(V_nodes, cluster.depot_id)
 
    for i in out_arcs: 
        for j in out_arcs[i]: 
            if cluster.nodes_dict[j] not in cluster.bridge_nodes:
                for k, v in list_of_agents.items(): 
                    cluster.problem += cluster.x[i,j,v] + cluster.x[j,i,v] <= 1, f"No_loops_in_path_for_{k}_at_{i}_{j}"
               
