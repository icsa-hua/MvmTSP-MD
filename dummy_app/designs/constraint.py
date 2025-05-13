import pulp as pl 
from collections import defaultdict


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



def constraint_0(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
    """
    Allow multiple visits (essential for MV-TSP)
    """
    for k, v in list_of_agents.items():
        for j in V_nodes: 
            cluster.problem += pl.lpSum(
                cluster.x[i,j,v] for i in V_nodes if i != j
            ) <= cluster.R_points[j], f"Allowed_visits_for_each_agent_{k}_for_node_{j}"


def constraint_1(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
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
            

def constraint_2(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
    """
    Allow an agent to start his journey whenever it fits best and finish as well at a different time that before (not time frame [-1])
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    for k, v in list_of_agents.items():
        for j in V_nodes:
            if j == depot_ind: continue 
            valid_departure_window = cluster.timeframe[:-(cluster.tr_times[(depot_ind, j)] + cluster.tr_times[(j, depot_ind)])]
            cluster.problem += pl.lpSum(cluster.t[depot_ind, j, v, t] for t in valid_departure_window) >= 1, \
                f"{k}_leaves_depot_{depot_ind}_for_node_{j}_within_valid_time"

            valid_return_window = cluster.timeFrame_per_cluster[-(cluster.tr_times[(j, depot_ind)] + 1):]
            cluster.problem += pl.lpSum(cluster.t[j, depot_ind, v, t] for t in valid_return_window) >= 1, \
                f"{k}_returns_to_depot_{depot_ind}_from_node_{j}_within_valid_time"
                
    
def constraint_3(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
    """
    Enforce that only a single journey from i -> j exists so that the reverse is not possible (j -> i) 
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    for k, v in list_of_agents.items():
        cluster.problem += pl.lpSum(cluster.x[depot_ind, j, v] for j in V_nodes if j != depot_ind ) + \
                        pl.lpSum(cluster.x[i, depot_ind, v] for i in V_nodes if i != depot_ind ) == 2, f"{k}_start_&_finishes_at_depot_{depot_ind}"  
                

def constraint_4(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
    """
    Prohibit Depot looping for each agent.
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    for k, v in list_of_agents.items():
        cluster.problem += cluster.x[depot_ind, depot_ind, v] == 0,  f"No_loop_at depot_{depot_ind}_for_{k}_at_any_timepoint"


def constraint_5(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
    """
    Only one visit from i to j and from j to i except for bridge nodes. We don't exclude the depots here as they also are a 1 out 1 in node. 
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    for k, v in list_of_agents.items(): 
        for i in V_nodes: 
            if cluster.nodes_dict[i] not in cluster.bridge_nodes and i != depot_ind:
                cluster.problem += pl.lpSum(cluster.x[i,j,v] for j in V_nodes if i != j) == 1, f"Only_one_visit_from_i_to_j_for_agent_{k}_for_node_{i}"
            elif cluster.nodes_dict[i] in cluster.bridge_nodes and i != depot_ind:
                cluster.problem += pl.lpSum(cluster.x[i,j,v] for j in V_nodes if i != j) == cluster.R_points[i], f"Only_one_visit_from_i_to_j_for_agent_{k}_for_node_{i}"

        for j in V_nodes:
            if cluster.nodes_dict[j] not in cluster.bridge_nodes and j != depot_ind:
                cluster.problem += pl.lpSum(cluster.x[i,j,v] for i in V_nodes if i != j) == 1, f"Only_one_visit_from_j_to_i_for_agent_{k}_for_node_{j}"
            elif cluster.nodes_dict[j] in cluster.bridge_nodes and j != depot_ind:
                cluster.problem += pl.lpSum(cluster.x[i,j,v] for i in V_nodes if i != j) == cluster.R_points[j], f"Only_one_visit_from_j_to_i_for_agent_{k}_for_node_{j}"


def constraint_6(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
    """
    Collision avoidance / Unique agent per node 
    """    
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    out_arcs, _ = get_arcs(V_nodes, depot_ind)
    for i in out_arcs: 
        for j in out_arcs[i]:
            # if nodes_dict[j] not in self.bridge_nodes and nodes_dict[i] not in self.bridge_nodes:
                for step in cluster.timeframe:
                    cluster.problem += pl.lpSum(cluster.t[i,j,v,step] for _,v in list_of_agents.items()) <= 1, f"Unique_Time_visits_constraint_at_travel_{i}_{j}_at_time_{step}" 



def constraint_7(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
    """
    Assign the appropriate values to each time step to cover for the duration 
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    for k, v in list_of_agents.items():
        for i in V_nodes:
            for j in V_nodes:
                if j == depot_ind or i == depot_ind: continue 
                cluster.problem += pl.lpSum(cluster.t[i,j,v,t] for t in cluster.timeframe) == cluster.tr_times[(i,j)], f"Travel_time_constraint_for_agent_{k}_from_{i}_to_{j}"


def constraint_8(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
    """
    Synchronization between the time and space 
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    
    T_max = max(cluster.tr_times[(i,depot_ind)] for i in V_nodes if i != depot_ind)

    for k, v in list_of_agents.items(): 
        for i in V_nodes: 
            for j in V_nodes:
                if i == j : continue 
                if i == depot_ind or j == depot_ind: continue 

                for step in cluster.timeframe: 
                    if step + cluster.tr_times[(i,j)] - 1 < T_max:    
                        cluster.problem += pl.lpSum(
                            cluster.t[i,j,v,t] for t in range(step, step + cluster.tr_times[(i,j)])
                        ) == cluster.tr_times[(i,j)]*cluster.x[i,j,v]


def constraint_9(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
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


def constraint_10(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
    for k, v in list_of_agents.items():
        for i in V_nodes:
            for j in V_nodes:
                if i == j : continue 
                if i == cluster.depot_id or j == cluster.depot_id: continue
                if cluster.nodes_dict[i] in cluster.bridge_nodes or cluster.nodes_dict[j] in cluster.bridge_nodes: continue 
                for t in cluster.timeframe:
                    if t + 1 in cluster.timeframe:
                        cluster.problem += cluster.t[i, j, v, t] + cluster.t[j, i, v, t + 1] <= 1, f"No_immediate_loop_{i}_{j}_time_{t}_agent_{k}"


def constraint_11(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
    out_arcs, _ = get_arcs(V_nodes, cluster.depot_id)

    for k1, v1 in list_of_agents.items() : 
        for k2, v2 in list_of_agents.items() : 
            if k1 != k2 : 
                for step in cluster.timeframe: 
                    cluster.problem += pl.lpSum(cluster.t[i, j, v1, step] - cluster.t[i,j, v2, step] for i in out_arcs for j in out_arcs[i]) != 0, f"Agent_unique_paths_for_{k1}_and_{k2}_at_time_{step}"


def constraint_12(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    out_arcs, in_arcs = get_arcs(V_nodes, depot_ind)
    for k, v in list_of_agents.items():
        for i in out_arcs:
            for j in out_arcs[i]:
                source, target = cluster.nodes_dict[i]-1, cluster.nodes_dict[j]-1
                cluster.problem += cluster.e[j,v] >= cluster.e[i,v] - builder.normalized_battery[source][target] * cluster.x[i,j,v], f"Update_remaining_energy_{i}_{j}_for_{k}"
                cluster.problem += cluster.e[i,v] >= builder.normalized_battery[source][target] * cluster.x[i, j, v],f"No_travel_if_low_energy_{i}_{j}_for_{k}"
            cluster.problem += cluster.e[i,v] >= builder.normalized_battery[source][cluster.nodes_dict[depot_ind]-1] * cluster.x[i, depot_ind, v],f"Enough_energy_to_return_to_depot_from_{i}_for_{k}"
        
        for i in out_arcs: 
            for j in out_arcs[i]: 
                source, target = cluster.nodes_dict[i]-1, cluster.nodes_dict[j]-1
                # for step in self.timeFrame_per_cluster[self.tr_times[(depot_ind,i)]:-(self.tr_times[(i,j)]+self.tr_times[(j,depot_ind)])]: 
                for step in cluster.timeframe: 
                    cluster.problem += cluster.e[j,v] >= cluster.e[i,v] - builder.normalized_battery[source][target] * pl.lpSum(cluster.t[i, j, v, step]), f"Energy_update_{i}_{j}_at_time_{step}_for_{k}"

            cluster.problem += cluster.e[i,v] >= builder.normalized_battery[source][target] * \
                pl.lpSum(
                    cluster.t[i,depot_ind,v,t] for t in cluster.timeframe[-cluster.tr_times[(i,depot_ind)]:]
                ), f"Ensure_depot_return_from{i}_for_{k}_for_correct_time_Steps"
            

def constraint_13(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    for k, v in list_of_agents.items():
        for i in V_nodes:
            if i == depot_ind: continue

            for t in cluster.timeframe: 
                cluster.problem += cluster.e[i,v] >= 0, f"Energy_cannot_be_negative_{i}_for_{k}_at_time_{t}"

        cluster.problem += cluster.e[depot_ind, v] == builder.max_battery_norm, f"Every_agent_starts_with_full_battery_{k}"
  

def constraint_14(cluster:object, builder:object , V_nodes:list, list_of_agents:dict): 
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    M = len(V_nodes)+ len(cluster.bridge_nodes) -1 
    for k, v in list_of_agents.items():
        for t in cluster.timeframe:
            # The agent is at the depot until they begin their first travel
            cluster.problem += cluster.p[v, t] == depot_ind + (1 - pl.lpSum(cluster.t[depot_ind, j, v, t] for j in V_nodes if j != depot_ind)) * M  # Big-M allows flexibility before departure
            cluster.problem += cluster.p[v ,t] == depot_ind + (1 - pl.lpSum(cluster.t[i, depot_ind, v, t] for i in V_nodes if i != depot_ind)) * M


def constraint_15(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
    for k, v in list_of_agents.items():
        for i in V_nodes:
            for j in V_nodes:
                if i == j : continue 
                travel_duration = cluster.tr_times[(i, j)]
                for t_start in cluster.timeframe[:-travel_duration]:
                    for dt in range(travel_duration):
                        t = t_start + dt
                        cluster.problem += cluster.busy[v, t] >= cluster.t[i, j, v, t_start], f"Busy_if_travel_{i}_{j}_starts_at_{t_start}_for_{k}_covers_{t}"


def constraint_16(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
    for k,v in list_of_agents.items():
        for t in cluster.timeframe:
            cluster.problem += cluster.busy[v, t] + cluster.wait[v, t] <= 1


def constraint_17(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    
    for k, v in list_of_agents.items():
        for i in V_nodes:
            if i == depot_ind: continue

            valid_departure_window = cluster.timeframe[:-(cluster.tr_times[(depot_ind, i)] + cluster.tr_times[(i, depot_ind)])]
            cluster.problem += pl.lpSum(cluster.t[depot_ind, i, v, t] for t in valid_departure_window
            ) >= cluster.x[depot_ind,i,v] , f"Dynamic_time_enforcement_{depot_ind}_{i}_for_{k}_time"
                    
        for j in V_nodes:
            if j == depot_ind: continue
            
            valid_return_window = cluster.timeframe[-(cluster.tr_times[(j, depot_ind)] + 1):]
            cluster.problem += pl.lpSum(cluster.t[j, depot_ind, v, t] for t in valid_return_window
            ) >= cluster.x[j, depot_ind, v], f"Dynamic_time_enforcement_{j}_{depot_ind}_for_{k}_time"


def constraint_18(cluster:object, builder:object , V_nodes:list, list_of_agents:dict):
    constraint_counter = 0
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    M = len(V_nodes) 
    for k, v in list_of_agents.items(): 
        for i in V_nodes: 
            for j in V_nodes:
                if i == j and (i==depot_ind or j==depot_ind) : continue 
                if cluster.nodes_dict[i] in cluster.bridge_nodes or cluster.nodes_dict[j] in cluster.bridge_nodes: continue
                trip_time = cluster.tr_times[(i,j)]
                for step in cluster.timeframe[:-trip_time]:
                    if (i, j, v, step) in cluster.t and cluster.t[i,j,v,step].name in cluster.problem.variablesDict():
                        cluster.problem += cluster.p[v, step + trip_time] <= j + (1 - cluster.t[i, j, v, step]) * M, f"Positional_alignment_with_step_{step}_for_{k}_at_{j}{i}"
                        cluster.problem += cluster.p[v, step + trip_time] >= j - (1 - cluster.t[i, j, v, step]) * M, f"Positional_alignment_with_step_{step}_for_{k}_at_-{j}{i}"
                        constraint_counter += 1


