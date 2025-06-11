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


def all_constraints(cluster:Any, builder:Any, V_nodes:list, list_of_agents:dict): 
    
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    cluster.route_time = pl.LpVariable.dicts("route_time", ((v) for v in cluster.employed_agents), lowBound=0, cat='Continuous')

    M = builder.max_battery
    
    for k, v in list_of_agents.items(): 

        # if builder.scenario == 'coverage': 
        #     cluster.problem += (
        #         cluster.route_time[v] == pl.lpSum(
        #             cluster.tr_times[(i,j)] * cluster.x[i,j,v]
        #             for i in V_nodes for j in V_nodes if i != j
        #         ) + builder.coverage_time * pl.lpSum(cluster.wait[v,t] for t in cluster.timeframe)
        #     )

            # cluster.problem += cluster.route_time[v] <= cluster.T_MAX
    #    elif builder.scenario == 'cooperative':
    #         for j in V_nodes: 
    #             cluster.problem += pl.lpSum(
    #                 cluster.arrive[j,v,t] for t in cluster.timeframe 
    #             ) - 1 <= cluster.y[j,v]

        # for i in V_nodes: 
        #     for t in cluster.timeframe: 
        #         if t - 1 < 0 : continue 
        #         cluster.problem += cluster.arrive >= cluster.atNode[i,v,t] - cluster.atNode[i,v,t-1]
        #         cluster.problem += cluster.arrive >= cluster.atNode[i,v,0]

        # cluster.problem += cluster.atNode[depot_ind,v,cluster.timeframe[0]] == 1 
        # for i in V_nodes:
        #     if i == depot_ind: continue
        #     cluster.problem += cluster.atNode[i,v,cluster.timeframe[0]] == 0 

        # cluster.problem += cluster.e[depot_ind,v] == builder.max_battery

        # LINK y_dep and x 
        for i in V_nodes: 
            for j in V_nodes: 
                if i == j : continue 
                cluster.problem += pl.lpSum(
                    cluster.depart[i,j,v,t] for t in cluster.timeframe 
                    if t + cluster.tr_times[(i,j)]<= cluster.timeframe[-1]
                ) == cluster.x[i,j,v]

        # Temporal State propagation 
        for t in cluster.timeframe: 
            cluster.problem += cluster.busy[v,t] + cluster.wait[v,t] == 1 

        for i in V_nodes:
            for j in V_nodes: 
                if i == j : continue 
                travel_duration = cluster.tr_times[(i, j)]
                max_start = len(cluster.timeframe) - travel_duration

                for t in cluster.timeframe[:max_start]:
                    if t + travel_duration - 1 > cluster.timeframe[-1]:continue
                    for d in range(travel_duration):
                        t_arrival = t + d
                        cluster.problem += cluster.busy[v,t_arrival] >= cluster.depart[i,j,v,t]

        # for t in cluster.timeframe: 
        #     cluster.problem += cluster.wait[v,t] == pl.lpSum(
        #         cluster.depart[i,j,v,taf] 
        #         for i in V_nodes for j in V_nodes if i != j and j != depot_ind
        #         for taf in cluster.timeframe 
        #         if taf + cluster.tr_times[(i,j)] == t and (i,j,v,taf) in cluster.depart
        #     )  

                    cluster.problem += cluster.wait[v, t + cluster.tr_times[(i, j)]] >= cluster.depart[i, j, v, t]
                    cluster.problem += cluster.atNode[j,v,t + cluster.tr_times[(i, j)]] >= cluster.depart[i,j,v,t]

        for j in V_nodes: 
            cluster.problem += pl.lpSum(cluster.atNode[j,v,t] for t in cluster.timeframe) >= 1
        
        for i in V_nodes: 
            source = cluster.nodes_dict[i]
            for j in V_nodes: 
                if i == j : continue 
                target = cluster.nodes_dict[j] 
                cluster.problem += cluster.e[j,v] >= cluster.e[i,v] - builder.move_energy[source][target] - M * (1 - cluster.x[i,j,v])
                cluster.problem += cluster.e[i,v] >= builder.move_energy[source][target] - M * (1 - cluster.x[i,j,v])
            cluster.problem += cluster.e[i,v] >= builder.move_energy[source][cluster.depot_id] * cluster.x[i,depot_ind,v]

        for t in cluster.timeframe: 
            for i in V_nodes: 
                if i == depot_ind: continue 
                cluster.problem += cluster.e[j,v] >= cluster.e[i,v] - builder.average_coverage_energy * cluster.wait[v,t]
                cluster.problem += cluster.e[i,v] >= builder.average_coverage_energy * cluster.wait[v,t]

        for i in V_nodes: 
            if i == depot_ind: 
                cluster.problem += cluster.e[i,v] >= 0 
                cluster.problem += cluster.e[i,v] >= builder.descend_energy.loc[cluster.depot_id, cluster.nodes_dict[i]] * cluster.x[i, depot_ind, v]
 
        cluster.problem += pl.lpSum(
                cluster.depart[depot_ind, j, v, cluster.timeframe[0]] 
                for j in V_nodes if depot_ind != j
            ) == 1
        

        # coverage 
        for j in V_nodes: 
            cluster.problem += pl.lpSum(cluster.atNode[j,v,t] for t in cluster.timeframe) >= 1 

        # Static flow constraints 
        cluster.problem += pl.lpSum(cluster.x[depot_ind, j, v] for j in V_nodes if j != depot_ind) == 1
        cluster.problem += pl.lpSum(cluster.x[j, depot_ind, v] for j in V_nodes if j != depot_ind) == 1
        cluster.problem += cluster.x[depot_ind,depot_ind,v] == 0 
        cluster.problem += pl.lpSum(cluster.x[depot_ind, j, v] for j in V_nodes if j != depot_ind) + \
                           pl.lpSum(cluster.x[j, depot_ind, v] for j in V_nodes if j != depot_ind) == 2 

        for i in V_nodes: 
            cluster.problem += pl.lpSum(cluster.x[i,j,v] for j in V_nodes if i!=j) == 1 
            cluster.problem += pl.lpSum(cluster.x[j,i,v] for j in V_nodes if i!=j) == 1
        
        for j in V_nodes: 
            if j == depot_ind: continue 
            cluster.problem += cluster.x[depot_ind, j, v] + cluster.x[j, depot_ind, v] <= 1
            
        for j in V_nodes: 
            if j == depot_ind: continue 
            cluster.problem += cluster.visit[j,v] >= pl.lpSum(
                cluster.depart[i,j,v,t]
                for i in V_nodes if i != j and i != depot_ind 
                for t in cluster.timeframe 
                if t + cluster.tr_times[(i,j)] < cluster.timeframe[-1]
            )

        cluster.problem += pl.lpSum(
            cluster.depart[i, depot_ind, v, t]
            for i in V_nodes if i != depot_ind 
            for t in cluster.timeframe 
            if t + cluster.tr_times[(i,depot_ind)] <= cluster.timeframe[-1]
        ) == 1

        for i in V_nodes: 
            for j in V_nodes: 
                if j == i : continue 
                if i == depot_ind or j == depot_ind : continue 
                cluster.problem += pl.lpSum(
                    cluster.depart[i,j,v,t]
                    for t in cluster.timeframe[:-cluster.tr_times[(i,j)]]
                ) == cluster.x[i,j,v] 

        cluster.problem += pl.lpSum(cluster.depart[depot_ind,j,v,cluster.timeframe[0]] for j in V_nodes if j!= depot_ind) == 1 
        
        u = pl.LpVariable.dicts(f"u{k}", V_nodes, 1, len(V_nodes)-1, cat='Integer')
        for i in V_nodes: 
            for j in V_nodes: 
                if i == j or i == depot_ind or j == depot_ind: continue 
                cluster.problem += u[i] - u[j] + (len(V_nodes)-1) * cluster.x[i,j,v] <= (len(V_nodes)-1)-1
        
        routeT = pl.LpVariable.dicts("routeTime", cluster.employed_agents, 0)

        # cluster.problem += routeT[v] == pl.lpSum(cluster.busy[v,t] + cluster.wait[v,t] for t in cluster.timeframe) 
        # cluster.problem += routeT[v] <= cluster.timeframe[-1]
        # cluster.problem += routeT[v] <= cluster.T_MAX
    
    for i in range(len(cluster.employed_agents)-1): 
        v1 = cluster.employed_agents[i]
        v2 = cluster.employed_agents[i+1] 

        cluster.problem += pl.lpSum(cluster.x[i,j,v1] for i in V_nodes for j in V_nodes if i != j) == \
                            pl.lpSum(cluster.x[i,j,v2] for i in V_nodes for j in V_nodes if i != j)


    for j in V_nodes: 
        if j == depot_ind: continue 
        cluster.problem += pl.lpSum(
            cluster.x[depot_ind,j,v]
            for v in list_of_agents.values()
        ) <= 1  

        cluster.problem += pl.lpSum(
            cluster.x[j, depot_ind, v]
            for v in list_of_agents.values() 
        ) <= 1 

    cluster.problem += pl.lpSum(
        cluster.x[depot_ind, i, v]
        for i in V_nodes
        for v in list_of_agents.values()
    ) == len(cluster.employed_agents) 

    cluster.problem += pl.lpSum(
        cluster.x[i, depot_ind, v]
        for i in V_nodes
        for v in list_of_agents.values()
    ) == len(cluster.employed_agents) 

    for i in V_nodes: 
        for j in V_nodes: 
            if i == j or i == depot_ind or j == depot_ind: continue 
            for taf in cluster.timeframe: 
                if taf + cluster.tr_times[(i,j)] > cluster.timeframe[-1]: continue 
                cluster.problem += pl.lpSum(
                    cluster.depart[i,j,v,taf] for v in list_of_agents.values()
                ) <= 1




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

    for j in V_nodes: 
        if j == depot_ind: continue 
        cluster.problem.addConstraint(
            name=f"Only_one_to_j_depot_for_node_{j}",
            constraint= pl.lpSum(cluster.x[j,depot_ind,v] for v in list_of_agents.values() ) <= 1,
        )    

    # Correct 
    cluster.problem.addConstraint(
        name=f"exactly_{len(cluster.employed_agents)}_depart_from_depot",
        constraint=pl.lpSum(cluster.x[depot_ind,i,v] for i in V_nodes for v in list_of_agents.values()) == len(cluster.employed_agents)
    )

    # Correct 
    cluster.problem.addConstraint(
        name=f"exactly_{len(cluster.employed_agents)}_return_to_depot",
        constraint=pl.lpSum(cluster.x[i,depot_ind,v] for i in V_nodes for v in list_of_agents.values()) == len(cluster.employed_agents)
    )

    logger.debug(f"Constraint 1: {len(cluster.problem.constraints)}")        


def constraint_2(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Allow an agent to start his journey whenever it fits best and finish as well at a different time that before (not time frame [-1])
    """
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    for k, v in list_of_agents.items(): 

        # Correct
        cluster.problem.addConstraint(
            name=f"{k}_leaves_depot_{depot_ind}_at_specific_interval", 
            constraint=pl.lpSum(
                cluster.y_depart[depot_ind, j, v, cluster.timeframe[0]] 
                for j in V_nodes if depot_ind != j
            ) == 1,
        ) 

        
        # Correct 
        cluster.problem.addConstraint(
            name=f"{k}_return_depot_{depot_ind}_at_dynamic_interval",
            constraint = pl.lpSum(
                            cluster.y_depart[i, depot_ind, v, t]
                            for i in V_nodes if i != depot_ind
                            for t in cluster.timeframe
                            if t + cluster.tr_times[(i,depot_ind)] <= cluster.timeframe[-1]
                         ) == 1,
        )

        # Correct 
        for t in cluster.timeframe:
            cluster.problem += pl.lpSum(
                cluster.y_depart[i,j,v,t] for i in V_nodes for j in V_nodes if i!=j
            ) <= 1, f"Single_active_leg_{k}_at_{t}"


        # # Correct
        # cluster.problem.addConstraint(
        #     name=f"{k}{depot_ind}_at_specific_interval", 
        #     constraint=pl.lpSum(cluster.t[depot_ind, j, v, cluster.timeframe[0]] for j in V_nodes if depot_ind != j ) == 1,
        # )   

        # cluster.problem += pl.lpSum(
        #     cluster.t[i, depot_ind, v, t]
        #     for i in V_nodes if i != depot_ind
        #     for t in cluster.timeframe[:-(max(cluster.tr_times[(i, depot_ind)], 1))]
        # ) >= 1, f"{k}_returns_to_depot_once"


  
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
            if cluster.nodes_dict[i] not in cluster.bridge_nodes: 
                cluster.problem.addConstraint(
                        name=f"Only_one_visit_from_i_to_j_for_agent_{k}_for_node_{i}",
                        constraint= pl.lpSum(cluster.x[i,j,v] for j in V_nodes if i != j) == 1,
                    )
            
            # Correct 
            if cluster.nodes_dict[i] in cluster.bridge_nodes :
                cluster.problem.addConstraint(
                    name=f"Bridge_NODES_from_i_to_j_for_agent_{k}_for_node_{i}",
                    constraint=pl.lpSum(cluster.x[i,j,v] for j in V_nodes if i != j) >= 1,
                )

        
        for j in V_nodes:

            # Correct 
            if cluster.nodes_dict[j] not in cluster.bridge_nodes: 
                cluster.problem.addConstraint(
                    name=f"Only_one_visit_from_j_to_i_for_agent_{k}_for_node_{j}",
                    constraint= pl.lpSum(cluster.x[i,j,v] for i in V_nodes if i != j ) == 1,
                )
            
            # Correct 
            if cluster.nodes_dict[j] in cluster.bridge_nodes :
                cluster.problem.addConstraint(
                    name=f"Bridge_NODES_from_j_to_i_for_agent_{k}_for_node_{j}",
                    constraint= pl.lpSum(cluster.x[i,j,v] for i in V_nodes if i != j) >= 1,
                )
                

    # Enforce Full Cluster Coverage (Coolectively)
    if builder.scenario == 'cooperative': 
        for j in V_nodes:
            if cluster.nodes_dict[j] in cluster.bridge_nodes : continue 
            cluster.problem += (
                pl.lpSum(cluster.x[i, j, v] for i in V_nodes if i != j for v in list_of_agents.values() ) == len(list_of_agents),
                f"At_least_one_agent_visit{j}"
            )

        for j in V_nodes:
            if cluster.nodes_dict[j] in cluster.bridge_nodes : continue 
            cluster.problem += (
                pl.lpSum(cluster.x[j,i,v] for i in V_nodes if i != j for v in list_of_agents.values() ) == len(list_of_agents),
                f"At_least_one_agent_exit_{j}"
            )


    logger.debug(f"Constraint 5: {len(cluster.problem.constraints)}")


def constraint_6(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Collision avoidance / Unique agent per node 
    """    
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    for i in V_nodes: 
        for j in V_nodes: 
            for step in cluster.timeframe: 
               cluster.problem.addConstraint(
                   name=f"Unique_Time_visits_constraint_at_travel_{i}_{j}_at_time_{step}",
                   constraint= pl.lpSum(cluster.t[i,j,v,step] for _,v in list_of_agents.items()) >= len(list_of_agents) 
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
    for k, v in list_of_agents.items():
        for i in V_nodes: 
            for j in V_nodes: 
                if i == depot_ind or j == depot_ind: continue
                cluster.problem += pl.lpSum(cluster.y_depart[i,j,v,t] for t in cluster.timeframe) == cluster.x[i,j,v] 

               
    # Suggestion don't use the T_MAX as it prevents later steps from being checked while it should check feasibility over the full timeline, not against the max travel time to the depot. 
    # Missing depot transitions which may be intentional - but it leaves depot timing uncontrolled unless differenct synchronization constraint is used. 
    for k, v in list_of_agents.items(): 
        for i in V_nodes: 
            for j in V_nodes: 
                if i == depot_ind or j == depot_ind: continue 
                
                d = cluster.tr_times[(i,j)]
                for t in cluster.timeframe: 
                    if t + d > cluster.timeframe[-1]: continue
                    cluster.problem.addConstraint(
                        name=f"Synchronization_time_and_space_{i,j}_for_agent_{k}_for_{t}",
                        constraint = pl.lpSum(
                                        cluster.t[i,j,v,step] for step in range(t, t + d)
                                     ) == d * cluster.y_depart[i,j,v,t],
                        
                    )

    logger.debug(f"Constraint 8: {len(cluster.problem.constraints)}")



def constraint_9(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    """
    Only one travel from i to j for the entirety of the time frame. This cannotbe used if we opt to align all time steps individually. 
    """

    # NOTE : Minor Effect ~ 1000 constraints 
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    
    for k, v in list_of_agents.items(): 
        cluster.problem += pl.lpSum(cluster.y_depart[depot_ind, j, v, t]
                                    for j in V_nodes if j != depot_ind
                                    for t in cluster.timeframe[:-(cluster.tr_times[(j,depot_ind)])])
    
    for k, v in list_of_agents.items(): 
        # NOTE: This constraint includes duration for the travel. 
        cluster.problem.addConstraint(
            name=f"Only_one_travel_from_depot_to_j_for_agent_{k}_based_on_time",
            constraint= pl.lpSum(cluster.t[depot_ind, j, v, t]
                            for j in V_nodes if j != depot_ind 
                            for t in cluster.timeframe[:-(cluster.tr_times[(j, depot_ind)])]
                        ) >= 1
        )

        cluster.problem.addConstraint(
            name=f"Only_one_travel_from_j_to_depot_for_agent_{k}_based_on_time",
            constraint= pl.lpSum(cluster.t[j,depot_ind,v,t]
                            for j in V_nodes if j != depot_ind 
                            for t in cluster.timeframe[:-(cluster.tr_times[(j, depot_ind)] )]
                        ) >= 1
        )

    logger.debug(f"Constraint 9: {len(cluster.problem.constraints)}")


def constraint_10(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):

    BIG_M = len(V_nodes)+len(cluster.timeframe)
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    if builder.scenario == "coverage": 
        # Correct For individual case 
        for k,v in list_of_agents.items():
            for i in V_nodes:
                cluster.problem.addConstraint(
                    name=f"Agent_{v}_must_visit_node_{i}",
                    constraint=cluster.visit[i,v] == 1 
                )

    elif builder.scenario == "cooperative": 
        for i in V_nodes: 
            cluster.problem += pl.lpSum(cluster.visit[i,v] for v in list_of_agents.values()) >= 1 
    
    
    for k, v in list_of_agents.items() : 
        for i in V_nodes: 
            # if i == depot_ind or cluster.nodes_dict[i] in cluster.bridge_nodes :continue 
            cluster.problem += (
                pl.lpSum(cluster.x[i,j,v] for j in V_nodes if i != j ) +
                pl.lpSum(cluster.x[j,i,v] for j in V_nodes if i != j ) <= BIG_M * cluster.visit[i,v]
            )

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
   
    visit_counts = [pl.lpSum(cluster.visit[i, v] for i in V_nodes) for k, v in list_of_agents.items()]
    for a in visit_counts:
        for b in visit_counts:
            cluster.problem += a <= b + 1  # Keep agent loads balanced
    

    logger.debug(f"Constraint 11: {len(cluster.problem.constraints)}")

    # The != 0 is not linear hence it might not be correctly interpreted by the solver. It is reformulated using binary auxiliary, big-M or indicator constraints. 
    # this also overconstrains the system. 


def constraint_12(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)

    M = builder.max_battery

    # NOTE : Try for all nodes: Before out_arcs 
    for k, v in list_of_agents.items():
        for i in V_nodes:
            source = cluster.nodes_dict[i]-1
            for j in V_nodes:
                if i == j: continue 
                target = cluster.nodes_dict[j]-1

                cluster.problem.addConstraint(
                    name=f"Update_remaining_energy_{i}_{j}_for_{k}",
                    constraint=cluster.e[j,v] >= cluster.e[i,v] 
                          - builder.move_energy[source][target] - M * (1-cluster.x[i,j,v])
                        #   - ascend_energy * cluster.x[depot_ind,i,v],
                )

                cluster.problem.addConstraint(
                    name=f"No_travel_if_low_energy_{i}_{j}_for_{k}",
                    constraint=cluster.e[i,v] >= builder.move_energy[source][target] - M *(1 - cluster.x[i, j, v])
                )

            cluster.problem.addConstraint(
                name=f"Enough_energy_to_return_to_depot_from_{i}_for_{k}",
                constraint=cluster.e[i,v] >= builder.move_energy[source][cluster.nodes_dict[depot_ind]] * cluster.x[i, depot_ind, v],
            )

        # if builder.scenario == 'coverage':
        for t in cluster.timeframe: 
            for i in V_nodes: 
                if i == depot_ind: continue 
                # Subtract coverage energy from remaining battery
                cluster.problem.addConstraint(
                    name=f"Update_remaining_energy_comm_{i}_at_{t}_for_{k}",
                    constraint=cluster.e[i, v] >= cluster.e[i, v] - builder.average_coverage_energy * cluster.wait[v, t],
                )

                # Optional: ensure energy is enough before waiting
                cluster.problem.addConstraint(
                    name=f"No_wait_if_low_energy_{i}_at_{t}_for_{k}",
                    constraint=cluster.e[i, v] >= builder.average_coverage_energy * cluster.wait[v, t],
                )

               
    logger.debug(f"Constraint 12: {len(cluster.problem.constraints)}")


def constraint_13(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    
    for k, v in list_of_agents.items():
        for i in V_nodes:
            if i == depot_ind: continue

            cluster.problem.addConstraint(
                name=f"Energy_cannot_be_negative_{i}_for_{k}",
                constraint=cluster.e[i,v] >= 0,
            )

            cluster.problem.addConstraint(
                name=f"Enough_energy_to_descend_at_depot_from_{i}_for_{k}",
                constraint=cluster.e[i,v] >= builder.descend_energy.loc[cluster.depot_id, cluster.nodes_dict[i]] * cluster.x[i, depot_ind, v]
            )

        cluster.problem.addConstraint(
            name=f"Every_agent_starts_with_full_battery_{k}",
            constraint= cluster.e[depot_ind, v] == builder.max_battery 
        )

            # cluster.problem.addConstraint(
            #     name=f"Every_agent_has_to_return_with_enough_battery_{k}_from_{i}",
            #     constraint= pl.lpDot(cluster.e[depot_ind,v], cluster.x[i,depot_ind,v])>= 0                            
            # )
            

    logger.debug(f"Constraint 13: {len(cluster.problem.constraints)}")


def constraint_14(cluster:Any, builder:Any, V_nodes:list, list_of_agents:dict): 
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    M = len(V_nodes)*len(cluster.timeframe)

    for k, v in list_of_agents.items():

        for t in cluster.timeframe:
            cluster.problem += cluster.p[v, t] == depot_ind + (1 - (t == cluster.return_step[v])) * M, \
                f"Return_alignment_at_time_{t}_for_agent_{k}"
        
        
        cluster.problem.addConstraint(
            name=f"Positional_variable_at_start_of_journey_for_{k}",
            constraint=cluster.p[v,cluster.timeframe[0]] == depot_ind, 
        )

            
    logger.debug(f"Constraint 14: {len(cluster.problem.constraints)}")


def constraint_15(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    
    for k, v in list_of_agents.items():
        for i in V_nodes:
            for j in V_nodes: 
                if i == j : continue 
                travel_duration = cluster.tr_times[(i, j)]
                max_start = len(cluster.timeframe) - travel_duration

                for t in cluster.timeframe[:max_start]:
                    if t + travel_duration - 1 > cluster.timeframe[-1]:continue
                    for d in range(travel_duration):
                        t_arrival = t + d
                        cluster.problem.addConstraint(
                            name=f"Busy_if_travel_{i}_{j}_starts_at_{t_arrival}_for_{k}_covers_{t}",
                            constraint=cluster.busy[v,t_arrival] >= cluster.y_depart[i,j,v,t]
                        )
                    
                    cluster.problem += cluster.wait[v, t + cluster.tr_times[(i, j)]] >= cluster.y_depart[i, j, v, t]

        # for i, j in valid_pairs:
        #     travel_duration = cluster.tr_times[(i, j)]
        #     max_start = len(cluster.timeframe) - travel_duration

        #     for t_start in cluster.timeframe[:max_start]:
        #         t_end = t_start + travel_duration 

        #         for dt in range(travel_duration):
        #             t = t_start + dt 

        #             cluster.problem.addConstraint(
        #                 name=f"Busy_if_travel_{i}_{j}_starts_at_{t_start}_for_{k}_covers_{t}",
        #                 constraint=cluster.busy[v,t] <= cluster.y_depart[i,j,v,t_start],
        #             )

        #         if t_end < cluster.timeframe[-1]:
        #             cluster.problem.addConstraint(
        #                 name=f"Wait_after_busy_{i,j,t_start}_for_{k}_covers_{t_end}", 
        #                 constraint=cluster.wait[v, t_end] >= cluster.y_depart[i, j, v, t_start]
        #             )

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
    # be at depot at t = 0
    for k,v  in list_of_agents.items():
        cluster.problem += cluster.t[depot_ind, depot_ind, v, cluster.timeframe[0]] == 1
        cluster.problem += pl.lpSum(cluster.y_depart[depot_ind,j,v,cluster.timeframe[0]] for j in V_nodes if j!= depot_ind) == 1 


    # arrival turns depot-stay on
    for k ,v in list_of_agents.items():
        for i in V_nodes:
            if i == depot_ind: continue
            for t in cluster.timeframe:
                d = cluster.tr_times[(i,depot_ind)]
                if t + d <= cluster.timeframe[-1]:
                    cluster.problem += cluster.t[depot_ind, depot_ind, v, t+d] >= cluster.y_depart[i, depot_ind, v, t]


    for k in list_of_agents.items():
        for j in V_nodes:
            if j == depot_ind: continue
            cluster.problem += pl.lpSum(cluster.y_depart[depot_ind, j, v, t] for t in cluster.timeframe[:-cluster.tr_times[(depot_ind,j)]]) == cluster.x[depot_ind, j, v]

            cluster.problem += pl.lpSum(cluster.y_depart[j, depot_ind, v, t] for t in cluster.timeframe
                                if t + cluster.tr_times[(j,depot_ind)] <= cluster.timeframe[-1]) == cluster.x[j, depot_ind, v]


    for k,v in list_of_agents.items():
        for j in V_nodes: 
            if j == depot_ind: continue 
            d = cluster.tr_times[(depot_ind,j)]
            for t in cluster.timeframe: 
                if t + d - 1 <= cluster.timeframe[-1]: 
                    cluster.problem += pl.lpSum(cluster.t[depot_ind,j,v,step] for step in range (t, t+d)) == d * cluster.y_depart[depot_ind, j, v, t]
                    
            cluster.problem += pl.lpSum(cluster.y_depart[depot_ind,j,v,t] for t in cluster.timeframe) == cluster.x[depot_ind,j,v]


    
    #Per agent 
    logger.debug(f"Constraint 17: {len(cluster.problem.constraints)}")


def constraint_18(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    # Symmetry-breaking constraint 
    for i in range(len(cluster.employed_agents) - 1):
        v1 = cluster.employed_agents[i]
        v2 = cluster.employed_agents[i + 1]

        cluster.problem += pl.lpSum(
            cluster.x[i, j, v1] for i in V_nodes for j in V_nodes if i != j
        ) == pl.lpSum(
            cluster.x[i, j, v2] for i in V_nodes for j in V_nodes if i != j
        ), f"Symmetry_break_by_edge_count_{v1}_vs_{v2}"

    logger.debug(f"Constraint 18: {len(cluster.problem.constraints)}")


def constraint_19(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    for k,v in list_of_agents.items(): 
        for i in V_nodes: 
            for j in V_nodes: 
                # if i == j: continue 
                if i == cluster.depot_id or j == cluster.depot_id: continue
                for t in cluster.timeframe: 
                    if t + cluster.tr_times[(i,j)] in cluster.timeframe:
                        cluster.problem.addConstraint(
                            name=f"Busy_constraint_{i}_{j}_for_{k}_at_time_{t}", 
                            constraint=cluster.t[i, j, v, t] >= cluster.busy[v, t + cluster.tr_times[(i,j)]],  
                        )

    logger.debug(f"Constraint 19: {len(cluster.problem.constraints)}")


def constraint_20(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    # NOTE: May be redundant since 15 also handles wait after busy
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    
    for k, v in list_of_agents.items(): 
        for i in V_nodes:
            for j in V_nodes:
                # if i == j: continue
                if i == depot_ind or j == depot_ind: continue 
                travel_duration = cluster.tr_times[(i, j)]
                for t in cluster.timeframe:
                    wait_time = t + travel_duration

                    if wait_time in cluster.timeframe:
                        # Wait one step after finishing the trip from i to j
                        cluster.problem.addConstraint(
                            name=f"Strict_wait_after_travel_{i}_{j}_start_{t}_for_agent_{k}", 
                            constraint=cluster.wait[v, wait_time] >= cluster.y_depart[i, j, v, t], 
                        )

                            

def constraint_21(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    # This is a good approach however a better approach would be to use an integer decision variable 
    # that penalizes every repeat visit not just presence. 
    
    for k, v in list_of_agents.items(): 
        for j in V_nodes: 
            cluster.problem.addConstraint(
                name=f"Penalize_repeat_visits_{j}_for_{k}", 
                constraint= pl.lpSum(
                                cluster.t[i,j,v,t]
                                for i in V_nodes if i != j 
                                for t in cluster.timeframe
                            ) <= cluster.y[j,v] 
            )


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
    MIN_VISITS = np.round(len(V_nodes) / len(cluster.employed_agents)) -1 

    for k, v in list_of_agents.items(): 
        if len(list_of_agents) == 1: 
            MIN_VISITS = len(cluster.nodes_dict)-1
            cluster.problem += pl.lpSum(
                cluster.visit[i,v] for i in V_nodes if i != depot_ind and cluster.nodes_dict[i] not in cluster.bridge_nodes
            ) >= MIN_VISITS, f"Min_coverage_for_agent_{k}"
        else: 
            cluster.problem += pl.lpSum(
                cluster.visit[i,v] for i in V_nodes if i != depot_ind and cluster.nodes_dict[i] not in cluster.bridge_nodes
            ) >= MIN_VISITS, f"Min_coverage_for_agent_{k}"

    for k, v in list_of_agents.items():
        for i in V_nodes:
            if i == depot_ind or cluster.nodes_dict[i] in cluster.bridge_nodes: continue
            cluster.problem += cluster.visit[i, v] >= pl.lpSum(cluster.x[i, j, v] for j in V_nodes if j != i and j != depot_ind and cluster.nodes_dict[j] not in cluster.bridge_nodes), \
                f"Visit_tracking_{i}_agent_{k}"
            
    # for j in V_nodes: 
    #     cluster.problem += pl.lpSum(cluster.x[i,j,v] for i in V_nodes for v in list_of_agents.values()) >= len(cluster.nodes_dict)
            
    logger.debug(f"Constraint 24: {len(cluster.problem.constraints)}")


def constraint_25(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    
    target_edge_count = len([j for j in V_nodes if j != depot_ind])

    for k, v in list_of_agents.items():
        cluster.problem += pl.lpSum(cluster.x[i,j,v] for i in V_nodes for j in V_nodes if i != j) >= target_edge_count, f"{k}_has_same_edge_count"
    
    logger.debug(f"Constraint 25: {len(cluster.problem.constraints)}")


def constraint_26(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    
    for k,v in list_of_agents.items():
        for j in V_nodes:
            if j == depot_ind: continue    # don’t count depot
            cluster.problem += cluster.visit[j,v] >= pl.lpSum(
                cluster.y_depart[i,j,v,t]
                for i in V_nodes if i != j
                for t in cluster.timeframe
                if t + cluster.tr_times[(i,j)] <= cluster.timeframe[-1]
            ), f"Visit_{j}_{k}"
    # Only set visit when agent arrives at node j after trip i → j completes
    # for k, v in list_of_agents.items(): 
    #     for i in V_nodes: 
    #         if i == depot_ind: continue 
    #         for j in V_nodes: 
    #             if j == depot_ind: continue
    #             if i == j : continue 

    #             for t_start in cluster.timeframe : 
    #                 if t_start + cluster.tr_times[(i,j)] > cluster.timeframe[-1]: continue 
    #                 cluster.problem += cluster.visit[j,v] >= cluster.y_depart[i,j,v,t_start], f"Visit_tracking_{i}_{j}_agent_{k}_for_{t_start}"


    logger.debug(f"Constraint 26: {len(cluster.problem.constraints)}")


def constraint_27(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    
    for k, v in list_of_agents.items():
        for i in V_nodes:
            for j in V_nodes:
                if i == j: continue

                for t in cluster.timeframe:
                    d = cluster.tr_times[(i,j)]
                    arrival_time = t + d

                    if arrival_time <= cluster.timeframe[-1]:
                        cluster.problem += cluster.y_depart[i, j, v, t] <= cluster.t[j, j, v,arrival_time], \
                            f"TimeProgress_{i}_{j}_at_{t}_agent_{k}" 
                            
    logger.debug(f"Constraint 27 : {len(cluster.problem.constraints)}")


def constraint_28(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    # depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    depot_ind = get_depot_node(cluster.depot_id, cluster.nodes_dict)
    n = len(V_nodes) -1 
    cluster.u = pl.LpVariable.dicts("u", ((i,v) for i in V_nodes for v in cluster.employed_agents), lowBound=1,upBound=n-1, cat='Integer')
    # L = sum(cluster.R_points) # Maximum number of visits 
    # K = np.round(len(V_nodes) / len(cluster.employed_agents)) -1 

    # for k, v in list_of_agents.items():
    #     for i in V_nodes:
    #         if i == depot_ind: continue 
    #         cluster.problem.addConstraint(
    #             name=f"Upper_Bound_nodes_visit_constraints_{k}_for_{i}",
    #             constraint=cluster.u[i,v] + pl.lpDot((L-2),cluster.x[depot_ind,i,v]) - cluster.x[i,depot_ind,v] <= L - 1
    #         )

    #         cluster.problem.addConstraint(
    #             name=f"Lower_Bound_nodes_visit_constraints_{k}_for_{i}",
    #             constraint=cluster.u[i,v] + cluster.x[depot_ind,i,v] + pl.lpDot(cluster.x[i,depot_ind,v],(2-K)) >= 2
    #         )

    #         for j in V_nodes: 
    #             if i == j: continue 
    #             if j == depot_ind: continue 
    #             cluster.problem.addConstraint(
    #                 name = f"Inequality_ensurance_{i,j}_for_{k}", 
    #                 constraint= cluster.u[i,v] - cluster.u[j,v] + pl.lpDot(L,cluster.x[i,j,v]) + pl.lpDot((L - 2),cluster.x[j,i,v]) <= L - 1
    #             )

    for k, v in list_of_agents.items(): 
        cluster.problem += cluster.u[depot_ind,v] == 0, f"u_depot_{v}"
        for i in V_nodes: 
            if i == depot_ind: continue 
            for j in V_nodes:  
                if j == depot_ind: continue 

                cluster.problem += (cluster.u[i,v] - cluster.u[j,v] + (n-1) * cluster.x[i,j,v] <= n-2), f"MTZ_{i,j}_{k}"


    logger.debug(f"Constraint 28: {len(cluster.problem.constraints)}")


def constraint_29(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):
    cluster.route_time = pl.LpVariable.dicts("route_time", ((v) for v in cluster.employed_agents), lowBound=0, cat='Continuous')
    for v in list_of_agents.values(): 
        cluster.problem += (
            cluster.route_time[v] == pl.lpSum(
                cluster.tr_times[(i,j)] * cluster.x[i,j,v]
                for i in V_nodes for j in V_nodes if i != j
            ) + builder.coverage_time * pl.lpSum(cluster.wait[v,t] for t in cluster.timeframe)
        )

    for v in list_of_agents.values(): 
        cluster.problem += cluster.route_time[v] <= cluster.T_MAX

    logger.debug(f"Constraint 29: {len(cluster.problem.constraints)}")


def constraint_30(cluster:Any, builder:Any , V_nodes:list, list_of_agents:dict):


    logger.debug(f"Constraint 30: {len(cluster.problem.constraints)}")

