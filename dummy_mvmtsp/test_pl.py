from pulp import LpProblem, LpMinimize, lpSum, LpBinary, LpStatus, LpVariable
from dummy_mvmtsp.construct_mvmtsp import ConstrainedMVMTSP 
import numpy as np 
import warnings

def call_mvmtsp_constructor(*,enable_dictionary:bool = False, useGA:bool = True ,number_of_agents:int=5, max_battery:int=1500, max_memory:int=2*1024*1024*1024):
    
    problem = ConstrainedMVMTSP(data=None, use_GA=useGA, regionalization=True, dictionary_flag=enable_dictionary, max_memory=max_memory)
    
    dist_path = 'swarms/distance_cost.csv'
    energy_path = 'swarms/energy_cost.csv'
    nodes_path = 'swarms/areas.csv'
    customers = 'swarms/customers.csv'
    
    data = problem.preprocess(cost_1_path=dist_path, cost_2_path=energy_path,
                              nodes_path=nodes_path, agents=number_of_agents, 
                              customers_path=customers,
                              max_battery=max_battery)
    
    enabled_constraints = ['const_1', 'const_2', 'const_3', 'const_4',
                           'const_5', 'const_6', 'const_7', 'const_8',
                           'const_9']
    
    problem.run_model(data=data,enabled_constraints=enabled_constraints)
    print("Problem has been solved! Extracting paths...")
    
    solution = problem.get_solution()
    problem.get_performance() 
    
    return solution


def main(): 
    
    # try: 
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore", RuntimeWarning)
            max_memory = 2 * 1024 *1024 *1024
            call_mvmtsp_constructor(enable_dictionary=True, useGA=False, number_of_agents=3, max_battery=1500,max_memory=max_memory)
    # except: 
    #     print("Calling the constructor failed!")


if __name__ == "__main__":
    main()





    ### ---ALL CONSTRAINTS---###
    #Constraint 1: Each node must be entered and exitted once by each agent 
    # for k in agents: 
    #     d = depots_for_agents[k]
    #     for j in V_nodes: 
    #         if j != d:
    #             model += lpSum(x[i,j,k] for i in V_nodes if i != j) == 1, f"Enter_{j}_by_{k}"
    #             model += lpSum(x[j,i,k] for i in V_nodes if i != j) == 1, f"Exit_{j}_by_{k}"

    #Constraint 2: Subtour elimination constraints SEC 
    # for k in agents: 
    #     d = depots_for_agents[k]
    #     for i in V_nodes[1:]: 
    #         model += x[i, d, k] + x[d,i,k] <=1 

    # for k in agents:
    #     d = depots_for_agents[k]
    #     for i in V_nodes: 
    #         for j in V_nodes: 
    #             if i!=j and i != d and j != d : 
    #                 model +=  u[i, k] - u[j, k] + v * x[i, j, k] <= v - 1 

    # for k in agents: 
    #     for i in V_nodes: 
    #         if i != V_nodes: 
    #             model += u[i,k] >= 1 
    #             model += u[i,k] <= v - 1

    #Constraint 3: Many visits for each city 
    # for j in V_nodes : 
    #     model += lpSum(x[i,j,k] for i in V_nodes if i!=j for k in agents) <= R_points[j-1] , f'Min_visits_{j}_{k}' #Constraint 2 For Visits

    # Constraint 4: Allow only a single agent to be on top of a city. 
    # for step in TS:
    #     d = depots_for_agents[k]
    #     model += lpSum(t[i,j,k,step] for k in agents for i in V_nodes for j in V_nodes if i!=j and j !=d) <= 1,f"Single_Agent_at_at_time_{step}"
    
    # Constraint 5: Each agents begins its journey at full energy 
    # for k in agents:
    #     d = depots_for_agents[k] 
    #     model += e[d,k] >= theta - M * (lpSum(z[d,k])), f"EnergyFull_at_start_{d}_{k}"

    # Constraint 6: Travel corresponds to energy 
    # for k in agents: 
    #         d = depots_for_agents[k]
    #         for i in V_nodes: 
    #             for j in V_nodes: 
    #                 if i!=j: 
    #                     model += e[j,k] >= e[i,k] - (cost_e[i-1][j-1] * x[i,j,k]),f"Update_remaining_energy_{i}_{j}_{k}"
    #                     model += e[i,k] >= cost_e[i-1][j-1] * x[i,j,k],f"No_travel_if_low_energy{i}_{j}_for_{k}"
    #             if i == d: 
    #                 model += e[j,k] == theta - M * (lpSum(z[d,k])), f"EnergyFull_at_end_{i}_{k}"
                        

    #Constraint 7: Start and finish at the same depot. 
    # for k in agents: 
    #     d = depots_for_agents[k]
    #     model += lpSum(x[d,j,k] for j in V_nodes if j not in depots_for_agents) ==1, f"Start_at_depot_{k}"
    #     model += lpSum(x[j,d,k] for j in V_nodes if j not in depots_for_agents) ==1, f"Finish_at_depot_{k}"
    #     model += x[d,d,k] == 0, f"No_loop_at_depot_{k}_{d}"
    #     for i in depot_values:
    #         if i != d: 
    #             model += x[d,i,k] == 0, f"No_travel_from_{i}_at_depot_{d}_for_{k}"
    #             model += x[i,d,k] == 0, f"No_travel_to_{i}_at_depot_{d}_for_{k}"
    #             model += x[d,i,k] + x[i,d,k] <= 0, f"No_travel_{i}_at_{d}_for_{k}"


    # #Constraint 9: Prevent loops and unecessarry visits at depots. 
    # for k in agents: 
    #     for step in TS[:-1]: 
    #             d = depots_for_agents[k]
    #             model += t[d,d,k,step] == 0 
    #             model += t[d,d,k,step] + t[d,d,k,step+1] <= 1, f"No_loop_at_depot_{d}_{step}_{k}"
    #             model += lpSum(t[d,j,k,step] for j in depots_for_agents ) == 0, f"No_unnecessary_visits_at_depot_{d}_{step}_{k}"

    #Constraint 10: Ensure that a customer/city is served by an agent. 
    # for i in C[0:10] : 
    #     a = i.item()
    #     model += lpSum(x[j,a,k] for j in V_nodes for k in agents) >= z[a,k], f"ServiceConstraint_{a}_{k}"
    

    #Set initial solution based on the GA 
    # for a in range(len(initial_population)): 
    #     for i in range(len(initial_population[a])-1): 
    #         x[initial_population[a][i],initial_population[a][i+1],a+1].setInitialValue(1)

    



        # for k in agents:
        #     d = depots_for_agents[k]
        #     sum_start = sum(x[d, j, k].varValue for j in V_nodes if j != d)
        #     sum_finish = sum(x[j, d, k].varValue for j in V_nodes if j != d)
        #     print(f"Agent {k} starts from depot {d}: {sum_start}")
        #     print(f"Agent {k} returns to depot {d}: {sum_finish}")
        # Retrieve the paths
        



# ### ---ALL CONSTRAINTS---###
# #Constraint 1: Each node must be entered and exitted once by each agent 
# for k in agents: 
#     d = depots[k]
#     for j in V_nodes: 
#         if j != d:
#             model += lpSum(x[i,j,k] for i in V_nodes if i != j) == 1, f"Enter_{j}_by_{k}"
#             model += lpSum(x[j,i,k] for i in V_nodes if i != j) == 1, f"Exit_{j}_by_{k}"

# #Constraint 2: Subtour elimination constraints SEC 
# for k in agents: 
#     d = depots[k]
#     for i in V_nodes[1:]: 
#         model += x[i, d, k] + x[d,i,k] <=1 

# for k in agents:
#     d = depots[k]
#     for i in V_nodes: 
#         for j in V_nodes: 
#             if i!=j and i != d and j != d : 
#                 model +=  u[i, k] - u[j, k] + v * x[i, j, k] <= v - 1 

# for k in agents: 
#     for i in V_nodes: 
#         if i != V_nodes: 
#             model += u[i,k] >= 1 
#             model += u[i,k] <= v - 1

# #Constraint 3: Many visits for each city 
# for j in V_nodes : 
#     model += lpSum(x[i,j,k] for i in V_nodes if i!=j for k in agents) <= required_visits[j-1] , f'Min_visits_{j}_{k}' #Constraint 2 For Visits

# # Constraint 4: Allow only a single agent to be on top of a city. 
# for step in TS:
#     d = depots[k]
#     model += lpSum(t[i,j,k,step] for k in agents for i in V_nodes for j in V_nodes if i!=j and j !=d) <= 1,f"Single_Agent_at_at_time_{step}"
    
# # Constraint 5: Each agents begins its journey at full energy 
# for k in agents:
#     d = depots[k] 
#     model += e[d,k] >= theta - M * (lpSum(z[d,k])), f"EnergyFull_at_start_{d}_{k}"

# # Constraint 6: Travel corresponds to energy 
# for k in agents: 
#         d = depots[k]
#         for i in V_nodes: 
#             for j in V_nodes: 
#                 if i!=j: 
#                     model += e[j,k] >= e[i,k] - (energy[i-1][j-1] * x[i,j,k]),f"Update_remaining_energy_{i}_{j}_{k}"
#                     model += e[i,k] >= energy[i-1][j-1] * x[i,j,k],f"No_travel_if_low_energy{i}_{j}_for_{k}"
#             if i == d: 
#                 model += e[j,k] == theta - M * (lpSum(z[d,k])), f"EnergyFull_at_end_{i}_{k}"
                    

# #Constraint 7: Start and finish at the same depot. 
# for k in agents: 
#     d = depots[k]
#     model += lpSum(x[d,j,k] for j in V_nodes if j not in depots) >=1, f"Start_at_depot_{k}"
#     model += lpSum(x[j,d,k] for j in V_nodes if j not in depots) >=1, f"Finish_at_depot_{k}"
#     model += x[d,d,k] == 0, f"No_loop_at_depot_{k}_{d}"
#     for i in depot_values:
#         if i != d: 
#             model += x[d,i,k] == 0, f"No_travel_from_{i}_at_depot_{d}_for_{k}"
#             model += x[i,d,k] == 0, f"No_travel_to_{i}_at_depot_{d}_for_{k}"
#             model += x[d,i,k] + x[i,d,k] <= 0, f"No_travel_{i}_at_{d}_for_{k}"

# # # Contraint 8: Balancing enter and exits from a city for agent k.  
# # for k in agents:
# #     d = depots[k]
# #     for j in V_nodes:
# #         if j != d: 
# #             model += lpSum(x[j,i,k] for i in V_nodes if i != d) == lpSum(x[i,j,k] for i in V_nodes if i != d), f"Flow_conservation_{j}_{i}_by_{k}"
    

# # #Constraint 9: Prevent loops and unecessarry visits at depots. 
# for k in agents: 
#     for step in TS[:-1]: 
#             d = depots[k]
#             model += t[d,d,k,step] == 0 
#             model += t[d,d,k,step] + t[d,d,k,step+1] <= 1, f"No_loop_at_depot_{d}_{step}_{k}"
#             model += lpSum(t[d,j,k,step] for j in depots ) == 0, f"No_unnecessary_visits_at_depot_{d}_{step}_{k}"

# #Constraint 10: Ensure that a customer/city is served by an agent. 
# for i in C[0:10] : 
#     a = i.item()
#     model += lpSum(x[j,a,k] for j in V_nodes for k in agents) >= z[a,k], f"ServiceConstraint_{a}_{k}"
 
# #Set initial solution based on the GA 
# for a in range(len(initial_population)): 
#     for i in range(len(initial_population[a])-1): 
#         x[initial_population[a][i],initial_population[a][i+1],a+1].setInitialValue(1)

