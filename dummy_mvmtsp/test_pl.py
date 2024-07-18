from pulp import LpProblem, LpMinimize, lpSum, LpBinary, LpStatus, LpVariable
import pulp as pl 
import pandas as pd 
import numpy as np 
import pdb 
import resource
import timeout_decorator
from ga_solver import GASOL
import time 
import matplotlib.pyplot as plt
from esda.moran import Moran
from libpysal.weights import Queen, KNN
import seaborn
import pandas
import geopandas
import numpy
from sklearn.preprocessing import robust_scale
from k_means_constrained import KMeansConstrained
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import euclidean_distances

def set_memory_limit(max_mem):
    resource.setrlimit(resource.RLIMIT_AS, (max_mem, max_mem))

@timeout_decorator.timeout(3600) #Half an hour limit 
def solve_problem(problem):
    problem.solve(pl.GLPK(msg=False,options=['--mipgap','0.05'] ))


def assign_agents_to_areas(m,D): 
    agents_per_area = m // len(D) 
    remaining_areas = m % len(D) 
    area_agents = {area: agents_per_area for area in D}

    agent_assignments = {agent : 0 for agent in range(1,m+1)} 
    agent_counter = 1
    for area in D: 
        num_agents = area_agents[area]
        for _ in range(num_agents): 
            agent_assignments[agent_counter] = area
            agent_counter += 1 
        
    for i in range(remaining_areas): 
        area = D[i]
        agent_assignments[agent_counter] += area 
        agent_counter += 1

    return agent_assignments



# max_mem = 300 * 1024 * 1024 # 300 MB
# Set max Memory Allocation to 2 GBs and terminate the problem 
# in case this is exceeded
max_mem = 2 * 1024 *1024 *1024 #2 GB
set_memory_limit(max_mem)

agents = [1, 2, 3] # The agents set 
m = len(agents) # numb er of agents
M = 25 # Sufficient large integer to add to some constraint as a stable factor. 

V_nodes = list(range(22)) #The vertices set
v = len(V_nodes) #The number of vertices. 

V = pd.read_csv('../areas.csv')
v = len(V)
C = pd.read_csv('../customers.csv') #The cities set (only areas that contain customers)

distances = pd.read_csv('../distance_cost.csv') #The cost of each edge in the graph. 
dist_columns = [f'dist_{i}' for i in range(1,len(V)+1)]
distances.columns = dist_columns 

energy  = pd.read_csv('../energy_cost.csv') #Cost of energy betwween every pair of nodes. 
ee_columns = [f'ee_{i}' for i in range(1,len(V)+1)]
energy.columns = ee_columns

required_visits = m * np.ones(v).astype(int) #The number of required visits for every single area. 
rv = pd.DataFrame({"R":required_visits[:]}) #The required visits dataframe.

y = [distances, energy, rv, V]
y = pd.concat(y, axis=1, join="inner")

gdf = geopandas.GeoDataFrame(y, geometry=geopandas.points_from_xy(y.X_coords, y.Y_coords), crs="EPSG:4326")
print(gdf.head())   

# w = Queen.from_dataframe(gdf)


cluster_variables = [
     #maximum number of noeds that a cluster can have
    "dist_1","dist_2","dist_3","dist_4","dist_5","dist_6","dist_7","dist_8","dist_9","dist_10","dist_11","dist_12",
    "dist_13","dist_14","dist_15","dist_16","dist_17","dist_18","dist_19","dist_20","dist_21","dist_22","dist_23","dist_24",
    "dist_25","dist_26","dist_27","dist_28","dist_29","dist_30","dist_31","dist_32","dist_33","dist_34","dist_35","dist_36",#distance between every pair of cities
    "ee_1","ee_2","ee_3","ee_4","ee_5","ee_6","ee_7","ee_8","ee_9","ee_10","ee_11","ee_12",
    "ee_13","ee_14","ee_15","ee_16","ee_17","ee_18","ee_19","ee_20","ee_21","ee_22","ee_23","ee_24",
    "ee_25","ee_26","ee_27","ee_28","ee_29","ee_30","ee_31","ee_32","ee_33","ee_34","ee_35","ee_36",
    "R", #energy between every pair of cities
]

# mi_results = [
#     Moran(gdf[variable], w) for variable in cluster_variables
# ]
# mi_results = [
#     (variable, res.I, res.p_sim)
#     for variable, res in zip(cluster_variables, mi_results)
# ]

# table = pandas.DataFrame(
#     mi_results, columns=["Variable", "Moran's I", "P-value"]
# ).set_index("Variable")
theta = 2000 #Maximum energy capacity. 

areas_in_path = int(theta / np.average(energy))
max_nodes = areas_in_path 
charge_points = int(np.floor(v / max_nodes) )  
db_scaled = robust_scale(gdf[cluster_variables])
kmeans = KMeansConstrained(n_clusters=charge_points+1,
                           size_min = charge_points, 
                           size_max = max_nodes-2,
                           random_state=0)
numpy.random.seed(1234)
k5cls = kmeans.fit_predict(db_scaled)
gdf['k5cls'] = k5cls

paths = {k: [] for k in agents}
#  f, ax = plt.subplots(1, figsize=(9, 9))

# Plot unique values choropleth including
# a legend and with no boundary lines
# gdf.plot(
#     column="k5cls", categorical=True, legend=True, linewidth=0, ax=ax, aspect=1
# )
# Remove axis
# ax.set_axis_off()
# # Display the map
# plt.show()

groups = gdf.groupby("k5cls")
depots = []
D = [V_nodes[8],V_nodes[9]]
for d in D: 
    depots.append(gdf.loc[gdf['Area_id'] == d])

depots_for_agents = assign_agents_to_areas(m,D)

####Create and use the GA ####

#Parameters
population_size = 200
generations = 100 

depot_values = [depots_for_agents[key] for key in depots_for_agents.keys()]
depot_values = set(depot_values)
# cost_d = distances.to_numpy().astype(int)
# cost_e = energy.to_numpy().astype(int)

TS = [range(1,v,1)] 

start_time = time.time() #Beginning of Solver execution

for group in groups: 
    
    nodes = list(group[1]['Area_id'].to_numpy().astype(int))
    
    cost_d = group[1][dist_columns].to_numpy()
    cost_e = group[1][ee_columns].to_numpy()
    R_points = list(group[1]["R"].to_numpy())

    for k in depots_for_agents.keys(): 
        if depots_for_agents[k] not in nodes:
            nodes.append(depots_for_agents[k])
            cost_d = np.append(cost_d,distances.loc[depots_for_agents[k]].to_numpy().reshape(1,v),axis=0)
            cost_e = np.append(cost_e,energy.loc[depots_for_agents[k]].to_numpy().reshape(1,v),axis=0)
            R_points.append(required_visits[depots_for_agents[k]])
   
    V_nodes = list(range(len(nodes)))
    V = {V_nodes[i]:nodes[i] for i in range(len(nodes))}
   
    ga = GASOL(population_size, generations, V_nodes,agents,depots_for_agents)
    best_paths, hof = ga.run(cthr=0.7, mthr=0.05, cost=cost_d) #Run the heuristic based on distance 
 
    #Create the initial solution set for the solver. 
    initial_population = [] 
    if len(best_paths) < len(agents): #if the GA return less paths than the number of agents, take the best path and fill it to the initial population. 
        initial_population = list(best_paths.keys())
        additional_path = next(iter(best_paths.keys()))
        for _ in range(len(agents)-len(best_paths)):
            initial_population.append(additional_path)
    else: 
        initial_population = list(best_paths.keys())
         

    model = LpProblem("Mvmtsp", LpMinimize) # Optimization Minimizatino problem. 

    # Optimization Parameters. 
    x = LpVariable.dicts("x", ((i,j,k) for i in V_nodes for j in V_nodes for k in agents), cat='Binary') #Variable to show if agent k traveld from area i to j. 
    u = LpVariable.dicts("u", ((i,k) for i in V_nodes for k in agents), lowBound=0, upBound=v-1, cat='Integer') #Variable to use for the subtour elimination constraints. 
    t = LpVariable.dicts("t", ((i,j,k,ts) for i in V_nodes for j in V_nodes for k in agents for ts in TS), cat='Binary') #Variable to handle the execution based on timing. 
    # e = LpVariable.dicts("e", ((i for i in V_nodes)),lowBound=0,upBound=theta, cat='Continuous') #Variable that holds information about the energy consumption between two nodes.  
    e = LpVariable.dicts("e", ((i,k) for i in V_nodes for k in agents),lowBound=0,upBound=theta, cat='Continuous') #Variable that holds information about the energy consumption between two nodes.  
    z = LpVariable.dicts("z_ik", ((i,k) for i in V_nodes for k in agents), lowBound=0,upBound=1,cat='Binary') #Varable to have information about the customer service. 


    #Objective Function
    model.setObjective(lpSum(cost_d[i][j] * x[i,j,k] + cost_e[i][j] * x[i,j,k] for i in V_nodes for j in V_nodes if i!=j for k in agents)) 

    ### ---ALL CONSTRAINTS---###
    #Constraint 1: Each node must be entered and exitted once by each agent 
    for k in agents: 
        d = depots_for_agents[k]
        for j in V_nodes: 
            if j != d:
                model += lpSum(x[i,j,k] for i in V_nodes if i != j) == 1, f"Enter_{j}_by_{k}"
                model += lpSum(x[j,i,k] for i in V_nodes if i != j) == 1, f"Exit_{j}_by_{k}"

    #Constraint 2: Subtour elimination constraints SEC 
    for k in agents: 
        d = depots_for_agents[k]
        for i in V_nodes[1:]: 
            model += x[i, d, k] + x[d,i,k] <=1 

    for k in agents:
        d = depots_for_agents[k]
        for i in V_nodes: 
            for j in V_nodes: 
                if i!=j and i != d and j != d : 
                    model +=  u[i, k] - u[j, k] + v * x[i, j, k] <= v - 1 

    for k in agents: 
        for i in V_nodes: 
            if i != V_nodes: 
                model += u[i,k] >= 1 
                model += u[i,k] <= v - 1

    #Constraint 3: Many visits for each city 
    for j in V_nodes : 
        model += lpSum(x[i,j,k] for i in V_nodes if i!=j for k in agents) <= R_points[j-1] , f'Min_visits_{j}_{k}' #Constraint 2 For Visits

    # Constraint 4: Allow only a single agent to be on top of a city. 
    for step in TS:
        d = depots_for_agents[k]
        model += lpSum(t[i,j,k,step] for k in agents for i in V_nodes for j in V_nodes if i!=j and j !=d) <= 1,f"Single_Agent_at_at_time_{step}"
    
    # Constraint 5: Each agents begins its journey at full energy 
    for k in agents:
        d = depots_for_agents[k] 
        model += e[d,k] >= theta - M * (lpSum(z[d,k])), f"EnergyFull_at_start_{d}_{k}"

    # Constraint 6: Travel corresponds to energy 
    for k in agents: 
            d = depots_for_agents[k]
            for i in V_nodes: 
                for j in V_nodes: 
                    if i!=j: 
                        model += e[j,k] >= e[i,k] - (cost_e[i-1][j-1] * x[i,j,k]),f"Update_remaining_energy_{i}_{j}_{k}"
                        model += e[i,k] >= cost_e[i-1][j-1] * x[i,j,k],f"No_travel_if_low_energy{i}_{j}_for_{k}"
                if i == d: 
                    model += e[j,k] == theta - M * (lpSum(z[d,k])), f"EnergyFull_at_end_{i}_{k}"
                        

    #Constraint 7: Start and finish at the same depot. 
    for k in agents: 
        d = depots_for_agents[k]
        model += lpSum(x[d,j,k] for j in V_nodes if j not in depots_for_agents) ==1, f"Start_at_depot_{k}"
        model += lpSum(x[j,d,k] for j in V_nodes if j not in depots_for_agents) ==1, f"Finish_at_depot_{k}"
        model += x[d,d,k] == 0, f"No_loop_at_depot_{k}_{d}"
        for i in depot_values:
            if i != d: 
                model += x[d,i,k] == 0, f"No_travel_from_{i}_at_depot_{d}_for_{k}"
                model += x[i,d,k] == 0, f"No_travel_to_{i}_at_depot_{d}_for_{k}"
                model += x[d,i,k] + x[i,d,k] <= 0, f"No_travel_{i}_at_{d}_for_{k}"


    # #Constraint 9: Prevent loops and unecessarry visits at depots. 
    for k in agents: 
        for step in TS[:-1]: 
                d = depots_for_agents[k]
                model += t[d,d,k,step] == 0 
                model += t[d,d,k,step] + t[d,d,k,step+1] <= 1, f"No_loop_at_depot_{d}_{step}_{k}"
                model += lpSum(t[d,j,k,step] for j in depots_for_agents ) == 0, f"No_unnecessary_visits_at_depot_{d}_{step}_{k}"

    #Constraint 10: Ensure that a customer/city is served by an agent. 
    # for i in C[0:10] : 
    #     a = i.item()
    #     model += lpSum(x[j,a,k] for j in V_nodes for k in agents) >= z[a,k], f"ServiceConstraint_{a}_{k}"
    

    #Set initial solution based on the GA 
    for a in range(len(initial_population)): 
        for i in range(len(initial_population[a])-1): 
            x[initial_population[a][i],initial_population[a][i+1],a+1].setInitialValue(1)

    try: 
        solve_problem(model)
    except timeout_decorator.TimeoutError:
        print("Time limit exceeded")
        exit(0)


    if pl.LpStatus[model.status] == "Optimal":

        # for k in agents:
        #     d = depots_for_agents[k]
        #     sum_start = sum(x[d, j, k].varValue for j in V_nodes if j != d)
        #     sum_finish = sum(x[j, d, k].varValue for j in V_nodes if j != d)
        #     print(f"Agent {k} starts from depot {d}: {sum_start}")
        #     print(f"Agent {k} returns to depot {d}: {sum_finish}")
        # Retrieve the paths
        
        for k in agents:
            # Start from the depot
            current_node = depots_for_agents[k]
            start_node = current_node
            route = [current_node]
            while True:
                next_node = None
                for j in V_nodes:
                    if j != current_node and x[current_node, j, k].varValue == 1:
                        next_node = j
                        break
                if next_node is None :
                    break
                if next_node == start_node:
                    # route.append(start_node)
                    break
                route.append(V[next_node])
                current_node = next_node
            paths[k].append(route)

        # Print the paths for each agent
        for k in paths:
            print(f"Agent {k} path: {' -> '.join(map(str, paths[k]))}")

        print("Optimal solution found.")
    else:
        print("No optimal solution available.")

# #Compose the paths 
# for k in paths.keys():
#     tmp = [] 
#     for tour in range(len(paths[k])):
#         tmp.extend(paths[k][tour])
#     paths[k] = []
#     paths[k].append(tmp)

# for k in paths:
#     print(f"Agent {k} path: {' -> '.join(map(str, paths[k]))}")
#     tmp = len(set(paths[k][0]))
#     print(tmp == v)
#     if tmp != v: 
#         print(sorted(set(paths[k][0])))
# pdb.set_trace()


# ga = GASOL(population_size, generations, V_nodes,agents,depots) #Create the GA heuristic
# best_paths, hof = ga.run(cthr=0.7, mthr=0.05, cost=distances) #Run the heuristic based on distance 

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


# try: 
#     solve_problem(model)
# except timeout_decorator.TimeoutError:
#     print("Time limit exceeded")
#     exit(0)

# model.solve(pl.GLPK(msg=False, options=['--mipgap','0.05']))  # Set msg=True to see GLPK output also set mipgap to 0.05 or 5% to stop execution as the optimal answer. 
exit_time = time.time() 

execution_time = exit_time - start_time 
print(f" Program Completion Time: {execution_time}s")


# if pl.LpStatus[model.status] == "Optimal":
#     # Retrieve the paths
#     paths = {k: [] for k in agents}
#     for k in agents:
#         # Start from the depot
#         current_node = D[0]
#         start_node = current_node
#         route = [current_node]
#         while True:
#             next_node = None
#             for j in V_nodes:
#                 if j != current_node and x[current_node, j, k].varValue == 1:
#                     next_node = j
#                     break
#             if next_node is None or next_node == start_node:
#                 break
#             route.append(next_node)
#             current_node = next_node
#         paths[k] = route

#     # Print the paths for each agent
#     for k in paths:
#         print(f"Agent {k} path: {' -> '.join(map(str, paths[k]))}")

#     print("Optimal solution found.")
# else:
#     print("No optimal solution available.")
