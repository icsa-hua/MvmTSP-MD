from dummy_app.designs.mvmtsp_config import MVMTSPConfig 
from dummy_app.tools.autonomize import deallocate_memory, extract_context_for_cluster, process_extraction
from dummy_app.tools.performance_metrics import Metrics
from dummy_app.tools.logger import logger 
from typing import Any, List, Dict, Union, Tuple

import math
import time 
import pulp as pl 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from collections import defaultdict




class MVMTSPBuilder(MVMTSPConfig):

    def __init__(self, config:Dict[str,Any]): 
        super().__init__(config)
        
        self.allow_regionalization = config['regionalization']
        self.enable_ga = config['genetic_algorithm']
        self.constraints = config['constraints']
        self.employed_agents:List[int] = [] 
        self.V = pd.DataFrame() 
        self.v:int = 0
        self.population = [] 
        self.best_path = [] 
        self.timeFrame_per_cluster = []
        self.initial_population = [] 
        self.metrics = Metrics(verbose=True) 
        self.clusters_times:Dict[int, int] = {} 
        self.cluster_id:int = 0 
        self.time_window:int = 5 #descrete time steps


    def create_problem(self, V:List[int])->None:

        # Combinatorial Optimization Problem 
        self.problem = pl.LpProblem("ContrainedMVMTSP", pl.LpMinimize)

        # Variable to show agent's travel from i to j. 
        self.x = pl.LpVariable.dicts("x", ((i, j, k) for i in V for j in V for k in self.agents), cat='Binary')
    
        # Variable to show position of agent's in the time frame T 
        self.p = pl.LpVariable.dicts("p", ((k,t) for k in self.agents for t in self.timeFrame_per_cluster), cat='Integer')
        
        # Variable to use for subtour elimination constraints 
        self.u = pl.LpVariable.dicts("u", ((i, k) for i in V for k in self.agents), lowBound=0, upBound=len(V)-1, cat='Integer')

        # Variable to handle the action based on timing 
        self.t = pl.LpVariable.dicts("t", ((i, j, k, ts) for i in V for j in V for k in self.agents for ts in self.timeFrame_per_cluster), cat='Binary')

        # Variable that holds information about the energy consumption between two nodes 
        self.e = pl.LpVariable.dicts("e", ((i, k) for i in V for k in self.agents),lowBound=0, upBound=self.max_battery, cat='Continuous')

        # Variable that handles customer service. 
        self.z = pl.LpVariable.dicts("z_ik", ((i, k) for i in V for k in self.agents), lowBound=0, upBound=1, cat='Binary')
        

    def set_objective(self, distance:Any, energy:Any, time:Any, nodes:Dict):
        
        self.problem.setObjective(
            pl.lpSum(
                distance[nodes[i]][nodes[j]-1] * self.t[i,j,k,t]
                + energy[nodes[i]][nodes[j]-1] * self.t[i,j,k,t]
                + time[nodes[i]][nodes[j]-1] * self.t[i,j,k,t]
                for t in self.timeFrame_per_cluster
                for i in nodes
                for j in nodes
                if i != j 
                for k in self.agents
            )
        )
         
    
    def call_genetic_algorithm(self, V_nodes:List[int], cost:Dict[str,float], depot:int, verbose:bool, population_size:int=200, generations:int=100)->List[int]:
        return super().call_genetic_algorithm(V_nodes, cost, depot, verbose, population_size, generations) 
    

    def assign_agents_to_areas(self, plethos:int=0, depots:Union[List[int], Dict[int,int]]=None)->Dict[int,int]:
        return super().assign_agents_to_areas(plethos, depots)
    

    def assign_depot_to_cluster(self, clusters:object, depots_df:pd.DataFrame):

        def find_duplicate_values(mapping:Dict)->Dict: 
            reverse = {} 
            for key, value in mapping.items():
                reverse.setdefault(value, []).append(key)
            return {val:keys for val, keys in reverse.items() if len(keys) > 1}

        cluster_depot = {} 

        for cluster_id, cluster_df in clusters:

            # Calculate cluster centroid directly
            centroid = cluster_df[['X_coords', 'Y_coords']].mean().values 

            # Find nearest depot 
            min_dist = float('inf')
            best_depot = None 
            for _, depot_row in depots_df.iterrows(): 
                depot_coords = depot_row[['X_coords', 'Y_coords']].values
                dist = np.linalg.norm(centroid - depot_coords) 
                if dist < min_dist:
                    min_dist = dist
                    best_depot = depot_row['Area_id']
    
            cluster_depot[cluster_id] = int(best_depot)
        dulpicates = find_duplicate_values(self.depots_for_agents) 
        return cluster_depot, dulpicates
    

    def separate_depots_from_clusters(self, data: pd.DataFrame): 
        """
        Separates depot rows from cluster data based on depots assigned to agents.

        Args:
            data (pd.DataFrame): Dataset containing Area_id and other attributes.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (cluster_data, depot_data)
        """
        unique_depots= list(set(self.depots_for_agents.values()))

        is_depot = data['Area_id'].isin(unique_depots)
        depot_data = data[is_depot].copy() 
    
        cluster_data = data[~is_depot].copy()
        return cluster_data, depot_data 
    

    def add_depot_data_to_cluster(self, cluster_df:Tuple[int,pd.DataFrame], depot_row:pd.Series, depot_id:int):
        depot_row= depot_row.copy()
        depot_row  = depot_row[depot_row['Area_id']==depot_id]
        depot_row.loc[:, 'cluster'] = np.mean(cluster_df[1]['cluster'])        
        # Append and reset index if needed
        updated_cluster = pd.concat([cluster_df[1], depot_row], ignore_index=True)
        
        return updated_cluster
    

    def allocate_agents_to_clusters(self, cluster_with_depots:Dict[int,int], priority:pd.DataFrame, duplicates:Dict[int,List[int]]): 
        
        def iterate_depots_for_assignment(cluster_df, depots_for_agents, assignments): 
            
            for depot in set(depots_for_agents.values()): 
                depot_agents = duplicates.get(depot, [
                    agent for agent in self.agents if self.depots_for_agents[agent] == depot
                ])
                depot_clusters = cluster_df[cluster_df['depot']==depot]
                if depot_clusters.empty: 
                    logger.debug(f"(Termination) No clusters found for depot {depot}")
                    break 

                top_cluster = depot_clusters.index[0]
                if (top_cluster, depot) not in assignments: 
                    assignments[(top_cluster, depot)] = depot_agents
                cluster_df = cluster_df.drop(index=top_cluster)
                
                # cluster_df.drop(index=top_cluster, inplace=True)

            return cluster_df, assignments 

        # Convert depot assignments to dataframe 
        cluster_df = pd.DataFrame.from_dict(cluster_with_depots, orient='index', columns=['depot'])
        
        # Join with priority dataframe (not sorted priority)
        cluster_df = cluster_df.join(priority) 
        cluster_df.sort_values(by='Rank', ascending=True, inplace=True)
        assignments = {} 

        while not cluster_df.empty:
            cluster_df, assignments = iterate_depots_for_assignment(cluster_df, self.depots_for_agents, assignments)

        return assignments 


    def solve_problem(self):
        return super().solve_problem()
    
    
    def preprocess(self, distances_path, energies, nodes_path, agents, customers_path, ground_users, max_battery):
        data = super().preprocess(distances_path, energies, nodes_path, agents, customers_path, ground_users, max_battery)
        
        self.depots_for_agents = self.assign_agents_to_areas(len(self.agents),self.depots)


        logger.info("Preprocessing completed successfully...")
        return data 


    def set_memory_limit(self, max_memory = 1024):
        return super().set_memory_limit(max_memory)
    
    
    def create_solution(self, V_nodes, nodes_dict):
        logger.info(f"Set Time Frame is {self.timeFrame_per_cluster}") 

        if pl.LpStatus[self.problem.status] != 'Optimal': 
            logger.info("Problem is not optimal, returning None...")
            exit(1)
        employed_agents = ["Agent_" + str(agent_id) for agent_id in self.employed_agents]
        reverse_dict = {v:k for k,v in nodes_dict.items()}
        list_of_agents = {name: int(name.split('_')[-1]) for name in employed_agents}
        maxiter_counter = len(V_nodes) * len(self.timeFrame_per_cluster)

        for agent_name, agent_id in list_of_agents.items() :
            start_node = self.depots_for_agents[agent_id] 
            current_node = reverse_dict[start_node] 

            current_time_step = self.timeFrame_per_cluster[0] -1 

            route = [] 
            iteration = 0 

            while iteration < maxiter_counter:
                next_steps = [
                    (current_node, j, t)
                    for t in self.timeFrame_per_cluster
                    for j in V_nodes 
                    if j != current_node and t > current_time_step
                    and self.x[current_node, j, agent_id].varValue == 1
                    and self.t[current_node, j, agent_id, t].varValue == 1
                ]
                iteration += 1

                if not next_steps: 
                    logger.debug(f"No next step found for Agent_{agent_id}. Ending route.")
                    break 

                if len(next_steps) > 1: 
                    logger.debug(f"Multiple next steps found for Agent_{agent_id}. Choosing the first one.")

                i, j, t = next_steps[0]
                route.append((nodes_dict[i], nodes_dict[j], t))
                current_node = j
                current_time_step = t

            if iteration == maxiter_counter: 
                logger.info(f"Maximum iterations reached for Agent_{agent_id}. Ending route.")
                break 

            self.paths[agent_id].append(route) 

        logger.debug(f"Solutions created for {len(self.employed_agents)} agents")
        self.moment += len(V_nodes) + 1 + self.time_window
        self.cluster_id += 1 
        self.clusters_times[self.cluster_id] = self.moment 

        memory_usage = self.metrics.get_memory_usage()
        logger.info(f"Memory usage: {memory_usage:.2f} MB")
        logger.info("Optimal Solution Found")
    

    def createGeoDataset(self, data):
        return super().createGeoDataset(data)
    

    def run(self):
        return super().run()
    

    def run_model(self, data:pd.DataFrame, cue_groups:Dict[int,List[object]]):
        logger.debug("Running combinatorial problem constructor...")
        with tqdm (total=8, desc="Preparing Problem with clustering") as pbar: 

           # Phase 1: Preprocessing and regionalization (geospatial clustering)
            try: 
                data, depots = self.separate_depots_from_clusters(data)
                pbar.update(1)
                logger.debug("Depots separated from clusters successfully...")
                gdf = self.createGeoDataset(data)
                pbar.update(1)
                logger.debug("GeoDataset created successfully...")
                clusters = self.regionalization(gdf)
                pbar.update(1)
                logger.debug("Clusters created successfully...")
                
            except Exception as e:
                logger.exception(f"Error occurred during regionalization: {e}")
                raise ValueError("Error occurred during regionalization.")
            time.sleep(1)
            # Phase 2: Clustering and Prioritization 
            try: 
                priority = self.cluster_prioritization(clusters, cue_groups)
                pbar.update(1)
                logger.debug("Clusters prioritized successfully...")
            except Exception as e:
                logger.exception(f"Error occurred during clustering: {e}")
                raise ValueError("Error occurred during clustering.")
            time.sleep(1)
            # Phase 3: Agent Assignment for all clusters 
            try: 
                cluster_with_depots, same_depot_agents = self.assign_depot_to_cluster(clusters, depots)
                pbar.update(1)
                logger.debug("Depots assigned to clusters successfully...")
                assignments = self.allocate_agents_to_clusters(cluster_with_depots, priority, same_depot_agents)
                pbar.update(1)
                logger.debug("Total Initial Assignments of all agents to all clusters based on priority")

            except Exception as e:
                logger.exception(f"Error occurred during agent assignment: {e}")
                raise ValueError("Error occurred during agent assignment.") 
            time.sleep(1)
            # Phase 4: Final clusters refinement and memory deallocation 

            try: 
                updated_clusters = [] 
                for contract in assignments.keys(): 
                    for cluster in clusters: 
                        flag = cluster[0] == contract[0]
                        if flag: 
                            updated_clusters.append(self.add_depot_data_to_cluster(cluster, depots, contract[1])) 
                pbar.update(1)
                deallocate_memory(data)
                deallocate_memory(gdf)
                deallocate_memory(clusters)
                deallocate_memory(priority)
                deallocate_memory(cluster_with_depots)
                deallocate_memory(same_depot_agents)
                deallocate_memory(depots)
                logger.debug("Final refinements added to clusters successfully...")

            except Exception as e:
                logger.exception(f"Error occurred during final cluster refinement: {e}")
                raise ValueError("Error occurred during final cluster refinement.")
            time.sleep(1)
            pbar.update(1) 
        
        logger.info("Setting up problem construction completed...")
        logger.info("Starting cluster optimization...")
        # Phase 5: Problem Construction and Solution
        # with tqdm(total=len(clusters), desc="Solving problem...", unit="step") as pbar:
        #     for (cluster_tuple, agents), cluster in zip(assignments.items(), updated_clusters):
        #         contracts = {k:cluster_tuple[1] for k in agents}
        #         self.clustering(cluster, cluster_tuple[0], contracts)
        #         pbar.update(1)
        #         logger.debug(f"Cluster {cluster_tuple[0]} solved successfully...")
        return assignments, updated_clusters


    def regionalization(self, GDF):
        return super().regionalization(GDF)
    
    
    def cluster_prioritization(self, clusters, cue_groups):
        return super().cluster_prioritization(clusters, cue_groups)
    

    def clustering(self, cluster, cluster_id, assignment, depot_id):
        
        logger.debug(f"Clustering with {cluster_id} and agents assigned to it: {assignment}")
        column_names = ["dists", "ees", "travel_times", "area_ids"]
        
        context = extract_context_for_cluster(
            cluster=cluster,
            columns=[
                self.distance_columns,
                self.energy_columns,
                self.travel_time_columns,
                'Area_id'
            ], 
            column_names=column_names 
        )
        
        # Step 1: Determine agents assigned to this cluster 
        # self.employed_agents = [
        #     f"Agent_{agent_id}"
        #     for agent_id, assigned_cluster_id in assignment.items() 
        #     if assigned_cluster_id == cluster_id 
        # ] 
        # NOTE: When entering here it is decided what agents go where so trying to determine which 
        # cluster is the destination is redundant 
        self.employed_agents = assignment

        if self.employed_agents is None: 
            logger.error(f"No agents assigned to cluster {cluster_id}")
            exit(1)
        
        logger.debug(f"Employed agents: {self.employed_agents} for Cluster ID {cluster_id}")


        # Step 2: Process inpute context 
        try: 
            cost, R_points, nodes_dict, self.initial_population = process_extraction(self, context, depot_id)
        except Exception as e: 
            logger.exception(f"Error processing cluster {cluster_id}: {e}")
            return 
        

        # Step 3: Calculate maximum travel times 
        max_time_steps = {} 
        V_nodes = list(nodes_dict.keys())
        for i in V_nodes: 
            max_time = 0 
            for j in V_nodes: 
   
                tmp_time = math.ceil(self.travel_cost[nodes_dict[i]-1][nodes_dict[j]-1])
                if tmp_time > max_time: 
                    max_time = tmp_time 

            max_time_steps[i] = max_time

        total_time = sum(max_time_steps.values()) 
        # NOTE: Try it without the self.moment variable. Every time frame is specific to that cluster NOT the whole simulation. 
        # self.timeFrame_per_cluster = list(range(self.moment, self.moment + total_time + 1))
        
        if total_time == 0: 
            logger.error(f"Total time is 0 for cluster {cluster_id}")
            raise ValueError(f"Total time is 0 for cluster {cluster_id}")
        
        self.timeFrame_per_cluster = list(range(0, total_time + 1))


        # Step 4: Create and configure the optimization problem 
        try: 
            self.create_problem(V_nodes)
            self.set_objective(
                distance=cost['distance'], 
                energy=cost['energy'], 
                time=cost['travel_time'],
                nodes=nodes_dict
            )

        except Exception as e:
            logger.exception(f"Error creating problem for cluster {cluster_id}: {e}")
            raise ValueError(f"Error in creating the problem for Cluster {cluster_id}")


        if len(self.employed_agents) > 1: 
            # Many Visits Multi TSP solution 
            try: 
                self.set_constraints_for_multi_agent(
                    V_nodes=V_nodes, 
                    nodes_dict=nodes_dict,
                    cost=cost, 
                    R_points=R_points,
                )
                logger.info("Many-visits multi-agent TSP problem created successfully...")
            except Exception as e:
                logger.exception(f"Error setting constraints for cluster {cluster_id}: {e}")
                raise ValueError(f"Error in setting constraints for Cluster {cluster_id}")
            try: 
                logger.info(f"Solving problem for cluster {cluster_id}...")  
                self.run()
                self.create_solution(V_nodes, nodes_dict)
            except Exception as e:
                logger.exception(f"Error solving problem for cluster {cluster_id}: {e}")
                raise ValueError(f"Error in solving the problem for Cluster {cluster_id}")
            
        elif len(self.employed_agents) == 1: 
            # Single Visit Many-visits TSP solution (TODO: Implement) 
            logger.info(f"Single Visit Many-visits TSP solution for Cluster {cluster_id}")
        
        else: 
            logger.error(f"No agents assigned for Cluster {cluster_id}")
            raise ValueError(f"No employed agents for cluster {cluster_id}")


    def set_constraints_for_multi_agent(self, V_nodes:List[int], nodes_dict:Dict[int, int], cost:Dict[str,List[float]], R_points:List[int]): 

        logger.info(f"Setting constraints for multi-agent problem...")
        employed_agents = ["Agent_" + str(agent_id) for agent_id in self.employed_agents]
        list_of_agents = {x:int(x.split('_')[-1]) for x in employed_agents}
        header = list_of_agents[employed_agents[0]]
        reverse_nodes = {v: k for k, v in nodes_dict.items()}
        depot_ind = self.get_depot_index(nodes_dict, header)
        valid_arcs = [(i,j) for i in V_nodes for j in V_nodes if i != j and i != depot_ind and j != depot_ind]

        in_arcs = defaultdict(list)
        out_arcs = defaultdict(list)

        for i, j in valid_arcs:
            out_arcs[i].append(j)
            in_arcs[j].append(i)

        tr_times = {(i,j):self.get_travel_time(i,j,nodes_dict) for i in V_nodes for j in V_nodes}
        

        deallocate_memory(valid_arcs)
 
        # for j in V_nodes:
            # if j != depot_ind:
            #     self.model += lpSum(self.x[i, j, v] for i in in_arcs[j]) == 1
            #     self.model += lpSum(self.x[j, i, v] for i in out_arcs[j]) == 1

        if self.enable_ga: 
            for a in self.employed_agents: 
                for i in range(len(self.initial_population[a][0])-1): 
                    node = reverse_nodes[self.initial_population[a][0][i]]
                    next_node = reverse_nodes[self.initial_population[a][0][i+1]] 
                    self.x[node, next_node, header].setInitialValue(1) 

        

        if "const_0" in self.constraints: 
            for j in V_nodes: 
                self.problem += pl.lpSum(self.x[i,j,v] for k,v in list_of_agents.items() for i in V_nodes if i != j and i!=depot_ind and j!=depot_ind) <= R_points[nodes_dict[j]], f"All_nodes_visited_by_agent_{j}"

            logger.debug(f"Constraint | const_0 - All nodes visited multiple times in total | set for cluster ")

        if "const_1" in self.constraints:
            try: 
                for k, v in list_of_agents.items():
                    self.problem += pl.lpSum(self.x[depot_ind, j, v] for j in V_nodes if j != depot_ind) == 1, f"{k}_enters_single_area_from_depot_{depot_ind}"
                    self.problem += pl.lpSum(self.x[i, depot_ind, v] for i in V_nodes if i != depot_ind) == 1, f"{k}_leaves_single_area_to_depot_{depot_ind}"
            
                logger.debug(f"Constraint | const_1 - Each agent enters and leaves the depot once | set for cluster ")
            except Exception as e:
                logger.exception(f"Error setting constraint const_1 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_1 for Cluster") 
            
        if "const_2" in self.constraints:
            try:
                for k, v in list_of_agents.items():
                    self.problem += self.p[v,self.timeFrame_per_cluster[0]] == depot_ind, f"Positional_variable_at_start_of_journey_for_{k}" 
                    self.problem += self.p[v,self.timeFrame_per_cluster[-1]] == depot_ind, f"Positional_variable_at_end_of_joureny_for_{k}"

                logger.debug(f"Constraint | const_2 - Positional variable at start and end of journey (depot)| set for cluster ")
            except Exception as e: 
                logger.exception(f"Error setting constraint const_2 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_2 for Cluster")
            
        if "const_3" in self.constraints: 
            try: 
                for k, v in list_of_agents.items(): 
                    self.problem += pl.lpSum(self.t[depot_ind, j, v, self.timeFrame_per_cluster[0]] for j in V_nodes if depot_ind != j ) ==1, f"{k}_leaves_depot_{depot_ind}_at_specific_interval"
                    self.problem += pl.lpSum(self.t[i, depot_ind, v, self.timeFrame_per_cluster[-1]] for i in V_nodes if depot_ind != i ) ==1, f"{k}_enters_depot_{depot_ind}_at_specific_interval"

                logger.debug(f"Constraint | const_3 - Each agent leaves and enters the depot at a specific interval | set for cluster ")
            except Exception as e:
                logger.exception(f"Error setting constraint const_3 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_3 for Cluster")
            
        if "const_4" in self.constraints: 
            try:
                for k, v in list_of_agents.items():
                    self.problem += pl.lpSum(self.x[depot_ind, j, v] for j in V_nodes if j != depot_ind) + \
                                    pl.lpSum(self.x[i, depot_ind, v] for i in V_nodes if i != depot_ind) == 2, f"{k}_start_&_finishes_at_depot_{depot_ind}"  
                logger.debug(f"Constraint | const_4 - Each agent starts and ends at the depot | set for cluster ") 
            except Exception as e:
                logger.exception(f"Error setting constraint const_4 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_4 for Cluster")
        
        if "const_5" in self.constraints:
            try: 
                for k, v in list_of_agents.items():
                    self.problem += self.x[depot_ind, depot_ind, v] == 0,  f"No_loop_at depot_{depot_ind}_for_{k}_at_any_timepoint"

                logger.debug(f"Constraint | const_5 - No loop at depot | set for cluster ")
            except Exception as e:
                logger.exception(f"Error setting constraint const_5 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_5 for Cluster")
            
        if "const_6" in self.constraints: 
            try: 
                for k, v in list_of_agents.items(): 
                    for j in V_nodes: 
                        if j != depot_ind: 
                            self.problem += pl.lpSum(self.x[i, j, v] for i in in_arcs[j]) == 1, f"Enter_{j}_for_{k}_exluding_depots"
                            self.problem += pl.lpSum(self.x[j, i, v] for i in out_arcs[j]) == 1, f"Leave_{j}_for_{k}_exluding_depots"

                logger.debug(f"Constraint | const_6 - Each agent enters and leaves each node once | set for cluster ") 
            except Exception as e:
                logger.exception(f"Error setting constraint const_6 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_6 for Cluster")
            
        if "const_7" in self.constraints: 
            try: 
                for k, v in list_of_agents.items():
                    self.problem += self.u[depot_ind, v] == 1, f"u_at_depot_{depot_ind}_for_{k}"
                logger.debug(f"Constraint | const_7 - u at depot | set for cluster ")
            except Exception as e:
                logger.exception(f"Error setting constraint const_7 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_7 for Cluster")
            
        if "const_8" in self.constraints: 
            try: 
                for k, v in list_of_agents.items():
                    for i in V_nodes: 
                        if i != depot_ind:
                            self.problem += self.u[i,v] >= 2, f"Lower_bound_on_{i}_for_{k}"
                            self.problem += self.u[i,v] <= len(V_nodes)-1, f"Upper_bound_on_{i}_for_{k}"

                logger.debug(f"Constraint | const_8 - Lower and upper bounds on u | set for cluster ")
            except Exception as e:
                logger.exception(f"Error setting constraint const_8 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_8 for Cluster")
            
        if "const_9" in self.constraints: 
            try: 
                for k, v in list_of_agents.items():
                    for j in V_nodes: 
                        if j != depot_ind: 
                            for step in self.timeFrame_per_cluster[:tr_times[(depot_ind, j)]]: 
                                self.problem += (
                                    self.t[depot_ind, j, v, step] <= self.t[depot_ind, j, v, self.timeFrame_per_cluster[0]], 
                                    f"Dynamic_time_enforcement_{depot_ind}_{j}_for_{k}_time_{step}"
                                )
                            
                            for step in self.timeFrame_per_cluster[:-tr_times[(j,depot_ind)]]:
                                self.problem += (
                                    self.t[j, depot_ind, v, step] <= self.t[j, depot_ind, v, self.timeFrame_per_cluster[-1]], 
                                    f"Dynamic_time_enforcement_{j}_to_{depot_ind}_for_{k}_time_{step}"
                                )
                logger.debug(f"Constraint | const_9 - Dynamic time enforcement | set for cluster ")

            except Exception as e:
                logger.exception(f"Error setting constraint const_9 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_9 for Cluster")
            
        if "const_10" in self.constraints: #NOTE: This is essential for the model to work 
            try: 
                for k, v in list_of_agents.items(): 
                    for j in V_nodes: 
                        if j != depot_ind: 
                            for step in self.timeFrame_per_cluster[:tr_times[(depot_ind, j)]]: 
                                self.problem += (
                                    self.t[depot_ind, j, v, step] == self.x[depot_ind, j, v], 
                                    f"Enforce synchronization_between_t_and_x_{depot_ind}_{j}_for_{k}_time_{step}"
                                )
                    for i in V_nodes: 
                        if i != depot_ind: 
                            for step in self.timeFrame_per_cluster[-tr_times[(i, depot_ind)]:]: 
                                self.problem += (
                                    self.t[i, depot_ind, v, step] == self.x[i, depot_ind, v],
                                    f"Enforce synchronization_between_t_and_x_{i}_to_{depot_ind}_for_{k}_time_{step}"
                                )
                logger.debug(f"Constraint | const_10 - Enforce synchronization between t and x | set for cluster ")

            except Exception as e:
                logger.exception(f"Error setting constraint const_10 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_10 for Cluster")
            
        if "const_11" in self.constraints: 
            try: 
                for k, v in list_of_agents.items():
                    for i in out_arcs:
                        for j in out_arcs[i]:
                            time_limit = len(self.timeFrame_per_cluster) - (tr_times[(i, j)] + tr_times[(j, depot_ind)] + 1)
                            for step_idx in range(time_limit):
                                t_step = self.timeFrame_per_cluster[step_idx]
                                t_step_j = self.timeFrame_per_cluster[step_idx + tr_times[(i, j)]]
                                t_step_jj = self.timeFrame_per_cluster[step_idx + 1 + tr_times[(i, j)]]

                                self.problem += self.t[i, j, v, t_step_j] <= self.t[j, j, v, t_step_jj], \
                                            f"Time_Progression_{t_step}_{i}_{j}_for_{k}"
                logger.debug(f"Constraint | const_11 - Time progression | set for cluster ")

            except Exception as e:
                logger.exception(f"Error setting constraint const_11 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_11 for Cluster")
            
        if "const_12" in self.constraints: # NOTE: Also necessary for the model to work. 
            try:
                for k, v in list_of_agents.items():
                    for i in out_arcs: 
                        for j in out_arcs[i]:
                            for step in self.timeFrame_per_cluster[(tr_times[(depot_ind,i)]):-(tr_times[(i,j)]+tr_times[(j,depot_ind)])]:
                                self.problem += (self.t[i, j, v, t] for t in range(step, step+tr_times[(i,j)])) == self.x[i,j,v], f"Link_x_and_t_{i}_{j}_for_{k}_at_time_{step}"

                logger.debug(f"Constraint | const_12 - Link x and t | set for cluster ")

            except Exception as e:
                logger.exception(f"Error setting constraint const_12 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_12 for Cluster")

        if "const_13" in self.constraints: # NOTE: This only seems to have an effect on the first agent it encounters. 
            try:
                for k, v in list_of_agents.items():
                    for i in out_arcs:
                        for j in out_arcs[i]:
                            self.problem += self.e[j,v] >= self.e[i,v] - cost['energy'][nodes_dict[i]][j] * self.x[i,j,v], f"Update_remaining_energy_{i}_{j}_for_{k}"
                            self.problem += self.e[i,v] >= cost['energy'][nodes_dict[i]][j] * self.x[i, j, v],f"No_travel_if_low_energy_{i}_{j}_for_{k}"
                        self.problem += self.e[i,v] >= cost['energy'][nodes_dict[i]][depot_ind] * self.x[i, depot_ind, v],f"Enough_energy_to_return_to_depot_from_{i}_for_{k}"
                logger.debug(f"Constraint | const_13 - Update remaining energy | set for cluster ")

            except Exception as e:
                logger.exception(f"Error setting constraint const_13 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_13 for Cluster")


        if "const_14" in self.constraints: 
            try: 
                for k, v in list_of_agents.items():
                    for i in out_arcs:
                        for t in self.timeFrame_per_cluster: 
                            self.problem += self.e[i,v] >= 0, f"Energy_cannot_be_negative_{i}_for_{k}_at_time_{t}"

                    self.problem += self.e[depot_ind,v] == self.max_battery, f"Energy_cannot_be_negative_at_depot_for_{k}"
                logger.debug(f"Constraint | const_14 - Energy cannot be negative | set for cluster ")
            except Exception as e:
                logger.exception(f"Error setting constraint const_14 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_14 for Cluster")

        if "const_15" in self.constraints:
            try:
                for i in out_arcs:
                    for j in out_arcs[i]:
                        if i == depot_ind or j == depot_ind:
                            continue
                        # NOTE: Slicing the time window even more here may be brittle: since we are already enforcing positional and journey to depots in specific timesteps. 
                        # for step in self.timeFrame_per_cluster[(tr_times[(depot_ind, i)]): -(tr_times[(i, j)] + tr_times[(j, depot_ind)])]:
                        for step in self.timeFrame_per_cluster:
                            self.problem += pl.lpSum(self.t[i,j,v,step] for _,v in list_of_agents.items()) <= 1, f"Unique_Time_visits_constraint_at_travel_{i}_{j}_at_time_{step}" 

                logger.debug(f"Constraint | const_15 - Unique Time visits constraint | set for cluster ")
            except Exception as e:
                logger.exception(f"Error setting constraint const_15 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_15 for cluster: {e}")
        

        if "const_16" in self.constraints: 
            try: 
                for k1, v1 in list_of_agents.items() : 
                    for k2, v2 in list_of_agents.items() : 
                        if k1 != k2 : 
                            for step in self.timeFrame_per_cluster: 
                                self.problem += pl.lpSum(self.t[i, j, v1, step] - self.t[i,j, v2, step] for i in out_arcs for j in out_arcs[i]) != 0, f"Agent_unique_paths_for_{k1}_and_{k2}_at_time_{step}"
                logger.debug(f"Constraint | const_16 - Agent unique paths | set for cluster ")

            except Exception as e: 
                logger.exception(f"Error setting constraint const_16 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_16 for cluster: {e}")
            
        
    def get_solution(self) -> List[int]: 
        if len(self.paths) != len(self.agents): 
            logger.error(f"Insufficient paths: expected {len(self.agents)}, got {len(self.paths)}") 
            raise RuntimeError(f"Insufficient paths: expected {len(self.agents)}, got {len(self.paths)}")                     
                            
        for k in self.paths: 
            # flatten paths if composed of subtours 
            full_path = [] 
            for tour in self.paths[k]: 
                full_path.extend(tour)
            self.paths[k] = [full_path]

        for k, path_list in self.paths.items(): 
            readable = ' -> '.join(f"{node}@T{time}" for node, time in path_list[0])
            logger.debug(f"Path for agent {k}: {readable}") 

            visited_nodes = {node for node, _ in path_list[0]}
            if len(visited_nodes) != self.v: 
                logger.error(f"Agent {k} visited only {len(visited_nodes)} nodes out of {self.v}")
                raise RuntimeError(f"Agent {k} visited only {len(visited_nodes)} nodes out of {self.v}")

        self.validate_paths() 
        return self.paths


    def get_depot_index(self, ordered_nodes, k): 
        axx = [i for i, value in enumerate(ordered_nodes.values()) if value == self.depots_for_agents[k]]
        return axx[0]


    def validate_paths(self):

        max_time_steps = self.timeFrame_per_cluster[-1]

        for agent_id, path_list in self.paths.items():
            path = path_list[0]  # Assuming single tour per agent
            for i, (node, time) in enumerate(path[:-1]):
                next_node, next_time = path[i + 1]
                if next_time <= time:
                    raise ValueError(f"Agent {agent_id} time regression: {node}@T{time} → {next_node}@T{next_time}")
                if next_time > max_time_steps:
                    raise ValueError(f"Agent {agent_id} exceeds max time step at node {next_node} (T={next_time})")

            logger.info(f"Agent {agent_id} path validated successfully.")

    def get_travel_time(self, i, j, nodes_dict): 
        return math.ceil(self.travel_cost[nodes_dict[i], nodes_dict[j]])