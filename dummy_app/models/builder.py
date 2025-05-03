from dummy_app.designs.mvmtsp_config import MVMTSPConfig 
from dummy_app.tools.autonomize import deallocate_memory, extract_context_for_cluster, process_extraction
from dummy_app.tools.performance_metrics import Metrics
from dummy_app.tools.logger import logger 
from typing import Any, List, Dict, Union, Tuple

import math
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
        self.timeFrame = []
        self.initial_population = [] 
        self.metrics = Metrics(verbose=True) 
        self.clusters_times:Dict[int, int] = {} 
        self.cluster_id:int = 0 


    def create_problem(self, V:List[int])->None:

        # Combinatorial Optimization Problem 
        self.problem = pl.LpProblem("ContrainedMVMTSP", pl.LpMinimize)

        # Variable to show agent's travel from i to j. 
        self.x = pl.LpVariable.dicts("x", ((i, j, k) for i in V for j in V for k in self.agents), cat='Binary')
    
        # Variable to show position of agent's in the time frame T 
        self.p = pl.LpVariable.dicts("p", ((k,t) for k in self.agents for t in self.timeFrame), cat='Integer')
        
        # Variable to use for subtour elimination constraints 
        self.u = pl.LpVariable.dicts("u", ((i, k) for i in V for k in self.agents), lowBound=0, upBound=len(V)-1, cat='Integer')

        # Variable to handle the action based on timing 
        self.t = pl.LpVariable.dicts("t", ((i, j, k, ts) for i in V for j in V for k in self.agents for ts in self.timeFrame), cat='Binary')

        # Variable that holds information about the energy consumption between two nodes 
        self.e = pl.LpVariable.dicts("e", ((i, k) for i in V for k in self.agents),lowBound=0, upBound=self.max_battery, cat='Continuous')

        # Variable that handles customer service. 
        self.z = pl.LpVariable.dicts("z_ik", ((i, k) for i in V for k in self.agents), lowBound=0, upBound=1, cat='Binary')
        

    def set_objective(self, alpha:Any, beta:Any, gamma:Any, weight:object):
        super().set_objective(alpha, beta, gamma, weight) 

    
    def call_genetic_algorithm(self, V_nodes:List[int], cost:Dict[str,float], depot:int, population_size:int=200, generations:int=100)->List[int]:
        return super().call_genetic_algorithm(V_nodes, cost, depot, population_size, generations) 
    

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
        
        # Convert depot assignments to dataframe 
        cluster_df = pd.DataFrame.from_dict(cluster_with_depots, orient='index', columns=['depot'])
        
        # Join with priority dataframe 
        cluster_df = cluster_df.join(priority)
        cluster_df.sort_values(by='Rank', ascending=True, inplace=True)
        assignments = {} 
        for depot in set(self.depots_for_agents.values()): 
            # Get all agents using this depot 
            depot_agents = duplicates.get(depot, [
                agent for agent in self.agents if self.depots_for_agents[agent] == depot
            ])
            # Find clusters served by this depot 
            depot_clusters = cluster_df[cluster_df['depot']==depot]
            if depot_clusters.empty: 
                logger.error(f"No clusters found for depot {depot}")
                continue 

            top_cluster = depot_clusters.index[0]
            
            for agent in depot_agents: 
                assignments[agent] = int(cluster_df.index[cluster_df['Rank']==top_cluster].item())

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
        logger.info(f"Set Time Frame is {self.timeFrame}") 

        if pl.LpStatus[self.problem.status] != 'Optimal': 
            logger.info("Problem is not optimal, returning None...")
            exit(1)

        reverse_dict = {v:k for k,v in nodes_dict.items()}
        list_of_agents = {name: int(name.split('_')[-1]) for name in self.employed_agents}
        maxiter_counter = len(V_nodes) * len(self.timeFrame)

        for agent_name, agent_id in list_of_agents.items() :
            start_node = self.depots_for_agents[agent_id] 
            current_node = reverse_dict[start_node] 

            current_time_step = self.timeFrame[0] -1 

            route = [] 
            iteration = 0 

            while iteration < maxiter_counter:
                next_steps = [
                    (current_node, j, t)
                    for t in self.timeFrame
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
    

    def run_model(self, data:pd.DataFrame)->None:
        logger.info("Running combinatorial problem constructor...")

        # Phase 1: Preprocessing and regionalization (geospatial clustering)
        try: 
            data, depots = self.separate_depots_from_clusters(data)
            logger.debug("Depots separated from clusters successfully...")
            gdf = self.createGeoDataset(data)
            logger.debug("GeoDataset created successfully...")
            clusters = self.regionalization(gdf)
            logger.debug("Clusters created successfully...")
            
        except Exception as e:
            logger.exception(f"Error occurred during regionalization: {e}")
            raise ValueError("Error occurred during regionalization.")

        # Phase 2: Clustering and Prioritization 
        try: 
            priority = self.cluster_prioritization(clusters)
            logger.debug("Clusters prioritized successfully...")
        except Exception as e:
            logger.exception(f"Error occurred during clustering: {e}")
            raise ValueError("Error occurred during clustering.")
        
        # Phase 3: Agent Assignment and Problem Construction
        try: 
            cluster_with_depots, same_depot_agents = self.assign_depot_to_cluster(clusters, depots)
            assignments = self.allocate_agents_to_clusters(cluster_with_depots, priority, same_depot_agents)
            clusters_tr = {} 
            for idx, cluster in enumerate(clusters): 
                refined_cluster = self.add_depot_data_to_cluster(cluster, depots, cluster_with_depots[priority.index[idx]])
                clusters_tr[int(priority.index[idx])] = [refined_cluster, priority.iloc[idx]['Rank'], cluster_with_depots[priority.index[idx]]]
                # TODO: Change this to get the actual cluster id as given through clusterign. 
            priority = priority.sort_values(by='Rank', ascending=True)

            logger.debug("Depots assigned to clusters successfully...")

        except Exception as e:
            logger.exception(f"Error occurred during agent assignment: {e}")
            raise ValueError("Error occurred during agent assignment.") 
        
        

        # Phase 4: Problem Construction and Solution
        try: 
            total_steps = len(clusters) * 2 
            deallocate_memory(data)
            deallocate_memory(gdf)

            with tqdm(total=total_steps, desc="Solving problem...", unit="step") as pbar:
                for cluster_id, cluster in clusters_tr.items():
                    
                    self.clustering(cluster, cluster_id, assignments)
                    pbar.update(1)
            
        except Exception as E: 
            logger.exception(f"Error occurred during problem construction: {E}")
            raise ValueError("Error occurred during problem construction.")


    def regionalization(self, GDF):
        return super().regionalization(GDF)
    
    
    def cluster_prioritization(self, clusters, cue_groups):
        return super().cluster_prioritization(clusters, cue_groups)
    

    def clustering(self, cluster, cluster_id, assignment):
        
        logger.debug(f"Clustering with {cluster_id} and assignment {assignment}")
        column_names = ["dists", "ees", "travel_times", "area_ids"]
        context = extract_context_for_cluster(
            cluster=cluster[0],
            columns=[self.distance_columns, self.energy_columns, self.travel_time_columns, 'Area_id'], 
            column_names=column_names 
        )

        # Step 1: Determine agents assigned to this cluster 
        self.employed_agents = [
            f"Agent_{agent_id}"
            for agent_id, assigned_cluster_id in assignment.items() 
            if assigned_cluster_id == cluster_id 
        ] 

        if self.employed_agents is None: 
            logger.error(f"No agents assigned to cluster {cluster_id}")
            exit(1)
        
        logger.debug(f"Employed agents: {self.employed_agents} for Cluster ID {cluster_id}")

        # Step 2: Process inpute context 
        try: 
            cost_d, cost_e, cost_t, R_points, V_nodes, nodes_dict, self.initial_population = process_extraction(self, context, cluster[-1])
        except Exception as e: 
            logger.exception(f"Error processing cluster {cluster_id}: {e}")
            return 
        

        # Step 3: Calculate maximum travel times 
        max_time_steps = {} 
        print(self.travel_cost)
        print(nodes_dict)

        for i in V_nodes: 
            max_time = 0 
            for j in V_nodes: 
   
                tmp_time = math.ceil(self.travel_cost[nodes_dict[i]][nodes_dict[j]])
                if tmp_time > max_time: 
                    max_tim = tmp_time 

            max_time_steps[i] = max_time

        total_time = sum(max_time_steps.values()) 
        self.timeFrame = list(range(self.moment, self.moment + total_time + 1))

        # Step 4: Create and configure the optimization problem 
        try: 
            self.create_problem(V_nodes)
            self.set_objective(
                cost_d, 
                cost_e, 
                cost_t,
                V_nodes, 
                nodes_dict
            )

        except Exception as e:
            logger.exception(f"Error creating problem for cluster {cluster_id}: {e}")
            raise ValueError(f"Error in creating the problem for Cluster {cluster_id}")


        if len(self.employed_agents) > 1: 
            # Many Visits Multi TSP solution 
            try: 
                self.set_constraints_for_multi_agent(
                    V_nodes, 
                    nodes_dict,
                    cost_d, 
                    cost_e,
                    cost_t,
                    R_points,
                )

            except Exception as e:
                logger.exception(f"Error setting constraints for cluster {cluster_id}: {e}")
                raise ValueError(f"Error in setting constraints for Cluster {cluster_id}")
            
            try: 
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


    def set_constraints_for_multi_agent(self, V_nodes, nodes_dict, cost_d, cost_e, cost_t, R_points): 

        logger.info(f"Setting constraints for multi-agent problem...")

        list_of_agents = {x:int(x.split('_')[-1]) for x in self.employed_agents}
        header = list_of_agents[self.employed_agents[0]]
        reverse_nodes = {v: k for k, v in nodes_dict.items()}
        depot_ind = self.get_depot_index(nodes_dict, header)
        tr_times = {(i,j) for i in V_nodes for j in V_nodes if i != j and i != depot_ind and j != depot_ind}
        valid_arcs = [(i,j) for i in V_nodes for j in V_nodes if i != j and i != depot_ind and j != depot_ind]

        in_arcs = defaultdict(list)
        out_arcs = defaultdict(list)

        for i, j in valid_arcs:
            out_arcs[i].append(j)
            in_arcs[j].append(i)

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
                    self.problem += self.p[v,self.timeFrame[0]] == depot_ind, f"Positional_variable_at_start_of_journey_for_{k}" 
                    self.problem += self.p[v,self.timeFrame[0]] == depot_ind, f"Positional_variable_at_end_of_joureny_for_{k}"

                logger.debug(f"Constraint | const_2 - Positional variable at start and end of journey (depot)| set for cluster ")
            except Exception as e: 
                logger.exception(f"Error setting constraint const_2 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_2 for Cluster")
            
        if "const_3" in self.constraints: 
            try: 
                for k, v in list_of_agents.items(): 
                    self.problem += pl.lpSum(self.t[depot_ind, j, v, self.timeFrame[0]] for j in V_nodes if depot_ind != j ) ==1, f"{k}_leaves_depot_{depot_ind}_at_specific_interval"
                    self.problem += pl.lpSum(self.t[i, depot_ind, v, self.timeFrame[-1]] for i in V_nodes if depot_ind != i ) ==1, f"{k}_enters_depot_{depot_ind}_at_specific_interval"

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
                            for step in self.timeFrame[:tr_times[(depot_ind, j)]]: 
                                self.problem += (
                                    self.t[depot_ind, j, v, step] <= self.t[depot_ind, j, v, self.timeFrame[0]], 
                                    f"Dynamic_time_enforcement_{depot_ind}_{j}_for_{k}_time_{step}"
                                )
                            
                            for step in self.timeFrame[:-tr_times[(j,depot_ind)]]:
                                self.problem += (
                                    self.t[j, depot_ind, v, step] <= self.t[j, depot_ind, v, self.timeFrame[-1]], 
                                    f"Dynamic_time_enforcement_{j}_to_{depot_ind}_for_{k}_time_{step}"
                                )
                logger.debug(f"Constraint | const_9 - Dynamic time enforcement | set for cluster ")

            except Exception as e:
                logger.exception(f"Error setting constraint const_9 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_9 for Cluster")
            
        if "const_10" in self.constraints: 
            try: 
                for k, v in list_of_agents.items(): 
                    for j in V_nodes: 
                        if j != depot_ind: 
                            for step in self.timeFrame[:tr_times[(depot_ind, j)]]: 
                                self.problem += (
                                    self.t[depot_ind, j, v, step] == self.x[depot_ind, j, v], 
                                    f"Enforce synchronization_between_t_and_x_{depot_ind}_{j}_for_{k}_time_{step}"
                                )
                    for i in V_nodes: 
                        if i != depot_ind: 
                            for step in self.timeFrame[-tr_times[(i, depot_ind)]:]: 
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
                            time_limit = len(self.timeFrame) - (tr_times[(i, j)] + tr_times[(j, depot_ind)] + 1)
                            for step_idx in range(time_limit):
                                t_step = self.timeFrame[step_idx]
                                t_step_j = self.timeFrame[step_idx + tr_times[(i, j)]]
                                t_step_jj = self.timeFrame[step_idx + 1 + tr_times[(i, j)]]

                                self.problem += self.t[i, j, v, t_step_j] <= self.t[j, j, v, t_step_jj], \
                                            f"Time_Progression_{t_step}_{i}_{j}_for_{k}"
                logger.debug(f"Constraint | const_11 - Time progression | set for cluster ")

            except Exception as e:
                logger.exception(f"Error setting constraint const_11 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_11 for Cluster")
            
        if "const_12" in self.constraints:
            try:
                for k, v in list_of_agents.items():
                    for i in out_arcs: 
                        for j in out_arcs[i]:
                            for step in self.timeFrame[(tr_times[(depot_ind,i)]):-(tr_times[(i,j)]+tr_times[(j,depot_ind)])]:
                                self.problem += (self.t[i, j, v, t] for t in range(step, step+tr_times[(i,j)])) == self.x[i,j,v], f"Link_x_and_t_{i}_{j}_for_{k}_at_time_{step}"

                logger.debug(f"Constraint | const_12 - Link x and t | set for cluster ")

            except Exception as e:
                logger.exception(f"Error setting constraint const_12 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_12 for Cluster")

        if "const_13" in self.constraints:
            try:
                for k, v in list_of_agents.items():
                    for i in out_arcs:
                        for j in out_arcs[i]:
                            self.problem += self.e[j,v] >= self.e[i,v] - cost_e[nodes_dict[i]][j] * self.x[i,j,v], f"Update_remaining_energy_{i}_{j}_for_{k}"
                            self.problem += self.e[i,v] >= cost_e[nodes_dict[i]][j] * self.x[i, j, v],f"No_travel_if_low_energy_{i}_{j}_for_{k}"
                    self.problem += self.e[i,v] >= cost_e[nodes_dict[i]][depot_ind] * self.x[i, depot_ind, v],f"Enough_energy_to_return_to_depot_from_{i}_for_{k}"
                logger.debug(f"Constraint | const_13 - Update remaining energy | set for cluster ")

            except Exception as e:
                logger.exception(f"Error setting constraint const_13 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_13 for Cluster")


        if "const_14" in self.constraints: 
            try: 
                for k, v in list_of_agents.items():
                    for i in out_arcs:
                        for t in self.timeFrame: 
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
                        for step in self.timeFrame[(tr_times[(depot_ind, i)]): -(tr_times[(i, j)] + tr_times[(j, depot_ind)])]:
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
                            for step in self.timeFrame: 
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
        max_time_steps = self.timeFrame[-1]

        for agent_id, path_list in self.paths.items():
            path = path_list[0]  # Assuming single tour per agent
            for i, (node, time) in enumerate(path[:-1]):
                next_node, next_time = path[i + 1]
                if next_time <= time:
                    raise ValueError(f"Agent {agent_id} time regression: {node}@T{time} → {next_node}@T{next_time}")
                if next_time > max_time_steps:
                    raise ValueError(f"Agent {agent_id} exceeds max time step at node {next_node} (T={next_time})")

            logger.info(f"Agent {agent_id} path validated successfully.")

