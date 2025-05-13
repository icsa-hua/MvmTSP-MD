from dummy_app.designs.mvmtsp_config import MVMTSPConfig 
from dummy_app.tools.autonomize import deallocate_memory, extract_context_for_cluster, process_extraction, create_model_graph, get_weights
from dummy_app.tools.performance_metrics import Metrics
from dummy_app.tools.logger import logger 
from typing import Any, List, Dict, Union, Tuple

import sys
import math
import time 
import pulp as pl 
import numpy as np 
import pandas as pd
import networkx as nx 
from tqdm import tqdm 
from collections import defaultdict




class MVMTSPBuilder(MVMTSPConfig):

    def __init__(self, config:Dict[str,Any]): 
        super().__init__(config)
        
        self.allow_regionalization:bool = config['regionalization']
        self.enable_ga:bool = config['genetic_algorithm']
        self.constraints:List[str] = config['constraints']
        self.employed_agents:List[int] = [] 
        self.V = pd.DataFrame() 
        self.v:int = 0
        self.best_path:List[int] = [] 
        self.timeFrame_per_cluster:List[int] = []
        self.initial_population:Dict[int, Tuple[List[int], float]] = {} 
        self.metrics:object = Metrics(verbose=True) 
        self.clusters_times:Dict[int, int] = {} 
        self.cluster_id:int = 0 
        self.time_window:int = 5 #descrete time steps
        self.bridge_nodes:List[int] = []
        self.tr_times:Dict[(Tuple[int,int],int)] = {}


    def create_problem(self, V:List[int])->None:

        # Combinatorial Optimization Problem 
        self.problem = pl.LpProblem("ContrainedMVMTSP", pl.LpMinimize)

        # Variable to show agent's travel from i to j. 
        self.x = pl.LpVariable.dicts("x", ((i, j, k) for i in V for j in V for k in self.agents), cat='Binary')
    
        # Variable to show position of agent's in the time frame T 
        self.p = pl.LpVariable.dicts("p", ((k,t) for k in self.agents for t in self.timeFrame_per_cluster), cat='Integer')
        
        # Variable to use for subtour elimination constraints 
        # self.u = pl.LpVariable.dicts("u", ((i, k) for i in V for k in self.agents), lowBound=0, upBound=len(V)-1, cat='Integer')

        # Variable to denot when an agent is busy 
        self.busy = pl.LpVariable.dicts("busy", ((k, t) for k in self.agents for t in self.timeFrame_per_cluster), cat='Binary')

        # Variable to handle the action based on timing 
        self.t = pl.LpVariable.dicts("t", ((i, j, k, ts) for i in V for j in V for k in self.agents for ts in self.timeFrame_per_cluster), cat='Binary')

        # Variable that holds information about the energy consumption between two nodes 
        self.e = pl.LpVariable.dicts("e", ((i, k) for i in V for k in self.agents),lowBound=0, upBound=self.max_battery, cat='Continuous')

        # # Variable that handles customer service. 
        # self.z = pl.LpVariable.dicts("z_ik", ((i, k) for i in V for k in self.agents), lowBound=0, upBound=1, cat='Binary')
        
        self.wait = pl.LpVariable.dicts("wait", ((k, t) for k in self.agents for t in self.timeFrame_per_cluster), cat="Binary")

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
        print(nodes_dict)
        
        if pl.LpStatus[self.problem.status] != 'Optimal': 
            logger.info(f"Problem is not optimal for number of constraints {len(self.problem.constraints)}, returning None...")
            sys.exit(1)

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
        logger.info(f"Optimal Solution Found | Number of Constraints {len(self.problem.constraints)}")

        logger.debug("Validating solutions....")
        self.validate_paths(paths=self.paths, nodes_dict=nodes_dict)
        logger.debug("Solutions validated successfully...")


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
            cost, R_points, self.bridge_nodes, nodes_dict, self.initial_population = process_extraction(self, context, depot_id, self.employed_agents)
        except Exception as e: 
            logger.exception(f"Error processing cluster {cluster_id}: {e}")
            return 
        
        # Step 3: Get the best solution from the initial paths 
        if self.initial_population is None:
            G = create_model_graph(
                cost=cost['travel_time'], 
                nodes=nodes_dict, 
                weights={'travel_time':1}
            )
            mst = nx.minimum_spanning_tree(G, weight='weight')
            estimated_time = sum(edge[2]['weight'] for edge in mst.edges(data=True))
            total_time = math.ceil(estimated_time)
        else: 
            best_agent = min(self.initial_population.items(), key=lambda item: item[1][1])
            best_path = best_agent[1][0]
            total_time = math.ceil(sum(
                self.get_travel_time(i, i+1, best_path)
                for i in range(len(best_path)-1)
            ))

        # NOTE: Try it without the self.moment variable. Every time frame is specific to that cluster NOT the whole simulation. 
        # self.timeFrame_per_cluster = list(range(self.moment, self.moment + total_time + 1))
        
        if total_time == 0: 
            logger.error(f"Total time is 0 for cluster {cluster_id}")
            raise ValueError(f"Total time is 0 for cluster {cluster_id}")
        
        self.timeFrame_per_cluster = list(range(0, total_time + 1))
        V_nodes = list(nodes_dict.keys())

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


    def set_constraints_for_multi_agent(self, V_nodes:List[int], nodes_dict:Dict[int, int], R_points:List[int]): 

        logger.info(f"Setting constraints for multi-agent problem...")
        employed_agents = ["Agent_" + str(agent_id) for agent_id in self.employed_agents]
        list_of_agents = {x:int(x.split('_')[-1]) for x in employed_agents}
        header = list_of_agents[employed_agents[0]]
        reverse_nodes = {v: k for k, v in nodes_dict.items()}
        
        # NOTE: depot_ind is confirmed to be correct. 
        depot_ind = self.get_depot_index(nodes_dict, header)
        valid_arcs = [(i,j) for i in V_nodes for j in V_nodes if i != j and i != depot_ind and j != depot_ind]

        in_arcs = defaultdict(list)
        out_arcs = defaultdict(list)

        for i, j in valid_arcs:
            out_arcs[i].append(j)
            in_arcs[j].append(i)

        self.tr_times = {(i,j):self.get_travel_time(i,j,nodes_dict) for i in V_nodes for j in V_nodes}
        T_max = max(self.tr_times[(i,depot_ind)] for i in V_nodes if i != depot_ind)

        deallocate_memory(valid_arcs)
        if self.enable_ga: # NOTE: Finalized 
              for a in self.employed_agents: 
                  for i in range(len(self.initial_population[a][0])-1): 
                      node = reverse_nodes[self.initial_population[a][0][i]]
                      next_node = reverse_nodes[self.initial_population[a][0][i+1]] 
                      self.x[node, next_node, header].setInitialValue(1) 

        # Allow multiple visits (essential for MV-TSP)
        if "const_0" in self.constraints: #NOTE: FINALIZED 
            try: 
                for k, v in list_of_agents.items(): 
                    for j in V_nodes: 
                        self.problem += pl.lpSum(
                            self.x[i,j,v] for i in V_nodes if i != j
                        ) <= R_points[j], f"Allowed_visits_for_each_agent_{k}_for_node_{j}"

                        self.problem += pl.lpSum(
                            self.x[j,i,v] for i in V_nodes if i != j
                        ) <= R_points[j], f"Allowed_exits_for_each_agent_{k}_for_node_{j}"



                logger.debug(f"Constraint | const_0 - All nodes visited multiple times in total | set for cluster ")
            except Exception as e:
                logger.exception(f"Error setting constraint const_0 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_0 for cluster")

        # Only allow a single travel from depot to all nodes and the reverse as well.
        if "const_1" in self.constraints:#NOTE: FINALIZED
            try: 
                for k, v in list_of_agents.items():
                    self.problem += pl.lpSum(
                        self.x[depot_ind, j, v]
                        for j in V_nodes if j != depot_ind and nodes_dict[j] not in self.bridge_nodes
                    ) == R_points[depot_ind], f"{k}_enters_single_area_from_depot_{depot_ind}"

                    self.problem += pl.lpSum(
                        self.x[i, depot_ind, v]
                        for i in V_nodes if i != depot_ind and nodes_dict[i] not in self.bridge_nodes
                    ) == R_points[depot_ind], f"{k}_leaves_single_area_to_depot_{depot_ind}"
            
                logger.debug(f"Constraint | const_1 - Each agent enters and leaves the depot once | set for cluster ")
            except Exception as e:
                logger.exception(f"Error setting constraint const_1 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_1 for Cluster") 
            
        # if "const_2" in self.constraints:#NOTE:FINALIZED
        #     try:
        #         for k, v in list_of_agents.items():
        #             self.problem += self.p[v,self.timeFrame_per_cluster[0]] == depot_ind, f"Positional_variable_at_start_of_journey_for_{k}" 
        #             self.problem += self.p[v,self.timeFrame_per_cluster[-1]] == depot_ind, f"Positional_variable_at_end_of_joureny_for_{k}"

        #         logger.debug(f"Constraint | const_2 - Positional variable at start and end of journey (depot)| set for cluster ")
        #     except Exception as e: 
        #         logger.exception(f"Error setting constraint const_2 for cluster: {e}")
        #         raise ValueError(f"Error in setting constraint const_2 for Cluster")

        # Allow an agent to start his journey whenever it fits best and finish as well at a different time that before (not time frame [-1])
        if "const_3" in self.constraints: #NOTE FInalized. Has to be bigger than 1 in order to not produce very long time results. >= than one means that for multiple time steps the depot to j is true which is something we want for the solution. 
            try: 
                for k, v in list_of_agents.items():
                    for j in V_nodes:
                        if j != depot_ind:
                            valid_departure_window = self.timeFrame_per_cluster[:-(self.tr_times[(depot_ind, j)] + self.tr_times[(j, depot_ind)])]
                            self.problem += pl.lpSum(self.t[depot_ind, j, v, t] for t in valid_departure_window) >= 1, \
                                f"{k}_leaves_depot_{depot_ind}_for_node_{j}_within_valid_time"
                             
                            valid_return_window = self.timeFrame_per_cluster[-(self.tr_times[(j, depot_ind)] + 1):]
                            self.problem += pl.lpSum(self.t[j, depot_ind, v, t] for t in valid_return_window) >= 1, \
                                f"{k}_returns_to_depot_{depot_ind}_from_node_{j}_within_valid_time"
                
                logger.debug(f"Constraint | const_3 - Each agent leaves and enters the depot at a specific interval | set for cluster ")
            
            except Exception as e:
                logger.exception(f"Error setting constraint const_3 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_3 for Cluster")

        # Enforce that only a single journey from i -> j exists so that the reverse is not possible (j -> i) 
        if "const_4" in self.constraints: 
            try:
                for k, v in list_of_agents.items():
                    self.problem += pl.lpSum(self.x[depot_ind, j, v] for j in V_nodes if j != depot_ind ) + \
                                    pl.lpSum(self.x[i, depot_ind, v] for i in V_nodes if i != depot_ind ) == 2, f"{k}_start_&_finishes_at_depot_{depot_ind}"  
                logger.debug(f"Constraint | const_4 - Each agent starts and ends at the depot | set for cluster ") 
            except Exception as e:
                logger.exception(f"Error setting constraint const_4 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_4 for Cluster")
        
        # Prohibit Depot looping for each agent.
        if "const_5" in self.constraints:#NOTE:FINALIZED
            try: 
                for k, v in list_of_agents.items():
                    self.problem += self.x[depot_ind, depot_ind, v] == 0,  f"No_loop_at depot_{depot_ind}_for_{k}_at_any_timepoint"

                logger.debug(f"Constraint | const_5 - No loop at depot | set for cluster ")
            except Exception as e:
                logger.exception(f"Error setting constraint const_5 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_5 for Cluster")
            
        # Only one visit from i to j and from j to i except for bridge nodes. We don't exclude the depots here as they also are a 1 out 1 in node. 
        if "const_6" in self.constraints: #NOTE:FINALIZED
            try: 
                for k, v in list_of_agents.items(): 
                    for i in V_nodes: 
                        if nodes_dict[i] not in self.bridge_nodes and i != depot_ind:
                            self.problem += pl.lpSum(self.x[i,j,v] for j in V_nodes if i != j) == 1, f"Only_one_visit_from_i_to_j_for_agent_{k}_for_node_{i}"
                        elif nodes_dict[i] in self.bridge_nodes and i!= depot_ind:
                            self.problem += pl.lpSum(self.x[i,j,v] for j in V_nodes if i != j) == R_points[i], f"Only_one_visit_from_i_to_j_for_agent_{k}_for_node_{i}"

                    for j in V_nodes:
                        if nodes_dict[j] not in self.bridge_nodes and j != depot_ind:
                            self.problem += pl.lpSum(self.x[i,j,v] for i in V_nodes if i != j) == 1, f"Only_one_visit_from_j_to_i_for_agent_{k}_for_node_{j}"
                        elif nodes_dict[j] in self.bridge_nodes and j!=depot_ind:
                            self.problem += pl.lpSum(self.x[i,j,v] for i in V_nodes if i != j) == R_points[j], f"Only_one_visit_from_j_to_i_for_agent_{k}_for_node_{j}"
                       
                logger.debug(f"Constraint | const_6 - Each agent enters and leaves each node once | set for cluster ") 
            except Exception as e:
                logger.exception(f"Error setting constraint const_6 for cluster: {e}")
                raise ValueError(f"Error in setting constraint const_6 for Cluster")
            
        # Collision avoidance / Unique agent per node 
        if "const_7" in self.constraints: #NOTE:FINALIZED
            for i in out_arcs: 
                for j in out_arcs[i]:
                    # if nodes_dict[j] not in self.bridge_nodes and nodes_dict[i] not in self.bridge_nodes:
                        for step in self.timeFrame_per_cluster:
                            self.problem += pl.lpSum(self.t[i,j,v,step] for _,v in list_of_agents.items()) <= 1, f"Unique_Time_visits_constraint_at_travel_{i}_{j}_at_time_{step}" 
    
        # Synchronization between the time and space 
        if "const_8" in self.constraints: #NOTE:FINALIZED
            # for k, v in list_of_agents.items(): 
            #     # for i in out_arcs: 
            #     #     for j in out_arcs[i]: 
            #     for i in V_nodes: 
            #         for j in V_nodes: 
            #             if i != j:        
            #                 for step in self.timeFrame_per_cluster: 
            #                     if step + self.tr_times[(i,j)] - 1 < T_max:    
            #                         self.problem += pl.lpSum(
            #                             self.t[i,j,v,t] for t in range(step, step + self.tr_times[(i,j)])
            #                         ) == self.tr_times[(i,j)]*self.x[i,j,v]


            # Performance improvements for the synchronization between time and space
            constraint_counter = 0  
            for k, v in list_of_agents.items(): 
                for i in out_arcs: 
                    for j in out_arcs[i]: 
                        if nodes_dict[i] not in self.bridge_nodes and nodes_dict[j] not in self.bridge_nodes: 
                            for step in self.timeFrame_per_cluster: 
                                duration = step + self.tr_times[(i,j)] -1 
                                if duration <= T_max:
                                   
                                    self.problem += pl.lpSum(
                                        self.t[i,j,v,t] for t in range(step, step + duration)
                                    ) <= duration*self.x[i,j,v]
                                   
                                    self.problem += pl.lpSum(
                                        self.t[i,j,v,t] for t in range(step, step + duration)
                                    ) >= duration*self.x[i,j,v]
                                    
                                    constraint_counter +=2 

            logger.info(f"Constraint | const_8 - Synchronization between time and space | set for cluster. Total constraints: {constraint_counter}")


        # Only one travel from i to j for the entirety of the time frame. This cannotbe used if we opt to align all time steps individually. 
        if "const_9" in self.constraints: #NOTE:FINALIZED
            for k, v in list_of_agents.items(): 
                
                self.problem += pl.lpSum(self.t[depot_ind, j, v, t]
                    for j in V_nodes if j != depot_ind 
                    for t in self.timeFrame_per_cluster[:-(self.tr_times[(j, depot_ind)] + 1)]
                ) >= 1, f"Only_one_travel_from_depot_to_j_for_agent_{k}_based_on_time"
               
                self.problem += pl.lpSum(self.t[j,depot_ind,v,t]
                    for j in V_nodes if j != depot_ind 
                    for t in self.timeFrame_per_cluster[-(self.tr_times[(j, depot_ind)] + 1):]
                ) >= 1, f"Only_one_travel_from_j_to_depot_for_agent_{k}_based_on_time"


        if "const_10" in self.constraints: #NOTE:FINALIZED
            for k, v in list_of_agents.items():
                for i in V_nodes:
                    for j in V_nodes:
                        if i != j:
                            for step in self.timeFrame_per_cluster:
                                arrival_time = step + self.tr_times[(i, j)]
                                # ensure time index exists
                                if arrival_time + 1 in self.timeFrame_per_cluster:
                                    self.problem += self.t[i, j, v, arrival_time] <= self.t[j, j, v, arrival_time + 1], \
                                        f"TimeProgress_{i}_{j}_at_{step}_agent_{k}" 
                                    

        if "const_11" in self.constraints: # NOTE : Checked. Maybe I need to change the nodes that are constrained to include the depots. 
            # for i in V_nodes: 
            #     for j in V_nodes:
            #         if i != j: 
            #             for k, v in list_of_agents.items(): 
            #                 self.problem += self.x[i,j,v] + self.x[j,i,v] <= 1, f"No_loops_in_path_for_{k}_at_{i}_{j}"
            for k, v in list_of_agents.items():
                for i in V_nodes:
                    for j in V_nodes:
                        if i != j and nodes_dict[i] not in self.bridge_nodes and nodes_dict[j] not in self.bridge_nodes: 
                            for t in self.timeFrame_per_cluster:
                                if t + 1 in self.timeFrame_per_cluster:
                                    self.problem += self.t[i, j, v, t] + self.t[j, i, v, t + 1] <= 1, f"No_immediate_loop_{i}_{j}_time_{t}_agent_{k}"

        if "const_12" in self.constraints: #NOTE : Checked. Changed the nodes included in the constraints to account for all travels between the agents. 
            for k1, v1 in list_of_agents.items() : 
                for k2, v2 in list_of_agents.items() : 
                    if k1 != k2 : 
                        for step in self.timeFrame_per_cluster: 
                            # self.problem += pl.lpSum(self.t[i, j, v1, step] - self.t[i,j, v2, step] for i in out_arcs for j in out_arcs[i]) != 0, f"Agent_unique_paths_for_{k1}_and_{k2}_at_time_{step}"
                            self.problem += pl.lpSum(self.t[i, j, v1, step] - self.t[i,j, v2, step] for i in out_arcs for j in out_arcs[i]) <= 0, f"Agent_unique_paths_for_{k1}_and_{k2}_at_time_{step}"

        if "const_13" in self.constraints: # NOTE: This only seems to have an effect on the first agent it encounters. 
                try:
                    for k, v in list_of_agents.items():
                        for i in out_arcs:
                            for j in out_arcs[i]:
                                self.problem += self.e[j,v] >= self.e[i,v] - self.normalized_battery[nodes_dict[i]-1][nodes_dict[j]-1] * self.x[i,j,v], f"Update_remaining_energy_{i}_{j}_for_{k}"
                                self.problem += self.e[i,v] >= self.normalized_battery[nodes_dict[i]-1][nodes_dict[j]-1] * self.x[i, j, v],f"No_travel_if_low_energy_{i}_{j}_for_{k}"
                            self.problem += self.e[i,v] >= self.normalized_battery[nodes_dict[i]-1][nodes_dict[depot_ind]-1] * self.x[i, depot_ind, v],f"Enough_energy_to_return_to_depot_from_{i}_for_{k}"
                      
                        for i in out_arcs: 
                            for j in out_arcs[i]: 
                                # for step in self.timeFrame_per_cluster[self.tr_times[(depot_ind,i)]:-(self.tr_times[(i,j)]+self.tr_times[(j,depot_ind)])]: 
                                for step in self.timeFrame_per_cluster: 
                                    self.problem += self.e[j,v] >= self.e[i,v] - self.normalized_battery[nodes_dict[i]-1][nodes_dict[j]-1] * pl.lpSum(self.t[i, j, v, step]), f"Energy_update_{i}_{j}_at_time_{step}_for_{k}"

                            self.problem += self.e[i,v] >= self.normalized_battery[nodes_dict[i]-1][nodes_dict[j]-1] * \
                                pl.lpSum(
                                    self.t[i,depot_ind,v,t] for t in self.timeFrame_per_cluster[-self.tr_times[(i,depot_ind)]:]
                                ), f"Ensure_depot_return_from{i}_for_{k}_for_correct_time_Steps"
                    logger.debug(f"Constraint | const_13 - Update remaining energy | set for cluster ")
  
                except Exception as e:
                  logger.exception(f"Error setting constraint const_13 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_13 for Cluster")

        if "const_14" in self.constraints: # NOTE this is unnecessary. 
            try: 
                for k, v in list_of_agents.items():
                    for i in out_arcs:
                        for t in self.timeFrame_per_cluster: 
                            self.problem += self.e[i,v] >= 0, f"Energy_cannot_be_negative_{i}_for_{k}_at_time_{t}"
  
                    self.problem += self.e[depot_ind, v] == self.max_battery_norm, f"Every_agent_starts_with_full_battery_{k}"
  
                logger.debug(f"Constraint | const_14 - Energy cannot be negative | set for cluster ")
            except Exception as e:
                  logger.exception(f"Error setting constraint const_14 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_14 for Cluster")

        if "const_15" in self.constraints:
            M = len(V_nodes) 
            for k, v in list_of_agents.items():
                for t in self.timeFrame_per_cluster:
                    # The agent is at the depot until they begin their first travel
                    # self.problem += self.p[v, t] == depot_ind + (1 - pl.lpSum(self.t[depot_ind, j, v, t] for j in V_nodes if j != depot_ind)) * M  # Big-M allows flexibility before departure
                    # self.problem += self.p[v ,t] == depot_ind + (1 - pl.lpSum(self.t[i, depot_ind, v, t] for i in V_nodes if i != depot_ind)) * M

                    # Improvements to include departure idle positioning and the wait variable 
                    self.problem += self.p[v,t] <= depot_ind + (1-self.wait[v,t]) * M, f"Wait_implies_at_depot_upper_{k}_{t}"
                    self.problem += self.p[v,t] >= depot_ind - (1-self.wait[v,t]) * M, f"Wait_implies_at_depot_lower_{k}_{t}"

        if "const_16" in self.constraints:
            for k, v in list_of_agents.items():
                for i in V_nodes:
                    for j in V_nodes:
                        if i != j:
                            travel_duration = self.tr_times[(i, j)]
                            for t_start in self.timeFrame_per_cluster[:-travel_duration]:
                                for dt in range(travel_duration):
                                    t = t_start + dt
                                    self.problem += self.busy[v, t] >= self.t[i, j, v, t_start], f"Busy_if_travel_{i}_{j}_starts_at_{t_start}_for_{k}_covers_{t}"

        if "const_17" in self.constraints:
            for k, v in list_of_agents.items():
                for t in self.timeFrame_per_cluster:
                    self.problem += self.busy[v, t] + self.wait[v, t] <= 1


        if "const_18" in self.constraints: 
            for k, v in list_of_agents.items():
                for i in V_nodes:
                    if i != depot_ind:
                        valid_departure_window = self.timeFrame_per_cluster[:-(self.tr_times[(depot_ind, i)] + self.tr_times[(i, depot_ind)])]
                        self.problem += pl.lpSum(self.t[depot_ind, i, v, t] for t in valid_departure_window
                        ) >= self.x[depot_ind,i,v] , f"Dynamic_time_enforcement_{depot_ind}_{i}_for_{k}_time_{step}"
                            
                for j in V_nodes:
                    if j != depot_ind:
                        valid_return_window = self.timeFrame_per_cluster[-(self.tr_times[(j, depot_ind)] + 1):]

                        self.problem += pl.lpSum(
                            self.t[j, depot_ind, v, t] 
                            for t in valid_return_window
                        ) >= self.x[j, depot_ind, v], f"Dynamic_time_enforcement_{j}_{depot_ind}_for_{k}_time_{step}"


        if "const_19" in self.constraints: 
            for k, v in list_of_agents.items(): 
                for step in self.timeFrame_per_cluster: 
                    self.problem += pl.lpSum(
                        self.t[i,j,v,step] for i in V_nodes for j in V_nodes if i!=j \
                        if nodes_dict[i] not in self.bridge_nodes and nodes_dict[j] not in self.bridge_nodes
                    ) == 1, f"Only_one_journey_per_agent_{k}_at_time_{step}"


        if "const_20" in self.constraints: 
            # for k, v in list_of_agents.items(): 
            #     self.problem += pl.lpSum(
            #         self.x[i,j,v] for i in V_nodes for j in V_nodes if i!=j
            #     ) <= R_points, f"Total_number_of_agents_in_the_system_{k}"

            for k,v in list_of_agents.items():
                for t in self.timeFrame_per_cluster:
                    # Agent waits at t if no departure has started yet
                    self.problem += self.wait[v, t] >= 1 - pl.lpSum(
                        self.t[depot_ind, j, v, tau]
                        for j in V_nodes if j != depot_ind
                        for tau in self.timeFrame_per_cluster if tau < t), \
                        f"Wait_before_departure_{v}_{t}"


        if "const_22" in self.constraints: 
            
            constraint_counter = 0
            M = len(V_nodes) 
            for k, v in list_of_agents.items(): 
                for i in V_nodes: 
                    for j in V_nodes:
                        if i == j and (i==depot_ind or j==depot_ind) : continue 
                        if nodes_dict[i] in self.bridge_nodes or nodes_dict[j] in self.bridge_nodes: continue
                        trip_time = self.tr_times[(i,j)]
                        for step in self.timeFrame_per_cluster[:-trip_time]:
                            if (i, j, v, step) in self.t and self.t[i,j,v,step].name in self.problem.variablesDict():
                                self.problem += self.p[v, step + trip_time] <= j + (1 - self.t[i, j, v, step]) * M, f"Positional_alignment_with_step_{step}_for_{k}_at_{j}{i}"
                                self.problem += self.p[v, step + trip_time] >= j - (1 - self.t[i, j, v, step]) * M, f"Positional_alignment_with_step_{step}_for_{k}_at_-{j}{i}"
                                constraint_counter += 1
            print(f"for Const_22 | added {constraint_counter} constraints")


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


    def validate_paths(self, paths, nodes_dict):
         
        max_time_steps = self.timeFrame_per_cluster[-1]
        all_paths = {} 
        key_points = {} 
        for agent_id, path in paths.items(): 
            visit_nodes = [] 
            seen_edges = set()
            # Reject agents that haven't been used at this point.
        
            if len(path) == 0 or len(path[0])==0: 
                continue 
            
            depot_ind = self.get_depot_index(nodes_dict, agent_id)
            if path[0][-1][1] != nodes_dict[depot_ind]: 
                logger.error(f"Agent {agent_id} did not finish at depot [{path[0][-1] }|{nodes_dict[depot_ind]}]")
            for i in range(len(path[0])-1):
                
                step = path[0][i] 
                source_node = step[0] 
                target_node = step[1]
                time_step = step[2] 

                edge = (source_node, target_node)
                keypoint = (target_node, time_step)
                
                if keypoint in key_points and target_node != nodes_dict[depot_ind] and source_node!=nodes_dict[depot_ind]: 
                    logger.error(f"Collision: Agent {agent_id} and Agent {key_points[keypoint]} from node {source_node} at node {keypoint[0]} at time {keypoint[1]}")


                key_points[keypoint] = agent_id
                
                if edge in seen_edges: 
                    logger.debug(f"Edge {edge} already seen for agent {agent_id}")
                    

                seen_edges.add(edge)
                next_step = path[0][i+1]
                next_source_node = next_step[0] 
                next_target_node = next_step[1] 
                next_time_step = next_step[2]
                 
                if next_time_step < time_step: 
                    logger.error(f"Agent {agent_id} visited node {next_source_node} at time {next_time_step} before visiting node {source_node} at time {time_step}")

                if next_source_node != target_node:
                    logger.error(f"Agent {agent_id} visited node {next_source_node} at time {next_time_step} instead of node {target_node} at time {time_step}")

                if next_target_node == target_node: 
                    logger.error(f"Agent {agent_id} visited node {target_node} at time {time_step} before visiting node {next_target_node} at time {next_time_step}") 

                if source_node not in visit_nodes and source_node != nodes_dict[depot_ind]: 
                    visit_nodes.append(source_node)

                if i != len(path[0])-2 and target_node == path[0][0][0]:
                    logger.error(f"Agent {agent_id} visited node {target_node} at time {time_step} before visiting node {path[0][0][0]} at time {path[0][-1][2]}")

                if time_step > max_time_steps: 
                    logger.error(f"Agent {agent_id} visited node {target_node} at time {time_step} which is greater than the maximum time step {max_time_steps}")

            edge_sequence = tuple((step[0], step[1]) for step in path[0])
            all_paths[agent_id] = edge_sequence

            if len(visit_nodes) != len(nodes_dict)-1 : 
                logger.error(f"Agent {agent_id} visited only {len(visit_nodes)} nodes out of {len(nodes_dict)-1}")

        agent_ids = list(all_paths.keys()) 
        for i in range(len(agent_ids)): 
            for j in range(i + 1, len(agent_ids)):
                if all_paths[agent_ids[i]] == all_paths[agent_ids[j]]:
                    logger.error(f"Agents {agent_ids[i]} and {agent_ids[j]} have identical paths!")

        
    def get_travel_time(self, i, j, nodes_dict): 
        return math.ceil(self.travel_cost[nodes_dict[i]-1, nodes_dict[j]-1])