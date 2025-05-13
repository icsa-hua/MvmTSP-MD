from dummy_app.designs.mvmtsp_config import MVMTSPConfig 
from dummy_app.tools.autonomize import deallocate_memory, extract_context_for_cluster, process_extraction, create_model_graph, get_weights
from dummy_app.tools.performance_metrics import Metrics
from dummy_app.designs.cluster import Cluster
from dummy_app.designs.constraint import * 
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


class Builder(MVMTSPConfig):

    def __init__(self, config:Dict[str,Any], trials): 
        super().__init__(config)
        
        self.allow_regionalization:bool = config['regionalization']
        self.enable_ga:bool = config['genetic_algorithm']
        self.constraints:List[str] = config['constraints']
        self.V:pd.DataFrame = pd.DataFrame() 
        self.v:int = 0 
        self.best_path:List[int] = [] 
        self.TimeFrame:List[int] = [range(0,trials,1)]
        self.metrics:object = Metrics(verbose=True) 
        self.clusters_times:Dict[int, int] = {} 
        self.recharge_time_window:int = 5 #descrete time steps
        


    def create_problem(self, V:List[int])->None:
        super().create_problem(V)


    def set_objective(self, distance:Any, energy:Any, time:Any, nodes:Dict):
        pass
         
    
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


    def solve_problem(self, cluster:object):
        cluster.problem.solve(pl.GLPK_CMD(msg=False, options=['--mipgap', '0.05']))
    
    
    def preprocess(self, distances_path, energies, nodes_path, agents, customers_path, ground_users, max_battery):
        data = super().preprocess(distances_path, energies, nodes_path, agents, customers_path, ground_users, max_battery)
        
        self.depots_for_agents = self.assign_agents_to_areas(len(self.agents),self.depots)


        logger.info("Preprocessing completed successfully...")
        return data 


    def set_memory_limit(self, max_memory = 1024):
        return super().set_memory_limit(max_memory)
    
    
    def create_solution(self, cluster:object):
        logger.info(f"Cluster Time Frame is {cluster.timeframe}") 
        if pl.LpStatus[cluster.problem.status] != 'Optimal': 
            logger.info("Problem is not optimal, returning None...")
            sys.exit(1)

        cluster.get_solution()
        

        logger.debug(f"Solutions created for {len(cluster.employed_agents)} agents")
        self.moment += len(cluster.nodes_dict.keys()) + 1 + self.recharge_time_window
        self.clusters_times[self.cluster_id] = self.moment 

        memory_usage = self.metrics.get_memory_usage()
        logger.info(f"Memory usage: {memory_usage:.2f} MB")
        logger.info("Optimal Solution Found!!!!!")
        logger.debug("Validating solutions....")
        # self.validate_paths(paths=self.paths, nodes_dict=nodes_dict)
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

            # Phase 2: Clustering and Prioritization 
            try: 
                priority = self.cluster_prioritization(clusters, cue_groups)
                pbar.update(1)
                logger.debug("Clusters prioritized successfully...")
            except Exception as e:
                logger.exception(f"Error occurred during clustering: {e}")
                raise ValueError("Error occurred during clustering.")

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

            pbar.update(1) 

        logger.info("Problem construction and solution follow...")

        # Phase 5: Problem Construction and Solution
        with tqdm(total=len(clusters), desc="Solving problem...", unit="step") as pbar:
            for (cluster_tuple, agents), cluster in zip(assignments.items(), updated_clusters):
                self.cluster_id = cluster_tuple[0]
                self.clustering(
                    cluster=cluster,
                      cluster_id=cluster_tuple[0],
                        assignment=agents,
                          depot_id=cluster_tuple[1])

                pbar.update(1)
                logger.debug(f"Cluster {cluster_tuple[0]} solved successfully...")
                time.sleep(10)


    def regionalization(self, GDF):
        return super().regionalization(GDF)
    
    
    def cluster_prioritization(self, clusters, cue_groups):
        return super().cluster_prioritization(clusters, cue_groups)
    

    def clustering(self, cluster, cluster_id, assignment, depot_id):

        # Step 1: Create the cluster object to accomodate the problem.  
        cluster_object = Cluster(
            cluster=cluster, 
            id=cluster_id, 
            assignment=assignment, 
            depot_id=depot_id
        )

        context = cluster_object.get_cluster_content(
            distance=self.distance_columns,
            energy=self.energy_columns,
            time=self.travel_time_columns,
            column_names= ["dists", "ees", "travel_times", "area_ids"]
        )

        logger.debug(f"Clustering with {cluster_id} and agents assigned to it: {assignment}")
       
        # Step 2: Process inpute context 
        try: 
            cluster_object.prepare_context(
                context=context, 
                builder=self 
            )
        except Exception as e: 
            logger.exception(f"Error processing cluster {cluster_id}: {e}")
            return 
        
        # Step 3: Estimate the timeframe from the initial paths 
        cluster_object.get_estimated_time_frame(self)


        # Step 4: Create and configure the optimization problem 
        try: 
            cluster_object.problem_formulation(builder=self) 
        except Exception as e:
            logger.exception(f"Error creating problem for cluster {cluster_id}: {e}")
            raise ValueError(f"Error in creating the problem for Cluster {cluster_id}")

        # Step 5 extract solution 
        cluster_object.get_solution() 

        self.moment = len(cluster_object.nodes_dict.keys()) + 1 + self.recharge_time_window
        self.clusters_times[cluster_id] = self.moment



    def set_constraints_for_multi_agent(self, cluster:object): 
        available_constraints = {
            "const_0":constraint_0, "const_1":constraint_1,
            "const_2":constraint_2, "const_3":constraint_3,
            "const_4":constraint_4, "const_5":constraint_5,
            "const_6":constraint_6, "const_7":constraint_7,
            "const_8":constraint_8, "const_9":constraint_9,
            "const_10":constraint_10, "const_11":constraint_11,
            "const_12":constraint_12, "const_13":constraint_13,
            "const_14":constraint_14, "const_15":constraint_15,
            "const_16":constraint_16, "const_17":constraint_17, 
            "const_18":constraint_18
        }

        logger.info(f"Setting constraints for multi-agent problem...")
        
        employed_agents = ["Agent_" + str(agent_id) for agent_id in cluster.employed_agents]
        list_of_agents = {x:int(x.split('_')[-1]) for x in employed_agents}
        header = list_of_agents[employed_agents[0]]
        reverse_nodes = {v: k for k, v in cluster.nodes_dict.items()}
        V_nodes = list(cluster.nodes_dict.keys())

        if self.enable_ga: # NOTE: Finalized 
              for a in cluster.employed_agents: 
                  for i in range(len(cluster.initial_population[a][0])-1): 
                      node = reverse_nodes[cluster.initial_population[a][0][i]]
                      next_node = reverse_nodes[cluster.initial_population[a][0][i+1]] 
                      cluster.x[node, next_node, header].setInitialValue(1) 


        for const in available_constraints:
            if const in self.constraints:
                try: 
                    available_constraints[const](
                        cluster=cluster,
                        builder=self,
                        V_nodes=V_nodes,
                        list_of_agents=list_of_agents,
                    )
                    logger.debug(f"Constraint {const} set successfully...")
                except Exception as e:
                    logger.exception(f"Error setting constraint {const} for cluster: {e}")
                    raise ValueError(f"Error setting constraint {const} for cluster")


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


    def validate_paths(self, paths, nodes_dict, cluster):
         
        max_time_steps = cluster.timeframe[-1]
        all_paths = {} 
        key_points = {} 
        for agent_id, path in paths.items(): 
            visit_nodes = [] 
            seen_edges = set()
            # Reject agents that haven't been used at this point. 
            if len(path) == 0: 
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