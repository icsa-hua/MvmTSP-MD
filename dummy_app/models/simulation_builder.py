from dummy_app.designs.mvmtsp_config import MVMTSPConfig 
from dummy_app.tools.common import deallocate_memory, extract_per_agent_metrics, calculate_totals_from_paths, load_generated_data, calculate_recharge_steps
from dummy_app.tools.performance_metrics import Metrics
from dummy_app.designs.cluster import Cluster
from dummy_app.designs.agents import TSPAgent 
from dummy_app.designs.constraint import * 
from dummy_app.tools.logger import logger 
from typing import Any, List, Dict, Union, Tuple, Mapping

import os 
import sys
import math
import pdb
import time
import uuid
import pulp as pl 
import numpy as np 
import pandas as pd
import networkx as nx
import timeout_decorator 
from tqdm import tqdm 
from collections import defaultdict


class Builder(MVMTSPConfig):

    def __init__(self, config:Dict[str,Any], trials:int): 
        
        super().__init__(
            env_type=config["env_type"], 
            max_battery=config["max_battery"],
            max_coverage_time=config["max_coverage_time"],
            enable_ga=config["enable_ga"],
            scenario=config["scenario"],
            objective_function=config["objective_function"]
        )
        
        self.metrics = Metrics(verbose=True) 
        self.recharge_time_window:int = 5 #descrete time steps
        self.num_constraints = 0 
        self.variables_count = 0
        self.Time = 0 
        self.problem_results = defaultdict()


    def call_genetic_algorithm(self, nodes_dict:Dict[int,int], cost:Dict[str,float], depot:int, verbose:bool=False, population_size:int=200, generations:int=100)->Tuple[List[int],Any]:
        return super().call_genetic_algorithm(nodes_dict, cost, depot, verbose, population_size, generations) 
    

    def assign_agents_to_areas(self, plethos, depots:Any)->Dict[int,int]:
        return super().assign_agents_to_areas(plethos, depots)
    

    def assign_depot_to_cluster(self, clusters:Any, depots_df:pd.DataFrame, distance_matrix):

        def find_duplicate_values(mapping:Dict)->Dict: 
            reverse = {} 
            for key, value in mapping.items():
                reverse.setdefault(value, []).append(key)
            return {val:keys for val, keys in reverse.items() if len(keys) > 1}

        cluster_depot = {} 

        for cluster_id, cluster_df in clusters:
            # Calculate cluster centroid directly
            centroid = cluster_df['Area_id']

            # Find nearest depot 
            min_dist = float('inf')
            best_depot:int = 0 
            for _, depot_row in depots_df.iterrows(): 

                depot_id = int(depot_row['Area_id'])
                for centre in centroid: 
                    dist = distance_matrix[depot_id,int(centre)]
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_depot = depot_id

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
    

    def add_depot_data_to_cluster(self, cluster_df:Tuple[int,pd.DataFrame], depot_row:pd.DataFrame, depot_id:int):
        depot_row= depot_row.copy()
        depot_row  = depot_row[depot_row['Area_id']==depot_id]
        depot_row.loc[:, 'cluster'] = np.mean(cluster_df[1]['cluster'])        
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
                    logger.debug(f"❌ No clusters found for depot {depot}")
                    continue

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


    @timeout_decorator.timeout(3600)
    def solve_problem(self, cluster:Any):
        cluster.problem.solve(pl.GLPK_CMD(msg=False, options=['--mipgap', '0.0','--seed', '42']))
    
    
    def preprocess(self, distances_path, energies_path, nodes_path, agents, customers_path, ground_users, max_battery):
        data = super().preprocess(distances_path, energies_path, nodes_path, agents, customers_path, ground_users, max_battery)
        self.depots_for_agents = self.assign_agents_to_areas(len(self.agents),self.depots)
        logger.info("✅ Preprocessing completed successfully...")
        return data 


    def preprocess_generated_data(self, distance_matrix:np.ndarray, centroids:list, depots:np.ndarray, num_of_agents:int,  v_ver:float,  v_hor:float,  altitude:int,  coverage_time:int,  user_points=defaultdict()):
        data = super().preprocess_generated_data(
            distance_matrix=distance_matrix,
            centroids=centroids, 
            user_points=user_points,
            depots=depots, 
            num_of_agents=num_of_agents,
            v_ver=v_ver, 
            v_hor=v_hor, 
            altitude=altitude,
            coverage_time=coverage_time
        )
        self.depots_for_agents = self.assign_agents_to_areas(plethos=len(self.agents), depots=self.depots)
        logger.debug("✅ Preprocessing of generated data completed successfully...")
        return data 


    def set_memory_limit(self, max_memory = 1024):
        return super().set_memory_limit(max_memory)
    
    
    def create_solution(self, cluster:Any)->Dict[str,List[int]]:
        logger.debug(f"Cluster Time Frame is {cluster.timeframe}") 
        if pl.LpStatus[cluster.problem.status] != 'Optimal': 
            logger.info("❌ Problem is not optimal, returning None...")
            sys.exit(1)

        logger.debug(f"Cluster bridge nodes are {cluster.bridge_nodes}")

        employed_agents = ["Agent_" + str(i) for i in cluster.employed_agents]
        list_of_agents = {name: int(name.split("_")[1]) for name in employed_agents}
        reverse_dict = {v: k for k, v in cluster.nodes_dict.items()}
        paths = {agent: [] for agent in employed_agents}
        max_iterations = len(cluster.nodes_dict.keys())*len(cluster.timeframe) + 1 + self.recharge_time_window

        # visit_nodes = defaultdict(list)
        # for i in cluster.nodes_dict.keys(): 
        #     for k,v in list_of_agents.items(): 
        #         if cluster.visit[i,v].varValue == 1:
        #             visit_nodes[k].append(int(cluster.nodes_dict[i]))

        for k in list_of_agents.values():
            legs = [(i,j, var.value()) for (i,j),var in cluster.t.items()
                    if j == k ]
            print(legs)
            legs = [(i,j,v) for (i,j,v),var in cluster.x.items()
                    if v == k and (var.value() ==1) ]
            print("DEDE")
            print(legs)
            print("FEOIFE")
            legs = [(v, var.value()) for (v),var in cluster.return_step.items()
                    if v == k ]
            print(legs)

        # for agent in list_of_agents.keys():
        #     if len(visit_nodes[agent]) != len(cluster.nodes_dict.keys()):
        #         logger.debug(f"❌ Based on Cluster.Visits Agent {agent} has not visited all nodes, returning None...")

        for agent,v in list_of_agents.items():
            legs =[(i,j,v,t) for (i,j,kk,t),var in cluster.t.items()
                if kk == v and var.value() == 1]
            x_legs =[(i,j,v) for (i,j,kk),var in cluster.x.items()
                if kk == v and var.value() == 1]
            
            print(legs)
            print(x_legs)

            # if len(visit_nodes[agent]) != len(cluster.nodes_dict.keys()):
            #     logger.debug(f"❌ Based on Cluster.Visits Agent {agent} has not visited all nodes, returning None...")
        print("Cluster ", cluster.nodes_dict)
        for agent_name, agent_id in list_of_agents.items():
            edges = set()
            step = -1 
            actual_time = 0

            current_node = reverse_dict[cluster.depot_id]
            iteration = 0 
            triplet = (cluster.depot_id,cluster.depot_id,step)


            while iteration < max_iterations: 
                found_next = False

                for timestep in cluster.timeframe: 
                    if timestep <= step: continue 
                    for next_node in cluster.nodes_dict.keys(): 
                        if next_node == current_node: continue 
                        if cluster.x[current_node, next_node, agent_id].varValue != 1: continue 
                        if cluster.t[current_node, next_node, agent_id, timestep].varValue != 1: continue
                        duration = cluster.tr_times[(current_node, next_node)]
                        # duble = (cluster.nodes_dict[current_node],cluster.nodes_dict[next_node])
                        # paths[agent_name].append(duble)
                        
                        # if duble in edges: 
                        #     logger.debug(f"Edge {duble[0],duble[1]} already visited")
                        
                        # current_node = reverse_dict[duble[1]]
                        # found_next = True

                        if all(
                            cluster.t[current_node, next_node, agent_id, t].varValue == 1
                            for t in range(timestep, timestep + duration)
                            if t in cluster.timeframe
                        ):
                            # This is a legitimate travel
                            triplet = (
                                cluster.nodes_dict[current_node],
                                cluster.nodes_dict[next_node],
                                timestep,
                            )
                     
                            paths[agent_name].append((triplet[0], triplet[1], actual_time))

                                # Optional: append once with duration, or multiple times
                            for d in range(duration):
                                actual_time += 1
                                # paths[agent_name].append((triplet[0], triplet[1], actual_time))


                            wait_step = timestep + duration

                            # if all(cluster.wait[agent_id,t].varValue == 1 
                            #        for t in range(wait_step, wait_step + self.coverage_time)
                            #        if t in cluster.timeframe): 
                               
                            #     for d in range(self.coverage_time): 
                            #         actual_time += 1 
                            #         next_node_idx = cluster.nodes_dict[next_node]
                            #         paths[agent_name].append((next_node_idx, next_node_idx, actual_time))
                            if cluster.nodes_dict[next_node] == cluster.depot_id: 
                                break 
                            wait_step = timestep + duration
                            paths[agent_name].append((cluster.nodes_dict[next_node], cluster.nodes_dict[next_node], actual_time))

                            if cluster.wait[agent_id,wait_step].varValue == 1 :
                                for d in range(self.coverage_time): 
                                    actual_time += 1 
                                    next_node_idx = cluster.nodes_dict[next_node]
                            paths[agent_name].append((cluster.nodes_dict[next_node], cluster.nodes_dict[next_node], actual_time))

                        edges.add(triplet)
                        current_node = reverse_dict[triplet[1]]
                        step = timestep + duration 
                        found_next = True
                        break  # next timestep

                    if found_next:
                        break
                           
                if not found_next:
                    logger.warning(f"❌ No valid move found for agent {agent_name} at iteration {iteration}. Ending early.")
                    break
                
                if current_node == cluster.depot_id and step > 0:
                    break

                iteration += 1
                 
            if paths[agent_name][-1][1] != cluster.depot_id:
                print(paths[agent_name]) 
                raise Exception(f"❌ Agent {agent_name} did not return to depot")
            
        
        print(paths) 
        logger.debug(f"✅ Solutions created for {len(cluster.employed_agents)} agents")
        unique_nodes_among_paths = set()
        
        for path in paths: 
            for duble in paths[path]:
                if duble[0] not in unique_nodes_among_paths: 
                    unique_nodes_among_paths.add(duble[0]) 
                if duble[1] not in unique_nodes_among_paths:  
                    unique_nodes_among_paths.add(duble[1])

        logger.info(f"The amount of unique nodes visited COLLECTIVELY is {len(unique_nodes_among_paths)}/{len(cluster.nodes_dict.values())}")
        memory_usage = self.metrics.get_memory_usage()
        logger.info(f"Memory usage: {memory_usage:.2f} MB")
        logger.info("✅ Optimal Solution Found!!!!!")
        self.num_constraints += len(cluster.problem.constraints)
        self.variables_count += len(cluster.problem.variables())
        logger.info(f"Cluster has nodes _dict {cluster.nodes_dict.values()} with length {len(cluster.nodes_dict.values())}")
        self.validate_paths(paths=paths, nodes_dict=cluster.nodes_dict, cluster=cluster)
        logger.debug("✅ Solutions validated successfully...")
        return paths 


    def createGeoDataset(self, data):
        return super().createGeoDataset(data)
     

    def run_model(self, distance_matrix:np.ndarray, data:pd.DataFrame, cue_groups:Dict[int,List[Any]])->Dict:
        self.user_points = cue_groups
        logger.debug("Running combinatorial problem constructor...")
        with tqdm (total=8, desc="Preparing Problem with clustering") as pbar: 

           # Phase 1: Preprocessing and regionalization (geospatial clustering)
            try: 
                data, depots = self.separate_depots_from_clusters(data)
                pbar.update(1)
                logger.debug("✅ Depots separated from clusters successfully...")
                gdf = self.createGeoDataset(data)
                pbar.update(1)
                logger.debug("✅ GeoDataset created successfully...")
                clusters = self.regionalization(gdf)
                pbar.update(1)
                logger.debug("✅ Clusters created successfully...")
                
            except Exception as e:
                logger.exception(f"❌ Error occurred during regionalization: {e}")
                raise ValueError("Error occurred during regionalization.")
            

            # Phase 2: Clustering and Prioritization 
            try: 
                priority = self.cluster_prioritization(clusters, cue_groups, distance_matrix)
                pbar.update(1)
                logger.debug("✅ Clusters prioritized successfully...")
            except Exception as e:
                logger.exception(f"❌ Error occurred during clustering: {e}")
                raise ValueError("Error occurred during clustering.")

            # Phase 3: Agent Assignment for all clusters 
            try: 
                cluster_with_depots, same_depot_agents = self.assign_depot_to_cluster(clusters, depots, distance_matrix=distance_matrix)
                pbar.update(1)
                logger.debug("✅ Depots assigned to clusters successfully...")
                assignments = self.allocate_agents_to_clusters(cluster_with_depots, priority, same_depot_agents)
                pbar.update(1)

            except Exception as e:
                logger.exception(f"❌ Error occurred during agent assignment: {e}")
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
                logger.debug("✅ Final refinements added to clusters successfully...")

            except Exception as e:
                logger.exception(f"❌ Error occurred during final cluster refinement: {e}")
                raise ValueError("Error occurred during final cluster refinement.")

            pbar.update(1) 

        paths = {} 
        self.metrics.start_performance_timer() 

        # Phase 5: Problem Construction and Solution
        with tqdm(total=len(clusters), desc="Solving problem...", unit="step") as pbar:
            for (cluster_tuple, agents), cluster in zip(assignments.items(), updated_clusters):
                paths[f"Cluster_{cluster_tuple[0]}"] = self.clustering(
                    cluster=cluster,
                      cluster_id=cluster_tuple[0],
                        assignment=agents,
                          depot_id=cluster_tuple[1])
                pbar.update(1)
                time.sleep(2)
                logger.debug(f"✅ Cluster {cluster_tuple[0]} solved successfully...")
        
        # Step 6: Agent Generation for simulation
        self.metrics.end_performance_timer() 
        logger.info("Total Number of Constraints : {}".format(self.num_constraints))
        logger.info("Total Number of Variables : {}".format(self.variables_count))
        pdb.set_trace()
        # Step 6: Flatten all the paths to form a single path for each agent
        paths = self.flatten_paths_on_time(paths)  
       
        # Step 7: Transform positions to coordinates
        agents_paths_clusters = defaultdict(dict)

        for agent in paths.keys(): 
            agents_paths_clusters[agent] = {
                'path': self.get_coordinates_for_path(paths[agent])
            }
            
        # Step 8: Add interpolation steps for the paths (visualizatino) 
        agents_paths_clusters = self.post_process_interpolation(agents_paths_clusters)
        path_times = []
        for agent in agents_paths_clusters.keys(): 
            path_times.append(len(agents_paths_clusters[agent]))
        
        return agents_paths_clusters     
    

    def get_coordinates_for_path(self, path): 
        coordinates = []
        for point in path: 
            try:
                current_node = self.V.index[self.V['Area_id'] == point[0]][0] 
                next_node = self.V.index[self.V['Area_id'] == point[1]][0]
                current_coords = (int(self.V['X_coords'].iloc[current_node]), int(self.V['Y_coords'].iloc[current_node]))
                next_coords = (int(self.V['X_coords'].iloc[next_node]), int(self.V['Y_coords'].iloc[next_node]))
                coordinates.append((current_coords, next_coords, point[2]))
            except IndexError:
                logger.error(f"❌ Node {point[0]} or {point[1]} not found in the dataframe.")
                continue
        return coordinates


    def regionalization(self, GDF):
        return super().regionalization(GDF)
    
    
    def cluster_prioritization(self, clusters, cue_groups: Mapping[int, Any], distance_matrix):
        return super().cluster_prioritization(clusters, cue_groups, distance_matrix)
    

    def clustering(self, cluster, cluster_id, assignment, depot_id)->Dict:

        SCENARIO = self.scenario 
        OBJECTIVE = self.objective_function

        # Step 1: Create the cluster object to accomodate the problem.  
        cluster_object = Cluster(
            cluster=cluster, 
            id=cluster_id, 
            assignment=assignment, 
            depot_id=depot_id, 
            max_battery=self.max_battery
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
            logger.debug(f"✅ Context prepared for cluster {cluster_id} successfully...")
        except Exception as e: 
            logger.exception(f"❌ Error processing cluster {cluster_id}: {e}")
            raise ValueError(f"Error processing cluster {cluster_id}: {e}")

        # Step 3: Estimate the timeframe from the initial paths 
        cluster_object.get_estimated_time_frame(self)

        if OBJECTIVE == 'coverage':
            self.get_cluster_coverage(cluster_object)

        # deallocate_memory(context)

        # cluster_object.R_points = list(np.ones(len(cluster_object.nodes_dict)))
        # Step 4: Create and configure the optimization problem 
        try: 
            paths = cluster_object.problem_formulation(builder=self, scenario=SCENARIO, objective_function=OBJECTIVE ) 
        except Exception as e:
            logger.exception(f"❌ Error creating problem for cluster {cluster_id}: {e}")
            raise ValueError(f"Error in creating the problem for Cluster {cluster_id}")

        # Step 5 extract solution 
        # paths = cluster_object.get_solution()
        # Step 6: Add the recharge phase & synchronize agents 
     
        results = extract_per_agent_metrics(
            paths=paths, 
            costs=cluster_object.cost,
            coverage_energy=self.average_coverage_energy
        )
        totalDistance, totalEnergy, totalTime = calculate_totals_from_paths(
            results=results
        )

        self.problem_results[f'Cluster_{cluster_object.id}'] = {
            "scenario_name":SCENARIO,
            "objective_function":OBJECTIVE,
            "agent_results":results, 
            "Total Distance": totalDistance, 
            "Total Energy":totalEnergy, 
            "Total Time": totalTime, 
            "Average Throughput": cluster_object.R, 
            "Average SINR" : cluster_object.sinr
        }

        filename = self.create_filename(cluster_object.id) 
        df = pd.DataFrame([self.problem_results[f'Cluster_{cluster_object.id}']])
        df.to_csv(filename, mode='a', index=False, header=False) 

        paths = self.synchronize_agent_paths(paths, cluster_object)
        print("Amount of CONSTRAINTs: ", len(cluster_object.problem.constraints))
        pdb.set_trace()
        deallocate_memory(cluster_object)
        return paths 


    def get_depot_index(self, ordered_nodes, k): 
        axx = [i for i, value in enumerate(ordered_nodes.values()) if value == self.depots_for_agents[k]]
        return axx[0]


    def validate_paths(self, paths, nodes_dict, cluster):
        max_time_steps = cluster.timeframe[-1]
        reverse = {v: k for k, v in nodes_dict.items()}
        depot_ind = reverse[cluster.depot_id]
        all_paths = {} 
        key_points = {} 
        
        for agent_id, path in paths.items(): 
            visit_nodes = set() 
            seen_edges = set()
            # Reject agents that haven't been used at this point. 
            if len(path) == 0: 
                logger.debug(f"Agent {agent_id} has no path")
                continue 
            if path[-1][1] != cluster.depot_id: 
                raise ValueError(f"{agent_id} does not return to depot")
            
            visit_nodes.add(cluster.depot_id)
               

            # 2.  include the very last arrival node
            for step in path:
                for node in (step[0], step[1]):
                    if node not in (cluster.depot_id,) + tuple(cluster.bridge_nodes):
                        visit_nodes.add(node)

            for i in range(len(path)-1):
                
                step = path[i] 
                source_node = step[0] 
                target_node = step[1]
                time_step = step[2] 

                edge = (source_node, target_node)
                keypoint = (target_node, time_step)
                
                if keypoint in key_points and target_node != nodes_dict[depot_ind] and source_node!=nodes_dict[depot_ind]: 
                    logger.debug(f"Collision: Agent {agent_id} and Agent {key_points[keypoint]} from node {source_node} at node {keypoint[0]} at time {keypoint[1]}")

                key_points[keypoint] = agent_id
                
                if edge in seen_edges: 
                    logger.debug(f"Edge {edge} already seen for agent {agent_id}")

                seen_edges.add(edge)
                next_step = path[i+1]
                next_source_node = next_step[0] 
                next_target_node = next_step[1] 
                next_time_step = next_step[2]
                 
                if next_time_step < time_step: 
                    logger.debug(f"Agent {agent_id} visited node {next_source_node} at time {next_time_step} before visiting node {source_node} at time {time_step}")

                if next_source_node != target_node:
                    logger.debug(f"Agent {agent_id} visited node {next_source_node} at time {next_time_step} instead of node {target_node} at time {time_step}")

                if next_target_node == target_node: 
                    logger.debug(f"Agent {agent_id} visited node {target_node} at time {time_step} before visiting node {next_target_node} at time {next_time_step}") 

                if source_node not in visit_nodes and source_node != nodes_dict[depot_ind]: 
                    visit_nodes.add(source_node)

                if i != len(path)-2 and target_node == cluster.depot_id:
                    logger.debug(f"Agent {agent_id} visited node {target_node} at time {time_step} before visiting node {cluster.depot_id} at time {time_step+1}")

                if time_step > max_time_steps: 
                    logger.debug(f"Agent {agent_id} visited node {target_node} at time {time_step} which is greater than the maximum time step {max_time_steps}")

            edge_sequence = tuple((step[0], step[1]) for step in path)
            all_paths[agent_id] = edge_sequence
            
            if self.scenario == "individual": 
                if len(visit_nodes) != len(nodes_dict)-1 : 
                    logger.debug(f"Agent {agent_id} visited only {len(visit_nodes)} nodes out of {len(nodes_dict)-1}")

            logger.info(f"Agent {agent_id} | Visited_nodes == > {sorted(visit_nodes)} | Cluster_nodes == > {cluster.nodes_dict.values()} | Bridge Nodes == > {cluster.bridge_nodes}")

             
        agent_ids = list(all_paths.keys()) 
        for i in range(len(agent_ids)): 
            for j in range(i + 1, len(agent_ids)):
                if all_paths[agent_ids[i]] == all_paths[agent_ids[j]]:
                    logger.debug(f"Agents {agent_ids[i]} and {agent_ids[j]} have identical paths!")

        
    def get_travel_time(self, i, j, nodes_dict): 
        return math.ceil(self.travel_cost[nodes_dict[i]-1, nodes_dict[j]-1])
             

    def post_process_interpolation(self, agents_paths_clusters): 

        interpolated_paths = {}

        for key in agents_paths_clusters.keys():
            path = agents_paths_clusters[key]['path']
            processed_path = []
            step = 0
            
            while step < len(path):
                i, j, t = path[step]
                journey = (i, j)

                # Count how many steps this journey spans
                duration = 1
                while (
                    step + duration < len(path) and
                    (path[step + duration][0], path[step + duration][1]) == journey
                ):
                    duration += 1
                
                # Get coordinates for interpolation
                x1, y1 = i  # coords is a dict: node -> (x, y)
                x2, y2 = j
                
                xs = np.linspace(x1, x2, duration + 1)[1:]  # exclude x1 (already included)
                ys = np.linspace(y1, y2, duration + 1)[1:]
                
                for d in range(duration):
                    processed_path.append((xs[d], ys[d], t + d))

                step += duration  # move to next journey

            interpolated_paths[key] = processed_path
        return interpolated_paths
    

    def synchronize_agent_paths(self, paths:Dict ,cluster:Any)->Dict: 
       
        # Calculate recharge steps 
        problem_results = self.problem_results[f'Cluster_{cluster.id}']['agent_results']
        tmp_times = defaultdict(float)
        for agent in paths: 
            tmp_times[agent] = calculate_recharge_steps(self.max_battery, energy_spent=problem_results[agent]['energy'])

        
       
        tmp_time_frame = 1000000000
        tmp_agent = None
        for agent, path in paths.items():
            if len(path) < tmp_time_frame:
                tmp_time_frame = (len(path))
                tmp_agent = agent
        logger.info(f"The shortest path length is {tmp_time_frame} for agent {tmp_agent}")
        cluster.timeframe = list(range(0,tmp_time_frame))
        
        for agent, path in paths.items(): 
            for t in range(self.recharge_time_window): 
                if len(path) < cluster.timeframe[-1]: 
                    time_diff = cluster.timeframe[-1] - len(path) +1
                    idle = [(int(cluster.depot_id),int(cluster.depot_id),path[-1][2]+step) for step in range(time_diff)] 
                    path.extend(idle)
                path.append((int(cluster.depot_id),int(cluster.depot_id),cluster.timeframe[-1]+t))

        return paths 
    

    def flatten_paths_on_time(self, paths): 

        order_of_clusters = list(paths.keys())
        employed_agents = ["Agent_" + str(agent_id) for agent_id in self.agents]
        
        single_agent_paths_for_clusters = {}
        for agent in employed_agents: 
            flattened_path = []
            for cluster_id in order_of_clusters:
                if agent in list(paths[cluster_id].keys()): 
                    
                    if len(flattened_path)==0: 
                        flattened_path.extend(paths[cluster_id][agent])

                    else: 
                        last_time = flattened_path[-1][2] 
                        new_time_steps_for_path = [step[2]+last_time + 1 for step in paths[cluster_id][agent]]
                        new_path = [(step[0], step[1], new_time_steps_for_path[i]) for i, step in enumerate(paths[cluster_id][agent])]
                        flattened_path.extend(new_path)
            
            single_agent_paths_for_clusters[agent] = flattened_path
        
        return single_agent_paths_for_clusters
    

    def get_cluster_coverage(self, cluster:Any):
                
        altitude = 1250/1e3 
        user_height = 12.5/1e3 
        terrain_type = 'urban'
        cluster.get_average_coverage(
            user_points = self.user_points,
            altitude = altitude,
            user_height = user_height,
            terrain_type = terrain_type,
        )


    def create_filename(self, cluster_id:int): 
        id = uuid.uuid4() 
        filename = f'Cluster_{cluster_id}_numerical_results_{id}.csv'
        directory = 'cluster_performance'
        parent_dir = f'{os.getcwd()}/assets/results'
        if not os.path.exists(os.path.join(parent_dir, directory)):
            os.mkdir(os.path.join(parent_dir, directory))

        filename = os.path.join(parent_dir, directory, filename) 

        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write('') 

        return filename
            

