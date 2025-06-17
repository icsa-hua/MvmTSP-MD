import os
import sys
import math 
import uuid
import random
import pandas as pd 
import numpy as np 
import pulp as pl 
import networkx as nx 

from copy import deepcopy
from collections import defaultdict
from typing import Dict, Tuple, List, Any

from dummy_app.tools.common import get_weights, extract_context_for_cluster, process_extraction, create_model_graph, get_weights
from dummy_app.tools.logger import logger
from dummy_app.models.coverage import coverage_u2c, coverage_probability
from dummy_app.designs.constraint import cooperative_scenario_constraints, individual_scenario_constraints


class Cluster: 

    def __init__(self, cluster:pd.DataFrame, id:int, assignment:List[int], depot_id:int, max_battery): 
        self.cluster = cluster 
        self.id = id
        self.employed_agents:List[int] = assignment 
        self.max_battery = max_battery
        self.nodes_dict = {} 
        self.timeframe = [] 
        self.bridge_nodes:List[int] = []
        self.initial_population:Dict[int, Tuple[List[int], float]] = {} 
        self.depot_id = depot_id
        self.tr_times:Dict[(Tuple[int,int],int)] = {}
        self.cost = {}
        self.virtual_nodes = defaultdict()
        self.problem = pl.LpProblem()
        self.R = defaultdict(float) 
        self.sinr = defaultdict(float)
        self.max_durations = defaultdict(int)
        self.original_nodes_dict = self.nodes_dict 
        self.V_nodes = list() 
        self.NODES = list() 
        

    def get_cluster_content(self, distance, energy, time, column_names )->Dict:


        context = extract_context_for_cluster(
            cluster=self.cluster, 
            columns=[
                distance, 
                energy, 
                time, 
                ['Area_id']
            ], 
            column_names=column_names
        )

        return context 
    

    def prepare_context(self, context:Dict, builder:Any): 

        self.cost, self.virtual_nodes, self.bridge_nodes, self.nodes_dict, raw_population = process_extraction(
            problem_builder=builder, 
            extraction=context,
            depot=self.depot_id, 
            employed_agents=self.employed_agents,
        )
        self.initial_population = {
            k: (list(v[0]), float(v[1])) for k, v in raw_population.items()
        }


    def get_estimated_time_frame(self, builder:Any): 
        total_time = 0 

        if self.initial_population is None: 
            G = create_model_graph(
                cost=self.cost['travel_time'], 
                nodes=self.nodes_dict, 
                weights={'travel_time':1}
            )

            mst = nx.minimum_spanning_tree(G, weight='weight')
            estimated_time = sum(edge[2]['weight'] for edge in mst.edges(data=True))
            total_time = math.ceil(estimated_time)

        else: 
            best_agent = min(self.initial_population.items(), key=lambda item: item[1][1])
            best_path = best_agent[1][0]
            if builder.scenario == 'cooperative': 
                num_travels = int((len(best_path) - 1) /len(self.employed_agents))
            elif builder.scenario == 'individual':
                num_travels = len(best_path) - 1 

            else: num_travels = len(best_path) 

            if hasattr(builder, 'get_travel_time'):
                total_time = math.ceil(sum(
                    self.get_travel_times(i, i+1, best_path, builder)
                    for i in range(len(best_path)-1)
                    )) + num_travels * builder.coverage_time
 
        if total_time == 0: 
            logger.error(f"Total time is 0 for cluster {self.id}")
            raise ValueError(f"Total time is 0 for cluster {self.id}")

        self.timeframe = list(range(0, total_time + 1))


    def problem_formulation(self, builder, scenario:str='cooperative', objective_function:str='energy'): 
        
        V_nodes = list(self.original_nodes_dict.keys())

        # Get duration of each trip (arc) 
        self.tr_times = {(self.original_nodes_dict[i],self.original_nodes_dict[j]):self.get_travel_times(i, j, self.original_nodes_dict, builder) for i in V_nodes for j in V_nodes}

        # Set the decision variables 
        self.create_problem(scenario=scenario, objective_functions=objective_function) 

        # Set the loss function 
        if objective_function == 'energy':
            self.set_energy_objective(
                distance=self.cost['distance'],
                energy=self.cost['energy'], 
                time=self.cost['travel_time'],
            )

        elif objective_function == 'coverage':
            self.set_coverage_objective(energy=self.cost['energy'])

        elif objective_function == 'idleness': 
            self.set_idleness_objective(energy=self.cost['energy'])     

        if not hasattr(builder, 'get_travel_time'):
            logger.error("Builder does not have get_travel_time method") 
            raise ValueError("Builder does not have get_travel_time method")    
        
        employed_agents = ["Agent_" + str(i) for i in self.employed_agents]
        list_of_agents = {name: int(name.split("_")[1]) for name in employed_agents}
        
        if scenario == 'cooperative':
            cooperative_scenario_constraints(cluster=self,builder=builder, V_nodes=self.V_nodes, list_of_agents=list_of_agents)
    
        elif scenario == 'individual':
            individual_scenario_constraints(cluster=self,builder=builder, V_nodes=self.V_nodes, list_of_agents=list_of_agents)

        builder.solve_problem(self) 

        return self.get_results(builder=builder)
    

    def create_problem(self, scenario:str='cooperative', objective_functions:str="energy")->None: 
        
        V = self.V_nodes
        NODES = self.NODES

        self.x = pl.LpVariable.dicts("x", ((i,j,v) for i in V for j in V for v in self.employed_agents), cat='Binary')
    
        self.t = pl.LpVariable.dicts("t", ((i,v) for i in V  for v in self.employed_agents), lowBound=0, cat='Continuous')
        
        # Node visitation per agent
        self.visit = pl.LpVariable.dicts("v",((j,v) for j in NODES for v in self.employed_agents),lowBound=0, upBound=1, cat='Binary')

        # toor ordering and loop avoidance 
        self.p = pl.LpVariable.dicts("p", ((j,v) for j in V for v in self.employed_agents),lowBound=0, upBound=len(V)-1, cat='Integer')
    
        # total number of nodes visited per agent
        self.u = pl.LpVariable.dicts("u", (v for v in self.employed_agents),lowBound=0, cat='Integer')
    
        # Return step 
        self.return_step = pl.LpVariable.dicts("return", ((k) for k in self.employed_agents), lowBound=self.timeframe[0], cat='Continuous')
        
        # self.wait = pl.LpVariable.dicts("wait", ((v,t) for v in self.employed_agents for t in self.timeframe), cat="Binary")

        self.e = pl.LpVariable.dicts("e", ((i,v) for i in V for v in self.employed_agents),lowBound=0, upBound=self.max_battery, cat='Continuous')

        if scenario == 'individual': 
            self.precedes = pl.LpVariable.dicts("precedes", ((j, k1, k2) for j in V for k1 in self.employed_agents for k2 in self.employed_agents if k1 < k2), cat='Binary')


        if objective_functions == 'energy': 
            self.problem = pl.LpProblem(name=f"MVMTSP_Cluster_{self.id}", sense=pl.LpMinimize)
            
        elif objective_functions == 'coverage':
            self.problem = pl.LpProblem(name=f"MVMTSP_Cluster_{self.id}", sense=pl.LpMinimize)

        elif objective_functions == "idleness":
            self.problem = pl.LpProblem(name=f"MVMTSP_Cluster_{self.id}", sense=pl.LpMinimize)
         

    def set_energy_objective(self, distance, energy, time): 
        V_nodes = list(self.nodes_dict.keys())
        weights = get_weights() 
        self.problem.setObjective(
                pl.lpSum(
                        self.x[i,j,v] * energy[self.nodes_dict[i]][self.nodes_dict[j]] * weights['energy'] +
                        self.x[i,j,v] * distance[self.nodes_dict[i]][self.nodes_dict[j]] * weights['distance'] +
                        self.x[i,j,v] * time[self.nodes_dict[i]][self.nodes_dict[j]] * weights['travel_time']
                        for i in V_nodes
                        for j in V_nodes if i != j
                        for v in self.employed_agents
                )   
            )
       

    def set_idleness_objective(self,energy):
        alpha = 0.6
        beta = 0.4

        spatial_cost = pl.lpSum(self.x[i,j,k] * energy[source][target] for i,source in self.nodes_dict.items() for j,target in self.nodes_dict.items() if i != j for k in self.employed_agents)

        timestep_cost = pl.lpSum(self.return_step[k] for k in self.employed_agents)

        self.problem.setObjective(
            alpha * spatial_cost + beta * timestep_cost
        )


    def set_coverage_objective(self,energy)->None:

        # Makespan 
        makespan = pl.LpVariable("makespan", lowBound=0, cat='Continuous')

        for k in self.employed_agents: 
           self.problem += makespan >= self.return_step[k] 

        alpha = 0.6 
        beta = 0.4
        spatial_cost = pl.lpSum(self.x[i,j,k] * energy[source][target] for i,source in self.nodes_dict.items() for j,target in self.nodes_dict.items() if i != j for k in self.employed_agents)

        self.problem.setObjective(
            alpha * spatial_cost + beta * makespan
        )


    def set_up_virtual_nodes_properties(self): 

        reverse_nodes = {v:k for k, v in self.nodes_dict.items()}
        self.original_nodes_dict = deepcopy(self.nodes_dict)
        remove_original_nodes = set(self.virtual_nodes.values()) 

        for node in remove_original_nodes: 
            self.nodes_dict.pop(reverse_nodes[node])

        self.nodes_dict = {i:v for i,(k,v) in enumerate(self.nodes_dict.items())} 
        
        self.V_nodes = list(self.nodes_dict.keys())
        reverse_nodes = {v:k for k, v in self.nodes_dict.items()}

        depot_id = reverse_nodes[self.depot_id]
        self.NODES = self.V_nodes[:depot_id] + self.V_nodes[depot_id+1:]


    def get_average_coverage(self, user_points, altitude, user_height, terrain_type='rural', filename='outuput.csv'):
        
        coverage_probability_filename = filename
        # coverage_probability_filename = f'cov_out_prob_{file_id}.csv'
        coverage_directory = f'{os.getcwd()}/assets/results/coverage_prob'
        
        self.check_results_file(
            name=coverage_probability_filename,
            directory='coverage_prob',
            type='csv',
        )
        
        average_R = defaultdict(float)
        average_sinr = defaultdict(float)
        R = defaultdict(list) 
        sinr = defaultdict(list) 
        user_per_area = defaultdict(int) 
        
        for i in self.nodes_dict.keys():
            
            area = self.nodes_dict[i]
            coords = self.cluster.loc[self.cluster['Area_id'] == area, ['X_coords', 'Y_coords']].values
            
            if area not in user_points: continue
            
            user_per_area[i] = 0 
            
            for user in user_points[area]:
                user_coords = (user.x, user.y)
                horizontal_distance = 0.0                 

                horizontal_distance = np.linalg.norm(np.array(coords) - np.array(user_coords))
                horizontal_distance = horizontal_distance / 1e3 # Convert to km

                height_difference = altitude - user_height
                dist = np.sqrt(horizontal_distance**2 + height_difference**2)

                r_value, sinr_value = coverage_u2c(
                    agent_to_user_dist=dist, 
                    agent_altitude=altitude, 
                    user_altitude=user_height, 
                    agent_pos=coords, 
                    terrain_type=terrain_type
                )
                
                R[i].append(r_value)
                sinr[i].append(sinr_value)
                user_per_area[i] += 1

            # Convert to numpy arrays for easier calculations
            average_R[i] = float(np.mean(R[i])/1e6) # Convert to Mbps
            average_sinr[i] = float(np.mean(sinr[i]))
            logger.info(f"R: {average_R[i] } Mbps, SINR: {average_sinr[i]} dB for area {area}")
        
        savefilename = f'{coverage_directory}/{coverage_probability_filename}'
        
        coverage_probability(self, num_users=user_per_area, savefile_name=savefilename, directory=coverage_directory, snr = average_sinr)
        
        self.R = average_R
        self.sinr = average_sinr


    def get_node_visits(self, builder:Any, node, dc):

        if dc[node] not in builder.visits_per_nodes:
            if dc[node] in self.virtual_nodes:    
                builder.visits_per_nodes[self.virtual_nodes[dc[node]]] = 1
            else:
                builder.visits_per_nodes[dc[node]] = 1
        else: 
            if dc[node] in self.virtual_nodes: 
                builder.visits_per_nodes[self.virtual_nodes[dc[node]]] += 1
            else:
                builder.visits_per_nodes[dc[node]] += 1

             
    def get_results(self,builder:Any): 
        
        logger.debug(f"Cluster Time Frame is {self.timeframe}") 
        if pl.LpStatus[self.problem.status] != 'Optimal': 
            logger.info("❌ Problem is not optimal, returning None...")
            sys.exit(1)
        # Assuming 'model' is your solved PuLP problem and depot_ind is your depot's index
        
        V_nodes = self.V_nodes
        NODES = self.NODES
        agents = self.employed_agents

        reverse_dict = {v:k for k, v in self.nodes_dict.items()}
        depot_ind = reverse_dict[self.depot_id]
        
        # First, find the starting point for each agent
        detailed_log = defaultdict(list)
        dc = self.nodes_dict


        # --- START OF DEBUGGING ---
        logger.debug(f"\n--- DEBUGGING CLUSTER {self.id} ---")
        logger.debug(f"Assigned Depot ID: {self.depot_id}")
        
        # This is the most critical part
        # Rebuild your node sets FROM SCRATCH for this run
        all_node_ids_in_cluster = [node for node in self.original_nodes_dict] # Or however you get the IDs
        
        # Ensure depot is correctly identified and separated
        V_nodes = list(self.nodes_dict.keys())
        reverse_dict = {v: k for k, v in self.nodes_dict.items()}
        depot_ind = reverse_dict[self.depot_id]
        NODES = [n for n in V_nodes if n != depot_ind]

        logger.debug(f"All Node IDs (original_nodes_dict): {all_node_ids_in_cluster}")
        logger.debug(f"All Node Indices (V_nodes): {V_nodes}")
        logger.debug(f"Depot Index for this run: {depot_ind}")
        logger.debug(f"Visitable Node Indices (NODES): {NODES}")
        logger.debug(f"Time frame for paths (cluster_object.timeframe): {self.timeframe}")
        logger.debug("---------------------------------------\n")

        for k in agents:
            start_node = -1 
            for j in NODES:
                if self.x[depot_ind, j, k].varValue > 0.5:
                    # Found the first step of the tour
                    start_node = j
                    break 

            real_start_node = dc[start_node]
            if dc[start_node] in self.virtual_nodes: 
                real_start_node = self.virtual_nodes[dc[start_node]] 

            if start_node != -1:
                # Handle the first leg: Depot -> Start Node 
                departure_from_depot = 0.0 
                arrival_at_start_node = (self.t[start_node,k].varValue) 
            
                # Generate "moving" events for the first leg
                for t_step in range(round(departure_from_depot), round(arrival_at_start_node)):
                    detailed_log[k].append((dc[depot_ind], real_start_node, t_step))

                self.get_node_visits(builder, start_node, dc)

                # Continue with the rest of the path
                current_node = start_node
                while current_node != depot_ind:

                    real_current_node = dc[current_node]
                    if dc[current_node] in self.virtual_nodes:
                        real_current_node = self.virtual_nodes[dc[current_node]]

                    # Find the next node in the path
                    next_node_in_path = -1
                    for next_node in V_nodes:
                        if self.x[current_node, next_node, k].varValue > 0.5 :
                            next_node_in_path = next_node
                            break

                    if next_node_in_path == -1:
                        logger.error(f"Warning: Path broken for agent {k} at node {current_node}.Could not find a path to node {next_node_in_path}.")
                        break

                    real_next_node_in_path = dc[next_node_in_path]
                    if dc[next_node_in_path] in self.virtual_nodes:
                        real_next_node_in_path = self.virtual_nodes[dc[next_node_in_path]]
                    
                    # --- Generate events for the current_node ---
                    arrival_at_current = self.t[current_node, k].varValue
                    departure_from_current = arrival_at_current + builder.coverage_time
                    # --- Generate events for the next_node_in_path ---

                    # Generate "waiting" events at the current node
                    for t_step in range(round(arrival_at_current), round(departure_from_current)):
                        detailed_log[k].append((real_current_node, real_current_node, t_step))

                    # --- Generate events for the travel: current_node -> next_node_in_path ---
                    # Handle the final leg back to the depot
                    start_t_move = round(departure_from_current)
                    if next_node_in_path == depot_ind:
                        arrival_at_next = self.return_step[k].varValue
                        # print(f"Agent {k} is returning to depot")
                        # # Generate "moving" events
                        # for t_step in range(round(departure_from_current), round(arrival_at_depot)):
                        #     detailed_log[k].append((dc[current_node], dc[depot_ind], t_step))
                   
                   # Handle a leg to another non-depot node
                    else:
                        arrival_at_next = self.t[next_node_in_path, k].varValue
                        # Generate "moving" events
                        # for t_step in range(round(departure_from_current), round(arrival_at_next)):
                        #     detailed_log[k].append((dc[current_node], dc[next_node_in_path], t_step))
                    
                    end_t_move = round(arrival_at_next)

                    # print(f"start_t_move: {start_t_move}, end_t_move: {end_t_move}, arrival_at_next: {arrival_at_next}, departure_from_current: {departure_from_current}")
                    if start_t_move >= end_t_move and arrival_at_next >= departure_from_current: 
                        end_t_move = start_t_move + 1 

                    for t_step in range(start_t_move, end_t_move): 
                        detailed_log[k].append((real_current_node, real_next_node_in_path, t_step))
                   
                    self.get_node_visits(builder, next_node_in_path, dc)

                    # Move to the next node for the next loop iteration
                    current_node = next_node_in_path


        # Now print the clean, ordered results
        # --- Now you can print or use the detailed_log ---
        for k, events in detailed_log.items():
            logger.debug(f"\n--- Detailed Event Log for Agent {k} ---")
            # Sort events by timestep just in case of rounding nuances
            events.sort(key=lambda x: x[2]) 
            for event in events:
                if event[0] == event[1]:
                    logger.debug(f"Time {event[2]:>3}: Agent {k} is WAITING at Node {event[0]}")
                else:
                    logger.debug(f"Time {event[2]:>3}: Agent {k} is MOVING from {event[0]} to {event[1]}")
        
        unique_nodes_among_paths = set()
        
        for path in detailed_log: 
            for duble in detailed_log[path]:
                if duble[0] not in unique_nodes_among_paths: 
                    unique_nodes_among_paths.add(duble[0]) 
                if duble[1] not in unique_nodes_among_paths:  
                    unique_nodes_among_paths.add(duble[1])

        logger.debug(f"✅ Solutions created for {len(self.employed_agents)} agents")
        logger.info("--------------------------------------------------------------------------")
        logger.info("✅ Optimal Solution Found!!!!!")
        memory_usage = builder.metrics.get_memory_usage()
        logger.info(f"Memory usage: {memory_usage:.2f} MB")
        builder.num_constraints += len(self.problem.constraints)
        builder.variables_count += len(self.problem.variables())
        logger.info(f"The amount of unique nodes visited COLLECTIVELY is {len(unique_nodes_among_paths)}/{len(self.NODES)}")
        builder.global_nodes_visited += len(unique_nodes_among_paths)
        builder.validate_paths(paths=detailed_log, nodes_dict=self.nodes_dict, cluster=self)
        logger.info("✅ Solutions validated successfully...")
        logger.info("--------------------------------------------------------------------------")
        return detailed_log


    def get_travel_times(self,i,j, nodes, builder): 
         
        source = nodes[i] 
        target = nodes[j]
        if  nodes[i] in self.virtual_nodes:
            source = self.virtual_nodes[nodes[i]]
            
        if nodes[j] in self.virtual_nodes: 
            target = self.virtual_nodes[nodes[j]]

        return math.ceil(builder.travel_cost[source, target])
             

    def check_results_file(self, name, directory, type): 
        parent_dir = os.getcwd() 
        assets_dir = os.path.join(parent_dir, 'assets') 
        results_dir = os.path.join(assets_dir, 'results')
        if not os.path.exists(results_dir): 
            os.mkdir(results_dir)

        dd = os.path.join(results_dir, directory) 
        if not os.path.exists(dd):
            os.mkdir(dd)
        accepted_files = ['csv', 'txt', 'png', 'jpg']
        
        if type not in accepted_files: 
            raise ValueError(f"Invalid file type. Accepted file types are: {', '.join(accepted_files)}")
        
        results_file = os.path.join(results_dir, name + f'.{type}')
        if os.path.exists(results_file):
            os.remove(results_file)


 
                