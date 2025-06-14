import sys
import math 
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
from dummy_app.models.coverage import coverage_u2c
from dummy_app.designs.constraint import cooperative_scenario_constraints, individual_scenario_constraints


class Cluster: 

    def __init__(self, cluster:pd.DataFrame, id:int, assignment:List[int], depot_id:int, max_battery): 
        self.cluster = cluster 
        self.id = id
        self.employed_agents:List[int] = assignment 
        if self.employed_agents is None: 
            logger.error(f"No agents assigned to cluster {self.id}")
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
            num_travels = len(best_path) - 1
            if hasattr(builder, 'get_travel_time'):
                total_time = math.ceil(sum(
                    self.get_travel_times(i, i+1, best_path, builder)
                    for i in range(len(best_path)-1)
                    )) + num_travels * builder.coverage_time
            #     total_time = math.ceil(sum(
            #         builder.get_travel_time(i, i+1, best_path)
            #         for i in range(len(best_path)-1)
            #     )) + num_travels * builder.coverage_time

            # 10 is added to each stop to denote the coverage time spend on each area.  

        if total_time == 0: 
            logger.error(f"Total time is 0 for cluster {self.id}")
            raise ValueError(f"Total time is 0 for cluster {self.id}")

        self.timeframe = list(range(0, total_time + 1))


    def problem_formulation(self, builder, scenario:str='cooperative', objective_function:str='energy'): 
        
        V_nodes = list(self.nodes_dict.keys())

        # Get duration of each trip (arc) 
        self.tr_times = {(self.nodes_dict[i],self.nodes_dict[j]):self.get_travel_times(i, j, self.nodes_dict, builder) for i in V_nodes for j in V_nodes}

        # Set the decision variables 
        self.create_problem(scenario=scenario, objective_functions=objective_function) 

        # Set the loss function 
        if objective_function == 'energy':
            self.set_energy_objective(
                distance=self.cost['distance'],
                energy=self.cost['energy'], 
                time=self.cost['travel_time'],
            )

            logger.info(f"Objective function set for energy scenario in cluster {self.id}")

        elif objective_function == 'coverage':
            self.set_coverage_objective()
            logger.info(f"Objective function set for coverage scenario in cluster {self.id}")

        if not hasattr(builder, 'get_travel_time'):
            logger.error("Builder does not have get_travel_time method") 
            raise ValueError("Builder does not have get_travel_time method")    
        
        if not hasattr(builder, 'set_constraints_for_multi_agent'):
            logger.error("Builder does not have set_constraints_for_multi_agent method") 
            raise ValueError("Builder does not have set_constraints_for_multi_agent method")

        employed_agents = ["Agent_" + str(i) for i in self.employed_agents]
        list_of_agents = {name: int(name.split("_")[1]) for name in employed_agents}
        
        if scenario == 'cooperative':
            cooperative_scenario_constraints(cluster=self,builder=builder, V_nodes=self.V_nodes, list_of_agents=list_of_agents)
    
        elif scenario == 'individual':
            individual_scenario_constraints(cluster=self,builder=builder, V_nodes=self.V_nodes, list_of_agents=list_of_agents)

        builder.solve_problem(self) 

        return self.get_results(builder=builder)


    def get_solution(self): # Test Trial #TODO: Implement this to extract the solution from the MILP problem. 
        reverse_dict = {v:k for k, v in self.nodes_dict.items()}
        employed_agents = ["Agent_" + str(i) for i in self.employed_agents]
        list_of_agents = {name: int(name.split("_")[1]) for name in employed_agents}
        paths =  {agent:[] for agent in employed_agents}
        for name, agent_id in list_of_agents.items():
            node_values = [v for k, v in self.nodes_dict.items()]
            node_values.remove(self.depot_id)
            
            step = self.timeframe[0] 
            next_node = 0 
            current_node = 0 

            while step in self.timeframe: 
                if step == self.timeframe[0]: 
                    
                    paths[name].extend([(self.depot_id,self.depot_id,step)])

                    current_node = self.depot_id
                    next_node = random.choice(node_values) 
                    node_values.remove(next_node)
                    step += 1 
                    continue
                elif step >= self.timeframe[-1]:
                    paths[name].extend([(self.depot_id,self.depot_id,step)])
                    break

                duration = self.tr_times[(reverse_dict[current_node], reverse_dict[next_node])]
                paths[name].extend([(current_node,next_node,t) for t in range(step, step + duration)])
                step += duration 
                current_node = next_node
                if len(node_values) == 0:
                    break
                next_node = random.choice(node_values)
                node_values.remove(next_node)

            time_difference = step - self.timeframe[-1]
            for t in range(0,time_difference):
                paths[name].extend([(current_node, next_node, step + t + 1)]) 

            if paths[name][-1][1] != self.depot_id:
                duration = self.tr_times[(reverse_dict[next_node], reverse_dict[self.depot_id])]

                paths[name].extend([(next_node,self.depot_id,step)])
            logger.debug(f"Agent {agent_id} path: {paths[name]}")

        return paths
    

    def create_problem(self, scenario:str='cooperative', objective_functions:str="energy")->None: 
        
        self.set_up_virtual_nodes_properties()

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

        if objective_functions == 'energy': 
            self.problem = pl.LpProblem(name="ContrainedMVMTSP", sense=pl.LpMinimize)
            
        elif objective_functions == 'coverage':
            self.problem = pl.LpProblem(name="ContrainedMVMTSP", sense=pl.LpMaximize)
    
        if scenario == 'individual': 
            self.precedes = pl.LpVariable.dicts("precedes", ((j, k1, k2) for j in V for k1 in self.employed_agents for k2 in self.employed_agents if k1 < k2), cat='Binary')

            

    def set_energy_objective(self, distance, energy, time): 
        V_nodes = list(self.nodes_dict.keys())
        penalty = 1
        alpha = 0.5
        weights = get_weights() 
        self.problem.setObjective(
            # pl.lpSum(
            #     alpha * self.y[j,v] + 
            #     # alpha * self.y[j,v] * self.R_points[j] +
            #     penalty * self.visit_miss[j, v] 
            #     for j in V_nodes
            #     for v in self.employed_agents
            # ) + 
                pl.lpSum(
                        self.x[i,j,v] * energy[self.nodes_dict[i]][self.nodes_dict[j]] * weights['energy'] +
                        self.x[i,j,v] * distance[self.nodes_dict[i]][self.nodes_dict[j]] * weights['distance'] +
                        self.x[i,j,v] * time[self.nodes_dict[i]][self.nodes_dict[j]] * weights['travel_time']
                        for i in V_nodes
                        for j in V_nodes if i != j
                        for v in self.employed_agents
                )   
            )
       


    def set_coverage_objective(self)->None:

        ALPHA = 0.1
        BETA = 1

        self.problem.setObjective(
            # ALPHA * self.T_MAX - 
            BETA * pl.lpSum(self.R[i] * self.visit[i,v] for i in self.NODES for v in self.employed_agents)
        )
        # self.problem.setObjective(
        #    pl.lpSum(self.R[i] * self.visit[i, v]
        #            for i in V_nodes
        #            for v in self.employed_agents)
        #     )


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


        

    def get_average_coverage(self, user_points, altitude, user_height, terrain_type='rural'):
        # for Area with id 
        average_R = defaultdict(float)
        average_sinr = defaultdict(float)
        R = defaultdict(list) 
        sinr = defaultdict(list) 
        for i in self.nodes_dict.keys():
            area = self.nodes_dict[i]
            coords = self.cluster.loc[self.cluster['Area_id'] == area, ['X_coords', 'Y_coords']].values
            if area not in user_points: continue
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
            # Convert to numpy arrays for easier calculations
            average_R[i] = float(np.mean(R[i])/1e6) # Convert to Mbps
            average_sinr[i] = float(np.mean(sinr[i]))
            logger.info(f"R: {average_R[i] } Mbps, SINR: {average_sinr[i]} dB for area {area}")

        self.R = average_R
        self.sinr = average_sinr

   
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
        solution_path = defaultdict()
        paths = defaultdict()

        for k in agents:
            solution_path[k] = []
            paths[k] = [] 
            step = 0
            for j in NODES:
                if self.x[depot_ind, j, k].varValue == 1:
                    # Found the first step of the tour
                    start_node = j
                    path = [depot_ind, start_node]
                    paths[k].append((depot_ind, start_node,))
                    arrival_info = [
                        f"Agent {k} departs Depot {depot_ind} at t=0",
                        f"Agent {k} arrives at Node {start_node} at t={self.t[start_node, k].varValue:.2f}"
                    ]
                    # Now follow the path until we return to the depot
                    current_node = start_node
                    while current_node != depot_ind:
                        found_next = False
                        for next_node in V_nodes: # V_nodes includes the depot
                            if self.x[current_node, next_node, k].varValue==1:
                                path.append(next_node)
                                if next_node != depot_ind:
                                    arrival_info.append(
                                        f"Agent {k} arrives at Node {next_node} at t={self.t[next_node, k].varValue:.2f}"
                                    )
                                    
                                    paths[k].append((current_node, next_node, int(self.t[next_node, k].varValue))) 
                                else:
                                    # Use the return_step variable for final arrival
                                    arrival_info.append(
                                        f"Agent {k} arrives back at Depot {next_node} at t={self.return_step[k].varValue:.2f}"
                                    )
                                    paths[k].append((current_node, next_node, int(self.return_step[k].varValue))) 

                                current_node = next_node
                                found_next = True
                                break
                        if not found_next:
                            break # Should not happen in a valid tour

                    solution_path[k] = (path, arrival_info)

        # Now print the clean, ordered results
        for k, (path, arrival_info) in solution_path.items():
            logger.info(f"v--- Agent {k} Final Tour ---v")
            logger.info(f"Path: {' -> '.join(map(str, path))}")
            logger.info("Schedule:")
            for step in arrival_info:
                logger.info(f"  {step}")
        
        unique_nodes_among_paths = set()
        
        for path in paths: 
            for duble in paths[path]:
                if duble[0] not in unique_nodes_among_paths: 
                    unique_nodes_among_paths.add(duble[0]) 
                if duble[1] not in unique_nodes_among_paths:  
                    unique_nodes_among_paths.add(duble[1])

        logger.debug(f"✅ Solutions created for {len(self.employed_agents)} agents")
        logger.info("✅ Optimal Solution Found!!!!!")
        memory_usage = builder.metrics.get_memory_usage()
        logger.info(f"Memory usage: {memory_usage:.2f} MB")
        builder.num_constraints += len(self.problem.constraints)
        builder.variables_count += len(self.problem.variables())
        logger.info(f"The amount of unique nodes visited COLLECTIVELY is {len(unique_nodes_among_paths)}/{len(self.nodes_dict.values())}")
        
        builder.validate_paths(paths=paths, nodes_dict=self.nodes_dict, cluster=self)
        logger.debug("✅ Solutions validated successfully...")

        return paths


    def get_travel_times(self,i,j, nodes, builder): 
         
        source = nodes[i] 
        target = nodes[j]
        if  nodes[i] in self.virtual_nodes:
            source = self.virtual_nodes[nodes[i]]
            
        if nodes[j] in self.virtual_nodes: 
            target = self.virtual_nodes[nodes[j]]

        return math.ceil(builder.travel_cost[source, target])
             





                