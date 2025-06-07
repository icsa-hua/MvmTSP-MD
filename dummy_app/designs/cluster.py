import math 
import random
import pandas as pd 
import numpy as np 
import pulp as pl 
import networkx as nx 
from collections import defaultdict
from typing import Dict, Tuple, List, Any
from geopy.distance import geodesic
from dummy_app.tools.common import deallocate_memory, extract_context_for_cluster, process_extraction, create_model_graph, get_weights
from dummy_app.tools.logger import logger
from dummy_app.models.coverage import coverage_u2c


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
        self.R_points = []
        self.problem = pl.LpProblem()
        self.R = defaultdict(float) 
        self.sinr = defaultdict(float)


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
        self.cost, self.R_points, self.bridge_nodes, self.nodes_dict, raw_population = process_extraction(
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
                    builder.get_travel_time(i, i+1, best_path)
                    for i in range(len(best_path)-1)
                )) + num_travels 

            # 10 is added to each stop to denote the coverage time spend on each area.  

        if total_time == 0: 
            logger.error(f"Total time is 0 for cluster {self.id}")
            raise ValueError(f"Total time is 0 for cluster {self.id}")

        self.timeframe = list(range(0, total_time + 1))


    def problem_formulation(self, builder, scenario:str='energy'): 

        V_nodes = list(self.nodes_dict.keys())

        # Set the decision variables 
        self.create_problem(scenario=scenario) 
        self.tr_times = {(i,j):builder.get_travel_time(i, j, self.nodes_dict) for i in V_nodes for j in V_nodes}

        # Set the loss function 
        if scenario == 'energy':
            self.set_objective(
                distance=self.cost['distance'],
                energy=self.cost['energy'], 
                time=self.cost['travel_time'],
                wait_energy=builder.average_coverage_energy
            )
            logger.info(f"Objective function set for energy scenario in cluster {self.id}")
        elif scenario == 'coverage':
            self.set_coverage_objective()
            logger.info(f"Objective function set for coverage scenario in cluster {self.id}")


        if not hasattr(builder, 'get_travel_time'):
            logger.error("Builder does not have get_travel_time method") 
            raise ValueError("Builder does not have get_travel_time method")    
        
        
        if not hasattr(builder, 'set_constraints_for_multi_agent'):
            logger.error("Builder does not have set_constraints_for_multi_agent method") 
            raise ValueError("Builder does not have set_constraints_for_multi_agent method")
        
        # if len(self.employed_agents) > 1: 
        #     builder.set_constraints_for_multi_agent(self)
        
        # else: 
        #     pass 
        builder.set_constraints_for_multi_agent(self)

        # import pdb;pdb.set_trace()
        builder.solve_problem(self) 
        return builder.create_solution(self)


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
    

    def create_problem(self, scenario:str='energy')->None: 
        V = list(self.nodes_dict.keys())
        
        if scenario == 'energy': 
            self.problem = pl.LpProblem(name="ContrainedMVMTSP", sense=pl.LpMinimize)
        elif scenario == 'coverage':
            self.problem = pl.LpProblem(name="ContrainedMVMTSP", sense=pl.LpMinimize)
        
        self.x = pl.LpVariable.dicts("x", ((i,j,v) for i in V for j in V for v in self.employed_agents), cat='Binary')
        self.t = pl.LpVariable.dicts("t", ((i, j, v, ts) for i in V for j in V for v in self.employed_agents for ts in self.timeframe), cat='Binary')
        
        self.p = pl.LpVariable.dicts("p", ((v,t) for v in self.employed_agents for t in self.timeframe), cat='Integer')
        self.busy = pl.LpVariable.dicts("busy", ((v,t) for v in self.employed_agents for t in self.timeframe), cat='Binary')
        self.wait = pl.LpVariable.dicts("wait", ((v,t) for v in self.employed_agents for t in self.timeframe), cat="Binary")

        self.e = pl.LpVariable.dicts("e", ((i,v) for i in V for v in self.employed_agents),lowBound=0, upBound=self.max_battery, cat='Continuous')
        
        # TODO: Try it like this but after checking the validity of an integer variable. 
        # self.y = pl.LpVariable.dicts("y", ((i,v) for i in V for v in self.employed_agents),lowBound=0, upBound=1, cat='Binary')
        self.y = pl.LpVariable.dicts("y", ((i,v) for i in V for v in self.employed_agents), lowBound=0, cat='Integer')

        self.active_agents = pl.LpVariable.dicts("active_agents", (v for v in self.employed_agents), lowBound=0, upBound=1, cat='Binary')

        self.return_step = pl.LpVariable.dicts("return_step", (v for v in self.employed_agents), lowBound=0, upBound=self.timeframe[-1], cat='Integer')

        self.visit = pl.LpVariable.dicts("visit", ((i,v) for i in V for v in self.employed_agents), lowBound=0, upBound=1, cat='Binary')

        self.visit_miss = pl.LpVariable.dicts("visit_miss", ((i,v) for i in V for v in self.employed_agents), lowBound=0, upBound=1, cat='Binary')

        self.T_MAX = pl.LpVariable(name='T_MAX', lowBound=0, cat='Continuous')


    def set_objective(self, distance, energy, time, wait_energy): 
        V_nodes = list(self.nodes_dict.keys())
        penalty = 100
        alpha = 50
        import pdb;pdb.set_trace()
        self.problem.setObjective(
            pl.lpSum(
                alpha * self.y[j,v] + 
                # alpha * self.y[j,v] * self.R_points[j] +
                penalty * self.visit_miss[j, v] 
                for j in V_nodes
                for v in self.employed_agents
            ) + 
            pl.lpSum(
                    self.x[i,j,v] * energy[self.nodes_dict[i]][self.nodes_dict[j]] +
                    self.x[i,j,v] * distance[self.nodes_dict[i]][self.nodes_dict[j]]
                    for i in V_nodes
                    for j in V_nodes if i != j
                    for t in self.timeframe
                    for v in self.employed_agents
            ) + pl.lpSum(
                    self.wait[v, t] * wait_energy
                    for v in self.employed_agents
                    for t in self.timeframe
            )
        )
        


    def set_coverage_objective(self)->None:
        V_nodes = list(self.nodes_dict.keys()) 
        ALPHA = 1.0 
        BETA = 0.001 

        self.problem.setObjective(
            ALPHA * self.T_MAX - BETA * pl.lpSum(self.R[i] * self.visit[i,v] for i in V_nodes for v in self.employed_agents)
        )
        # self.problem.setObjective(
        #    pl.lpSum(self.R[i] * self.visit[i, v]
        #            for i in V_nodes
        #            for v in self.employed_agents)
        #     )


    def get_average_coverage(self, user_points, altitude, user_height, terrain_type='rural', metric="euclidean"):
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
                if metric == "geodesic":
                    horizontal_distance = geodesic(coords, user_coords).km
                
                elif metric == "euclidean":
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










        