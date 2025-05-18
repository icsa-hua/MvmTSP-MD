import math 
import random
import pandas as pd 
import numpy as np 
import pulp as pl 
import networkx as nx 
from typing import Dict, Tuple, List, Union, Any
from dummy_app.tools.common import deallocate_memory, extract_context_for_cluster, process_extraction, create_model_graph, get_weights
from dummy_app.tools.logger import logger


class Cluster: 

    def __init__(self, cluster:pd.DataFrame, id:int, assignment:List[int], depot_id:int): 
        self.cluster = cluster 
        self.id = id
        self.employed_agents:List[int] = assignment 
        if self.employed_agents is None: 
            logger.error(f"No agents assigned to cluster {self.id}")
        self.max_battery = 1500
        self.nodes_dict = {} 
        self.timeframe = [] 
        self.start_time = 0 
        self.bridge_nodes:List[int] = []
        self.initial_population:Dict[int, Tuple[List[int], float]] = {} 
        self.cue_groups = {} 
        self.depot_id = depot_id
        self.tr_times:Dict[(Tuple[int,int],int)] = {}
        self.cost = {}
        self.R_points = []
        # self.paths =  {(id,agent):[] for agent in self.employed_agents}
        self.problem = pl.LpProblem()



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
            if hasattr(builder, 'get_travel_time'):
                total_time = math.ceil(sum(
                    builder.get_travel_time(i, i+1, best_path)
                    for i in range(len(best_path)-1)
                ))

        if total_time == 0: 
            logger.error(f"Total time is 0 for cluster {self.id}")
            raise ValueError(f"Total time is 0 for cluster {self.id}")
        
        self.timeframe = list(range(0, total_time+1))


    def problem_formulation(self, builder): 

        V_nodes = list(self.nodes_dict.keys())

        # Set the decision variables 
        self.create_problem() 

        # Set the loss function 
        self.set_objective(
            distance=self.cost['distance'],
            energy=self.cost['energy'], 
            time=self.cost['travel_time'],
        )
        if not hasattr(builder, 'get_travel_time'):
            logger.error("Builder does not have get_travel_time method") 
            raise ValueError("Builder does not have get_travel_time method")    
        
        self.tr_times = {(i,j):builder.get_travel_time(i, j, self.nodes_dict) for i in V_nodes for j in V_nodes}
        
        if not hasattr(builder, 'set_constraints_for_multi_agent'):
            logger.error("Builder does not have set_constraints_for_multi_agent method") 
            raise ValueError("Builder does not have set_constraints_for_multi_agent method")
        
        if len(self.employed_agents) > 1: 
            builder.set_constraints_for_multi_agent(self)
        
        else: 
            pass 

        builder.solve_problem(self) 
        builder.create_solution(self)


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
    

    def create_problem(self): 
        V = list(self.nodes_dict.keys())
        self.problem = pl.LpProblem("ContrainedMVMTSP", pl.LpMinimize)
        
        self.x = pl.LpVariable.dicts("x", ((i,j,v) for i in V for j in V for v in self.employed_agents), cat='Binary')
        self.t = pl.LpVariable.dicts("t", ((i, j, v, ts) for i in V for j in V for v in self.employed_agents for ts in self.timeframe), cat='Binary')
        
        self.p = pl.LpVariable.dicts("p", ((v,t) for v in self.employed_agents for t in self.timeframe), cat='Integer')
        self.busy = pl.LpVariable.dicts("busy", ((v,t) for v in self.employed_agents for t in self.timeframe), cat='Binary')
        self.wait = pl.LpVariable.dicts("wait", ((v,t) for v in self.employed_agents for t in self.timeframe), cat="Binary")

        self.e = pl.LpVariable.dicts("e", ((i,v) for i in V for v in self.employed_agents),lowBound=0, upBound=self.max_battery, cat='Continuous')
        
        # TODO: Try it like this but after checking the validity of an integer variable. 
        # self.y = pl.LpVariable.dicts("y", ((i,v) for i in V for v in self.employed_agents),lowBound=0, upBound=1, cat='Binary')
        self.y = pl.LpVariable.dicts("y", ((i,v) for i in V for v in self.employed_agents),lowBound=0, cat='Integer')

    def set_objective(self, distance, energy, time): 
        V_nodes = list(self.nodes_dict.keys())
        penalty = 0.8
        self.problem.setObjective(
            pl.lpSum(
                penalty * self.y[j,v] + 
                distance[self.nodes_dict[i]][self.nodes_dict[j]-1] * self.t[i,j,v,t]
                + energy[self.nodes_dict[i]][self.nodes_dict[j]-1] * self.t[i,j,v,t]
                + time[self.nodes_dict[i]][self.nodes_dict[j]-1] * self.t[i,j,v,t]
                for t in self.timeframe
                for i in V_nodes
                for j in V_nodes
                if i != j 
                for v in self.employed_agents
            )
        )



    





        