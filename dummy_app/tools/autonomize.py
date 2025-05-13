import gc
import sys 
import pandas as pd 
import numpy as np 
import networkx as nx 
from typing import Any, List, Dict
from dummy_app.models.central_hubs import CentralHub
import logging 


def deallocate_memory(variable:Any)->None:
    del variable 
    gc.collect() 


def extract_context_for_cluster(cluster:pd.DataFrame, columns:List[List[str]], column_names:List[str]) -> Dict[str, np.ndarray[str]]: 
    extraction = {k:cluster[v].to_numpy() for k,v in zip(column_names, columns)}
    return extraction


def process_extraction(problem_builder:object, extraction:Dict[str,np.ndarray[str]], depot:int, employed_agents:List[int]): 

    try: 
        area_ids = extraction['area_ids'].astype(int) 
        dists = extraction['dists']
        ees = extraction['ees']
        travel_times = extraction['travel_times']
    except KeyError as ke: 
        raise ValueError(f"KeyError: {ke}")

    cost_d = dict(zip(area_ids, dists))
    cost_e = dict(zip(area_ids, ees))
    cost_t = dict(zip(area_ids, travel_times))

    # Initialize structures 
    initial_population = {agent: () for agent in employed_agents} 
    nodes_dict = {i: int(node) for i, node in enumerate(area_ids)}

    cost_bundle = {
        'distance':cost_d, 
        'energy':cost_e,
        'travel_time':cost_t,
    }

    graph = create_model_graph(
        cost=cost_bundle, 
        nodes=nodes_dict,
        weights=get_weights() 
    )

    hub = CentralHub()
    
    bridge_nodes = hub.get_bridge_nodes(
        graph=graph, 
        cluster_nodes = nodes_dict.keys(), 
        cost_dist=cost_bundle['distance'],
        nodes_dict=nodes_dict,
        n_agents=len(employed_agents)
    )
    
    if not bridge_nodes: 
        raise ValueError("No bridge nodes were found")

    bridge_nodes = [nodes_dict[bridge_nodes[i]] for i in range(len(bridge_nodes))]
    R_points = [] 
    for i in nodes_dict: 
        if nodes_dict[i] in bridge_nodes: 
            allowed_visits = hub.number_allowed_visits[i]
        else: 
            allowed_visits = 1 
        R_points.append(allowed_visits)

    assert len(cost_d) == len(cost_e) == len(cost_t) == len(R_points), \
    "Mismatch between distance, energy, travel_time and R_points dictionary length"

    if problem_builder.enable_ga: 
        for agent in employed_agents: 
            solution_path, solution_cost = problem_builder.call_genetic_algorithm(
                V_nodes=nodes_dict, 
                cost=cost_bundle, 
                depot=depot, 
                verbose=False   
            )
            # print(f"Agent {agent} has solution path: {solution_path} with cost: {solution_cost}")
            initial_population[agent] = (solution_path, solution_cost) 

    return cost_bundle, R_points, bridge_nodes, nodes_dict, initial_population


def jupyter_logger(level=logging.INFO)->logging.StreamHandler: 
    jupyter_handler = logging.StreamHandler(sys.stdout)
    jupyter_handler.setLevel(level)

    jupyter_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ) 
    jupyter_handler.setFormatter(jupyter_formatter)

    return jupyter_handler


def create_model_graph(cost:Dict[str,np.ndarray], nodes:Dict[int,int], weights): 
    graph = nx.Graph()
    for source_node in nodes.values(): 
        for target_node in nodes.values(): 
            if source_node == target_node: continue 
            composite_cost = 0.0 
            # Calculate the composite cost for the edge
            for cost_type in cost.keys(): 
               composite_cost += weights[cost_type] * cost[cost_type][source_node][target_node-1] 

            graph.add_edge(source_node, target_node, cost=composite_cost)

    return graph 

        
def get_weights(): 
    return {
        'distance': 0.4,
        'energy': 0.4,
        'travel_time': 0.2
    }


def get_session_duration(paths): 
    agent_times = [] 
    for cluster in paths: 
        duration = 0 
        for agent in paths[cluster]: 
            duration = len(paths[cluster][agent])
        agent_times.append(duration) 

    session_duration = max(agent_times)
    return session_duration