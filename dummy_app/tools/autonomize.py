import gc
import sys 
import pandas as pd 
import numpy as np 
import networkx as nx 
from typing import Any, List, Dict
import logging 

def deallocate_memory(variable:Any)->None:
    del variable 
    gc.collect() 


def extract_context_for_cluster(cluster:pd.DataFrame, columns:List[List[str]], column_names:List[str]) -> Dict[str, np.ndarray[str]]: 
    extraction = {k:cluster[v].to_numpy() for k,v in zip(column_names, columns)}
    return extraction


def process_extraction(problem_builder:object, extraction:Dict[str,np.ndarray[str]], depot:int): 

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

    R_points = {node: len(problem_builder.employed_agents) for node in area_ids}

    assert len(cost_d) == len(cost_e) == len(cost_t) == len(R_points), \
    "Mismatch between distance, energy, travel_time and R_points dictionary length"

    # Initialize structures 
    problem_builder.initial_population = {agent: () for agent in problem_builder.employed_agents} 
    shortest_paths = {agent:[] for agent in problem_builder.employed_agents}
    nodes_dict = {i: int(node) for i, node in enumerate(area_ids)}

    cost_bundle = {
        'distance':cost_d, 
        'energy':cost_e,
        'travel_time':cost_t,
    }

    best_fit_score = np.inf 
    # best_agent = None 

    if problem_builder.enable_ga: 
        for agent in problem_builder.employed_agents: 
            print("Starting GA for agent: {}".format(agent))
            solution_path, solution_cost = problem_builder.call_genetic_algorithm(
                V_nodes=nodes_dict, 
                cost=cost_bundle, 
                depot=depot, 
                verbose=False   
            )
            print(f"Agent {agent} has solution path: {solution_path} with cost: {solution_cost}")
            problem_builder.initial_population[agent] = (solution_path, solution_cost) 

            if solution_cost < best_fit_score: 
                best_fit_score = solution_cost 
                # best_agent = agent 

    
    return cost_bundle, R_points, nodes_dict, problem_builder.initial_population



def jupyter_logger(level=logging.INFO)->logging.StreamHandler: 
    jupyter_handler = logging.StreamHandler(sys.stdout)
    jupyter_handler.setLevel(level)

    jupyter_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ) 
    jupyter_handler.setFormatter(jupyter_formatter)

    return jupyter_handler




def create_model_graph(cost, nodes): 
    graph = nx.Graph()

    # Check the costo dictionary. 

    for source in nodes: 
        for target in nodes: 
                pass