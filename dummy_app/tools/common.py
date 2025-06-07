import gc
import os 
import sys 
import joblib
import pandas as pd 
import numpy as np 
import networkx as nx 
import logging 
from collections import defaultdict
import matplotlib.pyplot as plt 


from typing import Any, List, Dict, Union, Tuple
from dummy_app.models.central_hubs import CentralHub
from dummy_app.tools.graphs import is_eulerian_digraph
from dummy_app.models.energy_model import DroneEnergyModel

def deallocate_memory(variable:Any)->None:
    del variable 
    gc.collect() 


def extract_context_for_cluster(cluster:pd.DataFrame, columns:List[List[str]], column_names:List[str]) -> Dict: 
    extraction = {k:cluster[v].to_numpy() for k,v in zip(column_names, columns)}
    return extraction


def process_extraction(problem_builder:Any, extraction:Dict[str,Union[List[str],np.ndarray]], depot:int, employed_agents:List[int]): 

    try: 
        area_ids = np.array(extraction['area_ids']).squeeze()
        dists = extraction['dists']
        ees = extraction['ees']
        travel_times = extraction['travel_times']
    except KeyError as ke: 
        raise ValueError(f"KeyError: {ke}")

    cost_d = dict(zip(area_ids, dists))
    cost_e = dict(zip(area_ids, ees))
    cost_t = dict(zip(area_ids, travel_times))

    # Initialize structures 
    initial_population = {} 
    nodes_dict = {i: int(node) for i, node in enumerate(area_ids)}
    
    cost_bundle = {'distance':cost_d, 'energy':cost_e,'travel_time':cost_t}

    nodes_for_graph = nodes_dict.copy() 
    graph = create_model_graph(
        cost=cost_bundle, 
        nodes=nodes_dict,
        weights=get_weights() 
    )

    try:
        is_eulerian_digraph(graph)
        if not is_eulerian_digraph(graph): 
            graph = nx.eulerian_circuit(graph)
    except:
        raise ValueError("Graph is not eulerian")


    hub = CentralHub()
    bridge_nodes = hub.get_bridge_nodes(
        graph=graph, 
        cluster_nodes = list(nodes_dict.keys()), 
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
    
    if hasattr(problem_builder, 'enable_ga') and problem_builder.enable_ga: 
        for agent in employed_agents: 
            solution_path, solution_cost = problem_builder.call_genetic_algorithm(
                nodes_dict=nodes_dict, 
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


def create_model_graph(cost:Dict[str,Dict[int,np.ndarray]], nodes:Dict[int,int], weights): 
    graph = nx.DiGraph()
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
    for agent in paths.keys(): 
        duration = len(paths[agent])
        last_time_step = paths[agent][-1][2]
        if duration > last_time_step:
            duration = last_time_step
        agent_times.append(duration) 

    session_duration = max(agent_times)
    return session_duration


def calculate_totals_from_paths(
    results
) -> Tuple:
    """
    Aggregate total distance, energy, and time across all agents' paths.
    """
    total_distance = 0.0
    total_energy = 0.0
    total_time = 0.0
    for agent in results:
        total_distance += results[agent]['distance']
        total_energy += results[agent]['energy']
        total_time += results[agent]['time']

    return total_distance, total_energy, total_time



def extract_per_agent_metrics(
    paths: Any,
    distance_costs: np.ndarray,
    energy_costs: np.ndarray,
    time_costs: np.ndarray,
    coverage_energy:float
) -> Dict[str,Dict[str, float]]:
    """
    Return individual distance, energy, and time for each agent's path.
    """
    results = defaultdict(dict)
    
    for agent in paths:
        visited_nodes = []
        dist = 0.0 
        energy = 0.0 
        duration = 0.0 
        for i, j, _ in paths[agent]:
            if (i,j) in visited_nodes:
                duration += 1
                continue 
            if i == j : 
                energy  += coverage_energy 
                dist += 0 
                duration += 1
                visited_nodes.append((i,j))
                continue
            dist += distance_costs[i, j]
            energy += energy_costs[i, j] 
            duration += time_costs[i, j]
            visited_nodes.append((i,j))
        results[agent] = {'distance': dist, 'energy': energy, 'time': duration}

    return results


def load_scalers(): 
    scaler_path = f"{os.getcwd()}/assets/scalers"
    scalers = defaultdict()

    for scaler_name in os.listdir(scaler_path): 
        if not scaler_name.endswith('.pkl'):continue
        filepath = os.path.join(scaler_path, scaler_name)
        scaler = joblib.load(filepath)
        name = scaler_name.split('.')[0]
        name = name.split('_')[1:] 
        name = "_".join(name)
        scalers[name] = scaler 

    return scalers 


def denormalize_cost(scalers:defaultdict, cost:dict):
    for cost_metric in cost.keys(): 
        if cost_metric == 'travel_time': 
            cost_metric = cost_metric.split('_')[-1]
        for scaler in scalers.keys(): 
            if cost_metric in scaler: 
                cost[cost_metric] = scalers[scaler].inverse_transform(cost[cost_metric])

    return cost             

def load_generated_data(): 
    cost = defaultdict()
    data_path = f"{os.getcwd()}/assets/data"
    for metric in os.listdir(data_path): 
        name = metric.split('.')[0]

        df = pd.read_csv(f'{data_path}/{metric}')
        cost[name] = df.values.astype(np.float32)
    
    return cost



def calculate_recharge_steps(max_battery:float, energy_spent:float):  

    energy_deficit = max_battery - energy_spent 
    energy_model = DroneEnergyModel(max_battery=max_battery) 
    time_steps = 0 
    while energy_deficit < max_battery: 
        recharge_energy = energy_model.recover_energy()
        energy_deficit +=  recharge_energy
        time_steps += energy_model.dt 

    time_steps  /= 60 # back into mins discrete  
    return time_steps


def draw_circular_graph(G:nx.DiGraph): 
    colors_top_10=['tab:orange','tab:blue','tab:green','lightsteelblue']
    #Draw graph
    pos= nx.circular_layout(G)
    nx.draw(G,pos,with_labels=True)

    #Setting up legend
    labels=['Top 10 deg cent','Top 10 bet cent','Top 10 deg and bet cent','no top 10']
    for i in range(len(labels)):
        plt.scatter([],[],label=labels[i],color=colors_top_10[i])
    plt.legend(loc='center')
    plt.show()