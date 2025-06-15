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
from copy import deepcopy
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
    
    bridge_nodes = [nodes_dict[bridge_nodes[i]] for i in range(len(bridge_nodes))]
    R_points = [] 
    bridge_nodes_idx = []
    for i in nodes_dict: 
        if nodes_dict[i] in bridge_nodes: 
            allowed_visits = hub.number_allowed_visits[i]
            bridge_nodes_idx.append(1)
        else: 
            allowed_visits = 1 
        R_points.append(allowed_visits)

    reverse_nodes = {v: k for k, v in nodes_dict.items()} 
    virtual_nodes = defaultdict(int)
  
    if not all(rp==1 for rp in R_points) or len(bridge_nodes) > 1:
        # Create virtual nodes inside the current dictionary 
        for node in bridge_nodes: 
            number_of_virtual_nodes = R_points[reverse_nodes[node]] 
            constant_length = len(cost_bundle['distance'][node])
            
            for kk in range(number_of_virtual_nodes):
                virtual_nodes[kk + (constant_length)] = node 

        count = len(nodes_dict)
        cost_bundle = add_virtual_nodes(cost_bundle=cost_bundle, clones=virtual_nodes, add_epsilon=True)
                
        for i in virtual_nodes:
            nodes_dict[count] = i      
            count += 1    


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

    return cost_bundle, virtual_nodes, bridge_nodes, nodes_dict, initial_population


def jupyter_logger(level=logging.INFO)->logging.StreamHandler: 
    jupyter_handler = logging.StreamHandler(sys.stdout)
    jupyter_handler.setLevel(level)

    jupyter_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ) 
    jupyter_handler.setFormatter(jupyter_formatter)

    return jupyter_handler


def create_model_graph(cost:Any, nodes:Dict[int,int], weights): 
    graph = nx.DiGraph()
    for source_node in nodes.values(): 
        for target_node in nodes.values(): 
            if source_node == target_node: continue 
            composite_cost = 0.0 
            # Calculate the composite cost for the edge
            for cost_type in cost.keys(): 
                    composite_cost += weights[cost_type] * cost[cost_type][source_node][target_node] 

            graph.add_edge(source_node, target_node, cost=composite_cost)

    return graph 

        
def get_weights(): 
    return {
        'distance': 0.3,
        'energy': 0.6,
        'travel_time': 0.1
    }


def get_session_duration(paths): 
    agent_times = [] 
    for agent in paths.keys(): 
        last_time_step = paths[agent][-1][2]
        agent_times.append(last_time_step) 

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
    costs: dict,
    coverage_energy:float, 
    virtual_nodes:dict, 
    area_ids:Any  

) -> Dict[str,Dict[str, float]]:
    """
    Return individual distance, energy, and time for each agent's path.
    """
    


    cost_bundle = {}
    for cost_type in costs.keys():
        new_type = ''
        if cost_type.startswith('d'):
            new_type = 'distance'
        elif cost_type.startswith('e'):
            new_type = 'energy'
        elif cost_type.startswith('t'):
            new_type = 'travel_time'
        cost_bundle[new_type] = dict(zip(area_ids, costs[cost_type])) 
            

    cost_bundle = add_virtual_nodes(cost_bundle=cost_bundle, clones=virtual_nodes, add_epsilon=False)

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

            dist += cost_bundle['distance'][i][j]
            energy += cost_bundle['energy'][i][j] 
            duration += cost_bundle['travel_time'][i][j]
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
        recharge_energy = energy_model.recover_energy()/3600
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


def add_virtual_nodes(cost_bundle: dict, clones: dict, add_epsilon:bool=True) -> dict:
    """
    Return a *new* cost_bundle in which every metric has been expanded so that:
      • each original row is k elements longer (one entry per clone);
      • k new rows (one per clone) have been added;
      • distance(u, clone_j) == distance(u, prototype_of_j).
    All arrays remain independent (no accidental views).
    """
    k = len(clones)                       
    clone_ids = list(clones.keys())
    prototypes = list(clones.values())
    prototype_lookup = {c: p for c, p in clones.items()}

    any_metric = next(iter(cost_bundle))
    any_row = next(iter(cost_bundle[any_metric].values()))
    n_old, dtype = len(any_row), any_row.dtype
    n_new = n_old + k

    out = deepcopy(cost_bundle)          


    for metric, rows in out.items():
        for node, row in rows.items():
            # collect, in order, the distance from this node to each prototype
            addon = np.fromiter((row[prototype_lookup[c]] for c in clone_ids),
                                dtype=dtype, count=k)
            rows[node] = np.concatenate([row, addon])

    for metric, rows in out.items():
        for j, clone_id in enumerate(clone_ids):
            p = prototype_lookup[clone_id]
            base_row  = rows[p][:n_old]            # original part (length n_old)
            zeros_blk = np.zeros(k, dtype=dtype) # clone-vs-clone block
            if add_epsilon: 
                zeros_blk = zeros_blk + 1   #epsilon factor. 
            new_row = np.concatenate([base_row, zeros_blk])
            new_row[clone_id] = 0 
            rows[clone_id] = new_row

    for metric, rows in out.items():
        for node, row in rows.items():
            assert len(row) == n_new, f"{metric}:{node} is wrong length"

    return out
