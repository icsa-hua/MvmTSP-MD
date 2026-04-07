from __future__ import annotations 
from dummy_app.models.genetic_algorithm import GASolution, get_weights
from dummy_app.models.simulation_builder import Builder 
from dummy_app.models.central_hubs import CentralHub
from dummy_app.tools.graphs import is_eulerian_digraph
from dummy_app.models.energy_model import DroneEnergyModel

import os 
import joblib
import pandas as pd 
import numpy as np 
import networkx as nx 

from copy import deepcopy
from typing import Any, List, Dict, Union, Tuple
from collections import defaultdict



def call_builder(config, trials) -> Builder: 
    return Builder(config, trials)


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

    graph = GASolution.create_model_graph(
        cost=cost_bundle, 
        nodes=nodes_dict,
        weights=get_weights(getattr(problem_builder, "objective_weights", None)) 
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

    ga_nodes = nodes_dict.copy() 
    for rem in bridge_nodes: 
        ga_nodes.pop(reverse_nodes[rem])

    assert len(cost_d) == len(cost_e) == len(cost_t) == len(R_points), \
    "Mismatch between distance, energy, travel_time and R_points dictionary length"
    
    if hasattr(problem_builder, 'enable_ga') and problem_builder.enable_ga: 
        for agent in employed_agents: 
            solution_path, solution_cost = problem_builder.call_genetic_algorithm(
                nodes_dict=ga_nodes, 
                cost=cost_bundle, 
                depot=depot, 
                verbose=False,
                generations=getattr(problem_builder, "ga_generations", 100),
            )
            initial_population[agent] = (solution_path, solution_cost) 
    return cost_bundle, virtual_nodes, bridge_nodes, nodes_dict, initial_population

        
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
    area_ids:Any, 
    file_id:Any,  

) -> Dict[str,Dict[str, float]]:
    """
    Return individual metrics for each agent by compressing the solved path log
    into contiguous travel and service segments.
    """

    canonical_costs = {
        "distance": np.asarray(costs["distance"], dtype=np.float32),
        "energy": np.asarray(costs["energy"], dtype=np.float32),
        "travel_time": np.asarray(costs["travel_time"], dtype=np.float32),
    }
    cost_bundle = {
        metric: dict(zip(area_ids, matrix))
        for metric, matrix in canonical_costs.items()
    }
    cost_bundle = add_virtual_nodes(cost_bundle=cost_bundle, clones=virtual_nodes, add_epsilon=False)

    results = defaultdict(dict)
    for agent, agent_path in paths.items():
        if not agent_path:
            results[agent] = {
                'distance': 0.0,
                'energy': 0.0,
                'time': 0.0,
                'travel_time': 0.0,
                'modeled_travel_time': 0.0,
                'service_time': 0.0,
                'travel_segments': 0,
                'service_segments': 0,
                'visited_nodes': [],
                'unique_nodes_visited': 0,
            }
            continue

        dist = 0.0
        energy = 0.0
        duration = 0.0
        travel_time = 0.0
        modeled_travel_time = 0.0
        service_time = 0.0
        travel_segments = 0
        service_segments = 0
        visited_nodes = []

        segments = []
        current_source, current_target, current_start = agent_path[0]
        current_length = 1
        for source, target, timestep in agent_path[1:]:
            if (source, target) == (current_source, current_target):
                current_length += 1
                continue
            segments.append((current_source, current_target, current_start, current_length))
            current_source, current_target, current_start = source, target, timestep
            current_length = 1
        segments.append((current_source, current_target, current_start, current_length))

        for source, target, _, segment_length in segments:
            visited_nodes.append(target)
            if source == target:
                service_segments += 1
                service_time += float(segment_length)
                duration += float(segment_length)
                energy += float(coverage_energy) * float(segment_length)
                continue

            travel_segments += 1
            arc_distance = float(cost_bundle['distance'][source][target])
            arc_energy = float(cost_bundle['energy'][source][target])
            modeled_arc_time = float(cost_bundle['travel_time'][source][target])

            dist += arc_distance
            energy += arc_energy
            duration += float(segment_length)
            travel_time += float(segment_length)
            modeled_travel_time += modeled_arc_time

        unique_nodes = sorted(set(visited_nodes))
        results[agent] = {
            'distance': dist,
            'energy': energy,
            'time': duration,
            'travel_time': travel_time,
            'modeled_travel_time': modeled_travel_time,
            'service_time': service_time,
            'travel_segments': travel_segments,
            'service_segments': service_segments,
            'visited_nodes': unique_nodes,
            'unique_nodes_visited': len(unique_nodes),
        }

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


def load_generated_data(data_path): 
    cost = defaultdict()
    
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
    import matplotlib.pyplot as plt
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

    try: 
        for metric, rows in out.items():
            for node, row in rows.items():

                # collect, in order, the distance from this node to each prototype
                try:
                    addon = np.fromiter((row[prototype_lookup[c]] for c in clone_ids), dtype=dtype, count=k)
                    rows[node] = np.concatenate([row, addon])

                except Exception as e: 
                    print(e)
                    print(prototype_lookup)
                    import pdb;pdb.set_trace()

    
        for metric, rows in out.items():
            for j, clone_id in enumerate(clone_ids):
                p = prototype_lookup[clone_id]
                base_row  = rows[p][:n_old]            # original part (length n_old)
                zeros_blk = np.zeros(k, dtype=dtype) # clone-vs-clone block
                if add_epsilon: 
                    zeros_blk = zeros_blk + 1   #epsilon factor. 
                new_row = np.concatenate([base_row, zeros_blk])
                # Find the correct positional index for clone_id in new_row
                clone_pos = n_old + j  # j is the index in clone_ids
                new_row[clone_pos] = 0 
                rows[clone_id] = new_row

    except Exception as e: 
        print(e)
        import pdb;pdb.set_trace() 

    for metric, rows in out.items():
        for node, row in rows.items():
            assert len(row) == n_new, f"{metric}:{node} is wrong length"

    return out


def add_session_time(paths, session_duration): 
    updated_paths = {}
    for agent, path in paths.items(): 
        updated_paths[agent] = [] 
        for triplet in path: 
            cur_pos = triplet[0]
            new_pos = triplet[1]
            time_step = triplet[2] + session_duration
            new_triplet = (cur_pos, new_pos, time_step)
            updated_paths[agent].append(new_triplet)

    return updated_paths
