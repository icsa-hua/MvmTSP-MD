from dummy_app.models.builder import MVMTSPBuilder 
from dummy_app.designs.envsim import EnvSim
from dummy_app.designs.mobility import GroundUserGroup
from dummy_app.designs.voronoi_map import Map 
from dummy_app.tools.logger import logger 
from dummy_app.tools.autonomize import extract_context_for_cluster, process_extraction

import os 
import numpy as np 
from pathlib import Path 


PROJECT_DIR = os.getcwd() 
PROJECT_ASSETS = f"{PROJECT_DIR}/assets"

max_memory = 2 * 1024 *1024 *1024
number_of_agents = 5
max_battery = 1500
constraints = ['const_0', 'const_1', 'const_2', 'const_3', 'const_4', 'const_5', 'const_6', 'const_7', 'const_8', 'const_9',
               'const_10', 'const_11','const_12','const_13','const_14','const_15','const_16']

config = {
    "regionalization":True, 
    "genetic_algorithm": True, 
    "individual_solution": False, 
    "constraints":constraints,
}

dist_path = f"{PROJECT_ASSETS}/env_settings/distance_cost.csv"
energy_path = f"{PROJECT_ASSETS}/env_settings/energy_cost.csv"
areas_path = f"{PROJECT_ASSETS}/env_settings/areas.csv"
customers_path = f"{PROJECT_ASSETS}/env_settings/customers.csv"
gues_path = f"{PROJECT_ASSETS}/env_settings/ground_users.csv"



problem = MVMTSPBuilder(config)

mobility_sim = EnvSim() 

map = Map(data_path=areas_path, incremental=False)
map.voronoi_tessellation()

ground_users = GroundUserGroup(
    env=mobility_sim.env, 
    map_obj=map, 
    alpha=0.85, 
    mean_velocity=1.0, 
    sigma=0.5
)


ground_users.load_users(
    data_path=gues_path,
    customers_path=customers_path
)

data = problem.preprocess(
    distances_path=dist_path, 
    energies=energy_path, 
    nodes_path=areas_path, 
    agents=number_of_agents, 
    customers_path=customers_path,
    ground_users=gues_path,
    max_battery=max_battery
)

assignments, updated_clusters =problem.run_model(data, ground_users.group)

for (cluster_tuple, agents), cluster in zip(assignments.items(), updated_clusters):
    problem.clustering(cluster=cluster, cluster_id=cluster_tuple[0], assignment=agents, depot_id=cluster_tuple[1])
    












