from dummy_app.models.simulation_builder import Builder 
from dummy_app.designs.envsim import EnvSim
from dummy_app.designs.mobility import GroundUserGroup
from dummy_app.designs.voronoi_map import Map 
from dummy_app.tools.logger import logger 

from tqdm import tqdm 
import os 
import time
import numpy as np 
from pathlib import Path 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PROJECT_DIR = os.getcwd() 
PROJECT_ASSETS = f"{PROJECT_DIR}/assets"

trials = 220 
max_memory = 2 * 1024 *1024 *1024
number_of_agents = 4 # MIN 4. 
max_battery = 1500
# TODO: INCLUDE HOVER ENERGY OR COVERAGE ENERGY. 
constraints = ['const_0 ', # NOTE: Constraint for many visits.
               'const_1', # NOTE: Constraint for entering and leaving the once.
                 'const_2', # NOTE: Constraint to have dynamic start and end time on the depot for each agent. 
                   'const_3', # NOTE: Constraint to enforce that only a single enter and exit can happen at a depot. 
                     'const_4', # NOTE: Constraint to stop depot looping. 
                       'const_5', # NOTE: Constraint to ensure single travel between nodes (except for bridge nodes) which we can enter and exit more times towards different nodes. 
                         'const_6', # NOTE: Constraint for collision avoidance and unique agent per node. This is a constraint that excludes depot and bridge nodes, as the agents can co exist there at the same time. 
            #             # #    'const_7', # NOTE: Constraint 
                             'const_8', # NOTE: Constraint to synchronize the time and space decision variables. 
                               'const_9', # NOTE: Constraint to allow a single travel from i to j for time variable. However consider that the problem has to return the paths including duration of travel. 
                                #  'const_10', # NOTE: Synchronization of depots for spatial and time variables
                                   'const_11', # NOTE: Constraint to ensure colision avoidance between agents on the time variable, excluding bridge nodes and depots. 
                                     'const_12', # NOTE: Constraint to ensure that the agent has enough energy to travel from i to j.
                                       'const_13', # NOTE: Constraint to ensure that the agent starts from the depot with enough energy.
                                         'const_14',  # NOTE: Constraint to ensure that the agent is idle before departue. 
                                           'const_15', # NOTE: Consrtaint to model time progression through agent business. 
                                             'const_16', # NOTE: Constraint to combine the wait and busy variables ensuring that the agent is either busy or waiting.
                                               'const_17', # NOTE: Constraint to ensure that the travel from depot to a node is synchronized correctly between time and space variables. 
                                                #  'const_18', 
                                                   'const_19',
                                                     'const_20', #
                                                       'const_21',
                                                        #  'const_22',
                                                           'const_23', 
                                                            #    'const_24', # COnstraint to limit the movement of the agent  to 1 for every time step. 
                                                            #    'const_25',
                                                                #  'const_26',
                                                #                    'const_27',
                                                #                      'const_28',
                                                                       'const_29',
                                                                         'const_30',
                                                                           'const_31',
                                                                             'const_32',
                                                                               'const_33',
                                                                                 'const_34',  
                                                                                #    'const_35'                                                    
]
 
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


problem = Builder(config, trials)

mobility_sim = EnvSim() 

map = Map(data_path=areas_path, incremental=False)
map.voronoi_tessellation()

ground_users = GroundUserGroup(
    mobility_env=mobility_sim, 
    map_obj=map, 
    alpha=0.85, 
    mean_velocity=2.0, 
    sigma=0.5
)


ground_users.load_users(
    data_path=gues_path,
    customers_path=customers_path
)

data = problem.preprocess(
    distances_path=dist_path, 
    energies_path=energy_path, 
    nodes_path=areas_path, 
    agents=number_of_agents, 
    customers_path=customers_path,
    ground_users=gues_path,
    max_battery=max_battery
)


# mobility_sim.simulations(
#     constructor=problem,
#     cues=ground_users, 
#     map=map.vor_map,
#     data=data, 
#     trials=trials
# )
# Create tqdm iterator
progress = tqdm(total=trials, desc="Progress")

def frame_generator():
    for i in range(trials):
        progress.update(1)
        yield i
    # raise StopIteration



if map.vor_map is None:
    raise ValueError("Voronoi map is not initialized. Ensure `voronoi_tessellation` is called successfully.")

mobility_sim.fig, mobility_sim.ax = ground_users.plot_users(map.vor_map)
# mobility_sim.simulations(
#     frame=None,
#     constructor=problem,
#     cues=ground_users,
#     data=data,
#     trials=trials,
#     map=map.vor_map
# ) 
try: 
    ani = FuncAnimation(
        mobility_sim.fig,
        mobility_sim.simulations,
        frames=frame_generator(),
        fargs=(problem, ground_users, map.vor_map, data, trials),
        interval=100,
        blit=False, 
        cache_frame_data=False)
    
    plt.show(block=False)

    # while True:
    #     plt.pause(0.001)  # keeps the plot interactive
    #     time.sleep(0.001)

    # Save as MP4 (requires ffmpeg)
    ani.save("simulation_output.mp4", writer='ffmpeg', fps=10)
    logger.info(f"The constraint use for this problem: {constraints}")

except KeyboardInterrupt as kb:
    plt.close(mobility_sim.fig)
    logger.exception(f"KeyboardInterrupt: {kb}")
    exit(1)

except Exception as e:
    plt.close(mobility_sim.fig)
    logger.exception(f"Exception: {e}")
    exit(1)

