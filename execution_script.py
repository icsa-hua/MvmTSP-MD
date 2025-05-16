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
number_of_agents = 5
max_battery = 1500
constraints = ['const_0', # NOTE: Constraint for many visits
               'const_1', # NOTE: Constraint for entering and leaving depot 
                #  'const_2', # NOTE: Constraint for position on first and last time step
                   'const_3', # NOTE: Constraint for time on depot at first and last time step 
                     'const_4', # NOTE: Constraint for ensuring that only 1 travel for depot in/out is allowed 
                    #    'const_5', # NOTE: Constraint to prohibit depot loop 
                        #  'const_6', # NOTE: Constraint to allow a single travel between nodes 
                        #    'const_7', # NOTE: Constraint to set the busy characteristic on the agent 
                            #  'const_8', # NOTE: Constraint to prevent overlaps with busy 
                              #  'const_9', # NOTE: Constraint to ensure that the time steps in the beginning and end are alligned with the depot decision journey. 
                                #  'const_10', # NOTE: Synchronization of depots for spatial and time variables
                                #    'const_11', # NOTE: Constraint to ensure time progression 
                                    #  'const_12', # NOTE: Synchronization between spatial and time variables 
                                    #    'const_13', # NOTE: Energy constraint 
                                        #  'const_14',  # NOTE: Constraint to ensure that energy won't be negative (failure) during travel
                                        #    'const_15', # NOTE: Constaint to ensure that for a specific time step only a single agent can be on that travel
                                            #  'const_16', # NOTE: Constraint to ensure that agents have unique paths 
                                            #    'const_17', # NOTE: Constraint to ensure that there are no loops in the paths i->j->i
                                                #  'const_18', 
                                                #    'const_19',
                                                    #  'const_20',
                                                      #  'const_21',
                                                        #  'const_22',
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

