from dummy_app.models.simulation_builder import Builder 
from dummy_app.designs.envsim import EnvSim
from dummy_app.designs.mobility import GroundUserGroup
from dummy_app.designs.voronoi_map import Map 
from dummy_app.tools.logger import logger 
from dummy_app.designs.voronoi_map import MapGenerator 
from dummy_app.models.energy_model import DroneEnergyModel
from dummy_app.models.coverage import * 
from geopy.distance import geodesic

from tqdm import tqdm 
import os 
import argparse 
import numpy as np 
from pathlib import Path 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def frame_generator():
    for i in range(TRIALS):
        progress.update(1)
        yield i

PROJECT_DIR = os.getcwd() 
PROJECT_ASSETS = f"{PROJECT_DIR}/assets"
TRIALS = 300 
MAX_BATTERY = 2000 #Wh 
NUMBER_OF_AGENTS = 4 # MIN 4. 
MAX_MEMORY = 2 * 1024 * 1024 * 1024 # 2GB
NUMBER_OF_AREAS = 50 # NOTE: used for Voronoi map generation.
VERTICAL_VELOCITY = 2.78 #m/s 
HORIZONTAL_VELOCITY = 5.55 #m/s

logger.debug(f"Configuration: Asset Directory -> {PROJECT_ASSETS}\n Trials -> {TRIALS}\n Number of Agents -> {NUMBER_OF_AGENTS}\n Max Battery -> {MAX_BATTERY} Wh\n Number of Areas -> {NUMBER_OF_AREAS}\n Vertical Velocity -> {VERTICAL_VELOCITY} m/s\n Horizontal Velocity -> {HORIZONTAL_VELOCITY} m/s")

progress = tqdm(total=TRIALS, desc="Progress")

parser = argparse.ArgumentParser()
parser.add_argument("--generate", action="store_true", help="Generate new Voronoi map and save it to assets.")
parser.add_argument("--show_map", action="store_true", help="Show the generated Voronoi map.")
parser.add_argument("--scenario", type=str, default="energy", help="Scenario to run.")
args = parser.parse_args()

SCENARIO = args.scenario # NOTE: Scenario to run

logger.debug(f"Arguments: Generate -> {args.generate}, Show Map -> {args.show_map}")

# TODO: INCLUDE HOVER ENERGY OR COVERAGE ENERGY. 
constraints = ['const_0', # NOTE: Constraint for many visits.
               'const_1', # NOTE: Constraint for entering and leaving the once.
                 'const_2', # NOTE: Constraint to have dynamic start and end time on the depot for each agent. 
                   'const_3', # NOTE: Constraint to enforce that only a single enter and exit can happen at a depot. 
                     'const_4', # NOTE: Constraint to stop depot looping. 
                       'const_5', # NOTE: Constraint to ensure single travel between nodes (except for bridge nodes) which we can enter and exit more times towards different nodes. 
                        #  'const_6', # NOTE: Constraint for collision avoidance and unique agent per node. This is a constraint that excludes depot and bridge nodes, as the agents can co exist there at the same time. 
                           'const_7', # NOTE: Constraint 
                             'const_8', # NOTE: Constraint to synchronize the time and space decision variables. 
                               'const_9', # NOTE: Constraint to allow a single travel from i to j for time variable. However consider that the problem has to return the paths including duration of travel. 
                            #     #  'const_10', # NOTE: Synchronization of depots for spatial and time variables
                            #     #    'const_11', # NOTE: Constraint to ensure colision avoidance between agents on the time variable, excluding bridge nodes and depots. 
                                     'const_12', # NOTE: Constraint to ensure that the agent has enough energy to travel from i to j.
                                       'const_13', # NOTE: Constraint to ensure that the agent starts from the depot with enough energy.
                                         'const_14',  # NOTE: Constraint to ensure that the agent is idle before departue. 
                                           'const_15', # NOTE: Consrtaint to model time progression through agent business. 
                                             'const_16', # NOTE: Constraint to combine the wait and busy variables ensuring that the agent is either busy or waiting.
                                               'const_17', # NOTE: Constraint to ensure that the travel from depot to a node is synchronized correctly between time and space variables. 
                            #                     #  'const_18', 
                                                #    'const_19',
                                                    #  'const_20', 
                                                       'const_21',
                                                         'const_22',
                                                           'const_23', 
                                                             'const_26'
                                                               'const_27'
                            # #                                     # 'const_29',
                                                                # 'const_30',
                                                                #   'const_31',
                                                                #   'const_32',
                                                                    #   'const_33',
                                                                        # 'const_34',  
                                                                        #   'const_35' 
                                                                           

]
 
config = {
    "regionalization":True, 
    "genetic_algorithm": True, 
    "individual_solution": False, 
    "constraints":constraints,
}

problem = Builder(config, TRIALS, scenario=SCENARIO)
logger.debug(f"BUilder Configuration: {problem.__dict__}")

mobility_sim = EnvSim() 

if not args.generate: 
    dist_path = f"{PROJECT_ASSETS}/env_settings/distance_cost.csv"
    energy_path = f"{PROJECT_ASSETS}/env_settings/energy_cost.csv"
    areas_path = f"{PROJECT_ASSETS}/env_settings/areas.csv"
    customers_path = f"{PROJECT_ASSETS}/env_settings/customers.csv"
    gues_path = f"{PROJECT_ASSETS}/env_settings/ground_users.csv"

    map = Map(data_path=areas_path, incremental=False)
    map.voronoi_tessellation()
    logger.debug(f"Voronoi Map Initialized: {map.__dict__}")

    ground_users = GroundUserGroup(
        mobility_env=mobility_sim, 
        map_obj=map, 
        alpha=0.85, 
        mean_velocity=2.0, 
        sigma=0.5
    )
    logger.debug(f"Ground Users Group Initialized: {ground_users.__dict__}")

    ground_users.load_users_from_csv(
        data_path=gues_path,
        customers_path=customers_path
    )
    logger.debug(f"Ground Users Loaded: {len(ground_users.group)} users")

    data = problem.preprocess(
        distances_path=dist_path, 
        energies_path=energy_path, 
        nodes_path=areas_path, 
        agents=NUMBER_OF_AGENTS, 
        customers_path=customers_path,
        ground_users=ground_users.group,
        max_battery=MAX_BATTERY
    )

    logger.debug(f"Preprocessed Data: {data}")

    if map.vor_map is None:
        raise ValueError("Voronoi map is not initialized. Ensure `voronoi_tessellation` is called successfully.")

    mobility_sim.fig, mobility_sim.ax = ground_users.plot_users(map.vor_map)

    try: 
        ani = FuncAnimation(
            mobility_sim.fig,
            mobility_sim.simulations,
            frames=frame_generator(),
            fargs=(problem, ground_users, map.vor_map, data, TRIALS),
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

else: 
    map_generator = MapGenerator(
        num_areas = NUMBER_OF_AREAS,
        users_per_area = NUMBER_OF_AGENTS, 
        low_lat=52.55, 
        high_lat=52.6,
        low_long=13.55,
        high_long=13.6,
        seed=42
    )
    logger.debug(f"Map Generator Initialized: {map_generator.__dict__}")

    map_generator.generate_points()
    logger.debug(f"Generated Points: {map_generator.points}")

    regions, centroids, user_points = map_generator.voronoi_polygons() 
    logger.debug(f"Voronoi Polygons: Regions -> {len(regions)}, Centroids -> {len(centroids)}, User Points -> {len(user_points)}")

    distance_matrix = map_generator.calculate_centroid_distance(centroids)
    all_user_points = [point for points in user_points.values() for point in points]

    # if args.show_map: 
        # map_generator.plot_map(regions, centroids, all_user_points)
        # input("Press Enter to continue...")
        # map_generator.plot_map_3D(regions, centroids, user_points)

    depots = map_generator.get_central_depots(sites=centroids)
    logger.debug(f"Depots from generated data: {depots}")

    ground_users = GroundUserGroup(
        mobility_env=mobility_sim, 
        map_obj=map_generator, 
        alpha=0.85, 
        mean_velocity=2.0, 
        sigma=0.5
    )
    logger.debug(f"Ground Users Group Initialized: {ground_users.__dict__}")

    ground_users.get_generated_users(user_points=user_points)
    logger.debug(f"Ground Users Loaded: {len(ground_users.group)} users")    
    
    if map_generator.vor_map is None:
        raise ValueError("Voronoi map is not initialized. Ensure `voronoi_tessellation` is called successfully.")

    if args.show_map: 
        mobility_sim.fig, mobility_sim.ax = ground_users.plot_generated_users(
            map_generator,
            regions=regions, 
            centroids=centroids,
            user_points=all_user_points
        )

    else:
        mobility_sim.fig, mobility_sim.ax = ground_users.plot_users(map_generator.vor_map)


    data = problem.preprocess_generated_data(
        distance_matrix=distance_matrix, 
        regions=regions,
        centroids=centroids,
        depots=depots,
        user_points=user_points,
        agents=NUMBER_OF_AGENTS,
        v_hor=HORIZONTAL_VELOCITY, 
        v_ver=VERTICAL_VELOCITY,
        max_battery=MAX_BATTERY
    )
    logger.debug(f"Preprocessed Data: {data}")  
    
    try: 
        ani = FuncAnimation(
            mobility_sim.fig,
            mobility_sim.simulations,
            frames=frame_generator(),
            fargs=(problem, ground_users, map_generator.vor_map, data, TRIALS),
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

