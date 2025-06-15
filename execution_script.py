from dummy_app.models.simulation_builder import Builder 
from dummy_app.designs.envsim import EnvSim
from dummy_app.designs.mobility import GroundUserGroup
from dummy_app.designs.voronoi_map import Map 
from dummy_app.tools.logger import logger 
from dummy_app.designs.voronoi_map import MapGenerator 
from dummy_app.models.energy_model import DroneEnergyModel
from dummy_app.models.coverage import * 
from dummy_app.tools.common import deallocate_memory






"""
TODO: 
1. Extract the duration included time paths for each agent. 
2. Try mninimizing idleness for agents in the individual scenario 
3. Improve visualization.   
4. OUTAGE/COVERAGE PROBABILITY - done
"""









from tqdm import tqdm 
import os 
import argparse
import matplotlib.pyplot as plt
# import matplotlib; matplotlib.use('Agg') s
from matplotlib.animation import FuncAnimation

def frame_generator():
    for i in range(TRIALS):
        progress.update(1)
        yield i


# Simulation Environment Configuration 
PROJECT_DIR = os.getcwd() 
PROJECT_ASSETS = f"{PROJECT_DIR}/assets"
TRIALS = 300 
MAX_BATTERY = 1500 #Wh 
NUMBER_OF_AGENTS = 6 # MIN 2. 
MAX_MEMORY = 2 * 1024 * 1024 * 1024 # 2GB
NUMBER_OF_AREAS = 50 # NOTE: used for Voronoi map generation.
NUMBER_OF_USERS = 7
VERTICAL_VELOCITY = 2.78 #m/s 
HORIZONTAL_VELOCITY = 5.55 #m/s
LATITUDE_ATHENS = 37.9795
LONGITUDE_ATHENS = 23.7162
LOW_BOUND = 1500 #Considered in meters 
HIGH_BOUND = 1500 #Considered in meters
ALTITUDE = 1250 # Optimal Coverage Altitude 
MAX_COVERAGE_TIME = 5
logger.debug(f"Configuration: Asset Directory -> {PROJECT_ASSETS}\n Trials -> {TRIALS}\n Number of Agents -> {NUMBER_OF_AGENTS}\n Max Battery -> {MAX_BATTERY} Wh\n Number of Areas -> {NUMBER_OF_AREAS}\n Vertical Velocity -> {VERTICAL_VELOCITY} m/s\n Horizontal Velocity -> {HORIZONTAL_VELOCITY} m/s")

scenario_choices = ['cooperative', 'individual']
objective_choices = ['energy', 'coverage', 'idleness']
env_choices = ['urban', 'rural', 'forest', 'mountain']

# Progress bar 
progress = tqdm(total=TRIALS, desc="Progress")

# User arguments 
parser = argparse.ArgumentParser()
parser.add_argument("--gen_areas", action="store_true", help="Generate new Voronoi map and save it to assets.")
parser.add_argument("--show_map", action="store_true", help="Show the generated Voronoi map.")
parser.add_argument("--scenario", type=str, default="cooperative", help="Scenario to run.")
parser.add_argument("--objective", type=str, default="energy", help="Objective to optimize.")
parser.add_argument("--enable_ga", type=str, default='yes', help="Initialize solver with Genetic Algorithm")
parser.add_argument("--num_agents", type=int, default=NUMBER_OF_AGENTS, help="Number of agents to simulate.")
parser.add_argument("--num_users", type=int, default=NUMBER_OF_USERS, help="Number of users to simulate.")
parser.add_argument("--max_battery", type=int, default=MAX_BATTERY, help="Maximum battery capacity.")
parser.add_argument("--max_coverage_time", type=int, default=MAX_COVERAGE_TIME, help="Maximum coverage time.")
parser.add_argument("--num_areas", type=int, default=NUMBER_OF_AREAS, help="Number of areas to simulate.")
parser.add_argument("--env", type=str, default="urban", help="Environment to simulate.")
args = parser.parse_args()

logger.debug(f"Arguments: Generate -> {args.gen_areas}, Show Map -> {args.show_map}")

NUMBER_OF_AGENTS = args.num_agents
NUMBER_OF_USERS = args.num_users
MAX_BATTERY = args.max_battery
MAX_COVERAGE_TIME = args.max_coverage_time
NUMBER_OF_AREAS = args.num_areas

if args.scenario not in scenario_choices: 
    logger.error(f"Invalid scenario choice. Please choose from: {scenario_choices}")
    exit(1)

if args.objective not in objective_choices:
    logger.error(f"Invalid objective choice. Please choose from: {objective_choices}")
    exit(1)

if args.env not in env_choices: 
    logger.error(f"Invalid environment choice. Please choose from: {env_choices}")
    exit(1)


# Declare which constraints to use 
"""
Const 0 --> Enables Multiple Visits per node in the 
Const 1 --> Only allow a single travel from depot to all nodes and reverse. (Per Agent & all agents)
Const 2 --> Allow dynamic return and static departure based on time
Const 3 --> Single Journey between nodes and depot
Const 4 --> Prohibit Depot Looping 
Const 5 --> Enable Arcs for all nodes 
Const 6 --> Collision Avoidance (Unique time visits)
Const 7 --> Miss Indicator per node 
Const 8 --> Synchronization between spatial and time variable
Const 9 --> Time dependency for travel (consider duration) 
Const 10 --> Account for visits per agent 
Const 11 --> Load Balancing for all agents 
Const 12 --> Energy Management (Including Move & Coverage) 
Const 13 --> Energy at Depot and non-negative 
Const 14 --> Positional dependency for start and finish of time frame 
Const 15 --> Busy and Wait decleration per agent for valid arcs 
Const 16 --> Connect busy and wait activity 
Const 17 --> Dynamic Time enforcement for departure and return (time based)
Const 18 --> Symmetry keeping (equal load balancing)
Const 19 --> Busy constraint (Double enforce) 
Const 20 --> Strict Wait after travel (Double enforce) 
Const 21 --> Penalize repeat nodes (energy objective)
Const 22 --> Flow Conservation Not time expanded 
Const 23 --> No loops in path 
Const 24 --> Minimum visits per agent (dual constraint accounting for single agent travel)
Const 25 --> Target Edge count (the number of arcs enabled by x) 
"""                                                              


# NOTE: Create different pipelines based on scenario choice.  
   
# Config should pass all the information necessary inside the builder. 
# -scenario, 
# -environment type 
# -max_battery 
# -max_coverage_time
config = {
    "genetic_algorithm": True if args.enable_ga=='yes' else False, 
    "env_type": args.env, 
    "max_battery":args.max_battery, 
    "max_coverage_time":args.max_coverage_time,
    "scenario":args.scenario, 
    "enable_ga":args.enable_ga,
    "objective_function":args.objective
}

# Create Builder -> Holds variables and functions to create the combinatorial problem. 
problem = Builder(config, TRIALS)

# Create Simulation environment to simulate mobility for users and agents
mobility_sim = EnvSim(trials=TRIALS) 

# This choice is only possible if there are files pre-crafted 
if not args.gen_areas: 
    dist_path = f"{PROJECT_ASSETS}/env_settings/distance_cost.csv"
    energy_path = f"{PROJECT_ASSETS}/env_settings/energy_cost.csv"
    areas_path = f"{PROJECT_ASSETS}/env_settings/areas.csv"
    customers_path = f"{PROJECT_ASSETS}/env_settings/customers.csv"
    gues_path = f"{PROJECT_ASSETS}/env_settings/ground_users.csv"

    map = Map(data_path=areas_path, incremental=False)
    map.voronoi_tessellation()
    logger.debug(f"✅Voronoi Map Initialized")

    ground_users = GroundUserGroup(
        mobility_env=mobility_sim, 
        map_obj=map, 
        alpha=0.85, 
        mean_velocity=2.0, 
        sigma=0.5
    )
    logger.debug(f"✅Ground Users Group Initialized: {ground_users.__dict__}")

    ground_users.load_users_from_csv(
        data_path=gues_path,
        customers_path=customers_path
    )
    logger.debug(f"✅Ground Users Loaded: {len(ground_users.group)} users")

    data = problem.preprocess(
        distances_path=dist_path, 
        energies_path=energy_path, 
        nodes_path=areas_path, 
        agents=NUMBER_OF_AGENTS, 
        customers_path=customers_path,
        ground_users=ground_users.group,
        max_battery=MAX_BATTERY
    )

    logger.debug(f"✅Preprocessed Data: {data}")

    if map.vor_map is None:
        raise ValueError("Voronoi map is not initialized. Ensure `voronoi_tessellation` is called successfully.")

    mobility_sim.fig, mobility_sim.ax = ground_users.plot_users(map.vor_map)
    
    vor_map = map.vor_map 
    del map
    try: 
        ani = FuncAnimation(
            mobility_sim.fig,
            mobility_sim.simulations,
            frames=frame_generator(),
            fargs=(problem, ground_users, vor_map, data, TRIALS),
            interval=100,
            blit=False, 
            cache_frame_data=False)
        
        plt.show(block=False)

        # while True:
        #     plt.pause(0.001)  # keeps the plot interactive
        #     time.sleep(0.001)

        # Save as MP4 (requires ffmpeg)
        ani.save("simulation_output.mp4", writer='ffmpeg', fps=10)

    except KeyboardInterrupt as kb:
        plt.close(mobility_sim.fig)
        logger.exception(f"KeyboardInterrupt: {kb}")
        exit(1)

    except Exception as e:
        plt.close(mobility_sim.fig)
        logger.exception(f"Exception: {e}")
        exit(1)

else: # Default Choice to Generate all points on the map and on the users. 

    # Generate Map Generator Object 
    map_generator = MapGenerator(
        ax = mobility_sim.ax,
        num_areas = NUMBER_OF_AREAS,
        users_per_area = NUMBER_OF_USERS, 
        lon=LONGITUDE_ATHENS, 
        lat=LATITUDE_ATHENS,
        seed=42, 
    )
    logger.debug(f"✅ Map Generator Initialized")

    regions, centroids, user_points, depots, distance_matrix, all_users = map_generator.create_environment(show_map=False, show_3d_map=False)

    # Generate the GroundUserGroup which handles the ground users collectively
    ground_users = GroundUserGroup(
        mobility_env=mobility_sim, 
        map_obj=map_generator, 
        alpha=0.85, 
        mean_velocity=2.0, 
        sigma=0.5
    )
    logger.debug(f"✅ Ground Users Group Initialized")

    # Extract the Ground Users as separate entities with individual velocity and angle
    ground_users.get_generated_users(user_points=user_points)
    logger.debug(f"✅Ground Users Loaded: {len(ground_users.group)} users")    

    if map_generator.vor_map is None:
        raise ValueError("Voronoi map is not initialized. Ensure `voronoi_tessellation` is called successfully.")
    
    all_user_points = [point for points in user_points.values() for point in points]
    
    if args.show_map: 
        mobility_sim.fig, mobility_sim.ax = ground_users.plot_generated_users(
            map_generator,
            regions=regions, 
            centroids=centroids,
            user_points=all_user_points
        )
    else:
        mobility_sim.fig, mobility_sim.ax = ground_users.plot_users(map_generator.vor_map)
    
    # Preprocess the data based on the map, the energy/coverage model and the ground users. 
    data = problem.preprocess_generated_data(
        distance_matrix=distance_matrix, 
        centroids=centroids,
        depots=depots if not isinstance(depots,list) else np.ndarray(depots),
        num_of_agents=NUMBER_OF_AGENTS,
        v_hor=HORIZONTAL_VELOCITY, 
        v_ver=VERTICAL_VELOCITY,
        altitude=ALTITUDE, 
        coverage_time=MAX_COVERAGE_TIME,
        user_points=user_points,
    )
    logger.debug(f"✅ Preprocessed Data Completed successfully")  

    vor_map = map_generator.vor_map
    deallocate_memory(map_generator)
    deallocate_memory(regions)
    deallocate_memory(centroids)
    deallocate_memory(user_points)

    try: 
        # From here the simulation initiates and solves the combinatorial problem and then displays the solution. 
        ani = FuncAnimation(
            mobility_sim.fig,
            mobility_sim.simulations,
            frames=frame_generator(),
            fargs=(problem, ground_users, vor_map, distance_matrix, data, TRIALS),
            interval=100,
            blit=False, 
            cache_frame_data=False)
        
        plt.show(block=False)

        # while True:
        #     plt.pause(0.001)  # keeps the plot interactive
        #     time.sleep(0.001)

        # Save as MP4 (requires ffmpeg)
        ani.save("simulation_output.mp4", writer='ffmpeg', fps=10)

    except KeyboardInterrupt as kb:
        plt.close(mobility_sim.fig)
        logger.exception(f"KeyboardInterrupt: {kb}")
        exit(1)

    except Exception as e:
        plt.close(mobility_sim.fig)
        logger.exception(f"Exception: {e}")
        exit(1)

