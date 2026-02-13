from dummy_app.models.simulation_builder import Builder 
from dummy_app.designs.envsim import EnvSim
from dummy_app.designs.mobility import GroundUserGroup
from dummy_app.designs.voronoi_map import Map 
from dummy_app.tools.logger import logger 
from dummy_app.designs.voronoi_map import MapGenerator 
from dummy_app.models.energy_model import DroneEnergyModel
from dummy_app.models.coverage import * 
from dummy_app.tools.common import deallocate_memory

import os 
import sys
import uuid 
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm 
from matplotlib.animation import FuncAnimation


"""
TODO: 
1. Extract the duration included time paths for each agent. -- idle 
2. Try mninimizing idleness for agents in the individual scenario 
3. Improve visualization.   
4. OUTAGE/COVERAGE PROBABILITY - done
"""

def frame_generator():
    for i in range(TRIALS):
        progress.update(1)
        yield i


# Simulation Environment Configuration 
PROJECT_DIR = os.getcwd() 
PROJECT_ASSETS = f"{PROJECT_DIR}/assets"
TRIALS = 2
MAX_BATTERY = 355.2 #Wh 
NUMBER_OF_AGENTS = 3 # MIN 2. 
MAX_MEMORY = 2 * 1024 * 1024 * 1024 # 2GB
NUMBER_OF_AREAS = 21 # NOTE: used for Voronoi map generation.
NUMBER_OF_USERS = 1
VERTICAL_VELOCITY = 2.78 #m/s 
HORIZONTAL_VELOCITY = 15.56 #m/s
LATITUDE_ATHENS = 37.961322948559
LONGITUDE_ATHENS = 23.708232317542667
LOW_BOUND = 75 #Considered in meters 
HIGH_BOUND = 120 #Considered in meters
ALTITUDE = 1250 # Optimal Coverage Altitude 
MAX_COVERAGE_TIME = 3

scenario_choices = ['cooperative', 'individual']
objective_choices = ['energy', 'coverage'] #  , 'sum_of_times','pareto']
env_choices = ['urban', 'rural', 'forest', 'mountain']
stage_options = [1,2,3]
agents_choices = [3,4,5,6,7,8,9,10]

# Progress bar 
progress = tqdm(total=TRIALS, desc="Progress")

# User arguments 
parser = argparse.ArgumentParser()
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
parser.add_argument("--stage_solution", type=int, default=1, help="What objective stage architecture to use.")
parser.add_argument("--priority", type=str, default="yes", help="Use prioritization")
parser.add_argument("--validate", action="store_true", help="Validate the solution.")
parser.add_argument("--trials", type=int, default=TRIALS, help="Number of trials to run.")
args = parser.parse_args()

TRIALS = args.trials
NUMBER_OF_AGENTS = args.num_agents
NUMBER_OF_USERS = args.num_users
MAX_BATTERY = args.max_battery
MAX_COVERAGE_TIME = args.max_coverage_time
NUMBER_OF_AREAS = args.num_areas
logger.debug(f"Configuration: Asset Directory -> {PROJECT_ASSETS}\n Trials -> {TRIALS}\n Number of Agents -> {NUMBER_OF_AGENTS}\n Max Battery -> {MAX_BATTERY} Wh\n Number of Areas -> {NUMBER_OF_AREAS}\n Vertical Velocity -> {VERTICAL_VELOCITY} m/s\n Horizontal Velocity -> {HORIZONTAL_VELOCITY} m/s")

if args.scenario not in scenario_choices: 
    logger.error(f"Invalid scenario choice. Please choose from: {scenario_choices}")
    sys.exit(1)

if args.objective not in objective_choices:
    logger.error(f"Invalid objective choice. Please choose from: {objective_choices}")
    sys.exit(1)

if args.env not in env_choices: 
    logger.error(f"Invalid environment choice. Please choose from: {env_choices}")
    sys.exit(1)

if args.stage_solution not in stage_options:
    logger.error(f"Invalid stage solution choice. Please choose from: {stage_options}")
    sys.exit(1)


if args.num_agents not in agents_choices:
    logger.error(f"Invalid number of agents choice. Please choose from: {agents_choices}")
    sys.exit(1)

# if args.stage_solution == 2 and args.scenario == 'cooperative': 
#     logger.error(f"Stage 2 is only available for individual scenarios.")
#     exit(1)
   
config = {
    "genetic_algorithm": True if args.enable_ga=='yes' else False, 
    "env_type": args.env, 
    "max_battery":args.max_battery, 
    "max_coverage_time":args.max_coverage_time,
    "scenario":args.scenario, 
    "enable_ga":args.enable_ga,
    "objective_function":args.objective,
    "stage_solution":args.stage_solution, 
    "priority":args.priority,
    "validate":args.validate,
    "NUMBER_OF_AGENTS":NUMBER_OF_AGENTS,
    "NUMBER_OF_USERS":NUMBER_OF_USERS,
    "NUMBER_OF_AREAS":NUMBER_OF_AREAS,
    "altitude": ALTITUDE
}

# Create Builder -> Holds variables and functions to create the combinatorial problem. 
problem = Builder(config, TRIALS)

# Create Simulation environment to simulate mobility for users and agents
mobility_sim = EnvSim(trials=TRIALS) 

# Generate Map Generator Object 
map_generator = MapGenerator(
    num_areas = NUMBER_OF_AREAS,
    users_per_area = NUMBER_OF_USERS, 
    lon=LONGITUDE_ATHENS, 
    lat=LATITUDE_ATHENS,
    low = LOW_BOUND,
    high = HIGH_BOUND,
    seed=42 
)

logger.debug(f"✅ Map Generator Initialized")

regions, centroids, user_points, depots, distance_matrix, all_users = map_generator.create_environment(show_map=False, show_3d_map=False)

# Generate the GroundUserGroup which handles the ground users collectively
ground_users = GroundUserGroup(
    mobility_env=mobility_sim, 
    map_obj=map_generator, 
    alpha=0.85, 
    mean_velocity=10.0, 
    sigma=0.5
)
logger.debug(f"✅ Ground Users Group Initialized")

# Extract the Ground Users as separate entities with individual velocity and angle
ground_users.get_generated_users(user_points=user_points)
logger.debug(f"✅Ground Users Loaded: {len(ground_users.group)} users")    

if map_generator.vor_map is None:
    raise ValueError("Voronoi map is not initialized. Ensure `voronoi_tessellation` is called successfully.")

all_user_points = [point for points in user_points.values() for point in points]

mobility_sim.fig,mobility_sim.ax = ground_users.plot_users(map_generator.vor_map)
# import pdb;pdb.set_trace()
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

# Deallocate all the non necessary components
# deallocate_memory(map_generator)
# deallocate_memory(regions)
# deallocate_memory(centroids)
# deallocate_memory(user_points)
#
animation_directory = f"{PROJECT_ASSETS}/animations" 
if not os.path.exists(animation_directory): 
    os.makedirs(animation_directory) 

animation_filename = f"{animation_directory}/simulation_output_{uuid.uuid4()}.mp4"
try: 
    # From here the simulation initiates and solves the combinatorial problem and then displays the solution. 
    ani = FuncAnimation(
        mobility_sim.fig,
        mobility_sim.simulations,
        frames=frame_generator(),
        fargs=(problem,
               ground_users, 
               vor_map, 
               distance_matrix, 
               data, 
               regions,
               centroids,
               user_points,
               ALTITUDE,
               TRIALS),

        interval=100,
        blit=False, 
        cache_frame_data=False)
    
    # Save as MP4 (requires ffmpeg)
    ani.save(animation_filename, writer='ffmpeg', fps=10)

except KeyboardInterrupt as kb:
    plt.close(mobility_sim.fig)
    logger.exception(f"KeyboardInterrupt: {kb}")

    sys.exit(1)

except Exception as e:
    plt.close(mobility_sim.fig)
    logger.exception(f"Exception: {e}")
    sys.exit(1)

