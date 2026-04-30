from dummy_app.tools.logger import logger 
from dummy_app.models.RL import analyze_dataset, build_instance_specs, compare_baselines, generate_dataset, validate_action_catalog
from dummy_app.designs.envsim import EnvSim
from dummy_app.designs.mobility import GroundUserGroup
from dummy_app.designs.voronoi_map import MapGenerator
from dummy_app.tools.common import call_builder
from program_config import * 

import os
import sys
import uuid 
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm 
from matplotlib.animation import FuncAnimation

_MPL_CONFIG_DIR = "/tmp/mvmtsp-mpl"
_XDG_CACHE_HOME = "/tmp/mvmtsp-xdg-cache"
os.makedirs(_MPL_CONFIG_DIR, exist_ok=True)
os.makedirs(_XDG_CACHE_HOME, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG_DIR)
os.environ.setdefault("XDG_CACHE_HOME", _XDG_CACHE_HOME)

"""
TODO: 
1. Extract the duration included time paths for each agent. -- idle 
2. Try mninimizing idleness for agents in the individual scenario 
3. Improve visualization.   
4. OUTAGE/COVERAGE PROBABILITY - done
"""
progress = None

def frame_generator():
    for i in range(TRIALS):
        if progress is not None:
            progress.update(1)
        yield i


def parse_csv_ints(raw_value):
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_csv_strings(raw_value):
    return [item.strip() for item in raw_value.split(",") if item.strip()]


# Simulation Environment Configuration 

# PROJECT_DIR = os.getcwd() 
# PROJECT_ASSETS = f"{PROJECT_DIR}/assets"
# TRIALS = 2
# MAX_BATTERY = 355.2 #Wh 
# NUMBER_OF_AGENTS = 3 # MIN 2. 
# MAX_MEMORY = 2 * 1024 * 1024 * 1024 # 2GB
# NUMBER_OF_AREAS = 21 # NOTE: used for Voronoi map generation.
# NUMBER_OF_USERS = 1
# VERTICAL_VELOCITY = 2.78 #m/s 
# HORIZONTAL_VELOCITY = 15.56 #m/s
# LATITUDE_ATHENS = 37.961322948559
# LONGITUDE_ATHENS = 23.708232317542667
# LOW_BOUND = 75 #Considered in meters 
# HIGH_BOUND = 120 #Considered in meters
# ALTITUDE = 1250 # Optimal Coverage Altitude 
# MAX_COVERAGE_TIME = 3
#
# scenario_choices = ['cooperative', 'individual']
# objective_choices = ['energy', 'coverage'] #  , 'sum_of_times','pareto']
# env_choices = ['urban', 'rural', 'forest', 'mountain']
# stage_options = [1,2,3]
# agents_choices = [3,4,5,6,7,8,9,10]
#
def main(): 

    # User arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_map", action="store_true", help="Show the generated Voronoi map.")
    parser.add_argument("--scenario", type=str, default=SCENARIO_OPTIONS[0], help="Scenario to run.")
    parser.add_argument("--objective", type=str, default=OBJECTIVE_OPTIONS[0], help="Objective to optimize.")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Optimization runtime/model to execute.")
    parser.add_argument("--enable_ga", type=str, default=ENABLE_GA, help="Initialize solver with Genetic Algorithm")
    parser.add_argument("--num_agents", type=int, default=NUMBER_OF_AGENTS, help="Number of agents to simulate.")
    parser.add_argument("--num_users", type=int, default=NUMBER_OF_USERS, help="Number of users to simulate.")
    parser.add_argument("--max_battery", type=int, default=MAX_BATTERY, help="Maximum battery capacity.")
    parser.add_argument("--max_coverage_time", type=int, default=MAX_COVERAGE_TIME, help="Maximum coverage time.")
    parser.add_argument("--num_areas", type=int, default=NUMBER_OF_AREAS, help="Number of areas to simulate.")
    parser.add_argument("--env", type=str, default=ENVIRONMENT_OPTIONS[0], help="Environment to simulate.")
    parser.add_argument("--stage_solution", type=int, default=STAGE_OPTIONS[0], help="What objective stage architecture to use.")
    parser.add_argument("--priority", type=str, default=PRIORITY, help="Use prioritization")
    parser.add_argument("--validate", action="store_true", help="Validate the solution.")
    parser.add_argument("--trials", type=int, default=TRIALS, help="Number of trials to run.")
    parser.add_argument("--solver_backend", type=str, default=SOLVER_BACKEND, help="MILP solver backend.")
    parser.add_argument("--subtour_strategy", type=str, default=SUBTOUR_STRATEGY, help="Subtour elimination strategy.")
    parser.add_argument("--objective_strategy", type=str, default=OBJECTIVE_STRATEGY, help="Objective orchestration strategy.")
    parser.add_argument("--scenario_constraint_set", type=str, default=SCENARIO_CONSTRAINT_SET, help="Constraint set variant to use.")
    parser.add_argument("--solver_time_limit_seconds", type=int, default=SOLVER_TIME_LIMIT, help="Optional MILP solver time limit in seconds; 0 disables the override.")
    parser.add_argument("--enable_learning", action="store_true", help="Enable contextual bandit configuration selection.")
    parser.add_argument("--learning_alpha", type=float, default=LEARNING_ALPHA, help="Exploration factor for the contextual bandit.")
    parser.add_argument("--workflow", type=str, default=WORKFLOW, help="simulate, validate_actions, dataset, analyze_dataset, compare_baselines")
    parser.add_argument("--dataset_output_dir", type=str, default=RL_DATASET_OUTPUT_DIR, help="Directory for dataset workflow artifacts.")
    parser.add_argument("--dataset_path", type=str, default=RL_EXISTING_DATASET, help="Path to an existing dataset CSV for analysis/comparison.")
    parser.add_argument("--dataset_seed_count", type=int, default=SEED_COUNT, help="Number of seeds per dataset instance configuration.")
    parser.add_argument("--dataset_area_values", type=str, default=AREA_VALUES, help="Comma-separated area counts for dataset generation.")
    parser.add_argument("--dataset_user_values", type=str, default=USER_VALUES, help="Comma-separated user densities for dataset generation.")
    parser.add_argument("--dataset_agent_values", type=str, default=AGENT_VALUES, help="Comma-separated agent counts for dataset generation.")
    parser.add_argument("--dataset_env_values", type=str, default=ENV_VALUES, help="Comma-separated environment values for dataset generation.")
    parser.add_argument("--dataset_scenario_values", type=str, default=SCENARIO_VALUES, help="Comma-separated scenario values for dataset generation.")
    parser.add_argument("--dataset_spread_values", type=str, default=SPREAD_VALUES, help="Comma-separated map spread values for dataset generation.")
    parser.add_argument("--dataset_action_ids", type=str, default=ACTION_IDS, help="Comma-separated action ids to evaluate, or all.")

    args = parser.parse_args()

    logger.debug(f"Configuration: Asset Directory -> {PROJECT_ASSETS}\n Trials -> {args.trials} \n Number of Agents -> {args.num_agents}\n Max Battery -> {args.max_battery} Wh\n Number of Areas -> {args.num_areas}\n Vertical Velocity -> {VERTICAL_VELOCITY} m/s\n Horizontal Velocity -> {HORIZONTAL_VELOCITY} m/s")

    if args.scenario not in SCENARIO_OPTIONS: 
        logger.error(f"Invalid scenario choice. Please choose from: {SCENARIO_OPTIONS}")
        sys.exit(1)

    if args.objective not in OBJECTIVE_OPTIONS:
        logger.error(f"Invalid objective choice. Please choose from: {OBJECTIVE_OPTIONS}")
        sys.exit(1)

    if args.env not in ENVIRONMENT_OPTIONS: 
        logger.error(f"Invalid environment choice. Please choose from: {ENVIRONMENT_OPTIONS}")
        sys.exit(1)

    if args.stage_solution not in STAGE_OPTIONS:
        logger.error(f"Invalid stage solution choice. Please choose from: {STAGE_OPTIONS}")
        sys.exit(1)

    if args.num_agents not in AGENTS_OPTIONS:
        logger.error(f"Invalid number of agents choice. Please choose from: {AGENTS_OPTIONS}")
        sys.exit(1)

    config = {
        "model_name": args.model_name,
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
        "solver_backend": args.solver_backend,
        "subtour_strategy": args.subtour_strategy,
        "objective_strategy": args.objective_strategy,
        "scenario_constraint_set": args.scenario_constraint_set,
        "solver_time_limit_seconds": None if args.solver_time_limit_seconds == 0 else args.solver_time_limit_seconds,
        "NUMBER_OF_AGENTS":args.num_agents,
        "NUMBER_OF_USERS":args.num_users,
        "NUMBER_OF_AREAS":args.num_areas,
        "altitude": ALTITUDE,
        "learning_enabled": args.enable_learning,
        "learning_alpha": args.learning_alpha,
        "learning_output_dir": RL_LEARNING_OUTPUT_DIR,
    }

    if args.workflow != "simulate":
        dataset_output_dir = args.dataset_output_dir
        dataset_path = args.dataset_path if args.dataset_path else f"{dataset_output_dir}/solver_dataset.csv"

        if args.workflow == "validate_actions":
            validation = validate_action_catalog(config, args.trial)
            os.makedirs(dataset_output_dir, exist_ok=True)
            validation_path = f"{dataset_output_dir}/action_validation.csv"
            validation.to_csv(validation_path, index=False)
            logger.info(f"Action validation saved to {validation_path}")
            sys.exit(0)

        if args.workflow == "dataset":
            specs = build_instance_specs(
                area_values=parse_csv_ints(args.dataset_area_values),
                user_values=parse_csv_ints(args.dataset_user_values),
                agent_values=parse_csv_ints(args.dataset_agent_values),
                env_values=parse_csv_strings(args.dataset_env_values),
                scenario_values=parse_csv_strings(args.dataset_scenario_values),
                spread_values=parse_csv_ints(args.dataset_spread_values),
                battery_values=[int(args.max_battery)],
                objective_function=args.objective,
                seed_count=args.dataset_seed_count,
            )

            action_ids = None if args.dataset_action_ids == "all" else parse_csv_strings(args.dataset_action_ids)
            created_path = generate_dataset(
                base_config=config,
                trials=args.trials,
                altitude=ALTITUDE,
                lat=LATITUDE_COORDS,
                lon=LONGITUDE_COORDS,
                vertical_velocity=VERTICAL_VELOCITY,
                horizontal_velocity=HORIZONTAL_VELOCITY,
                coverage_time=args.max_coverage_time,
                output_dir=dataset_output_dir,
                specs=specs,
                action_ids=action_ids,
            )
            logger.info(f"Dataset saved to {created_path}")
            sys.exit(0)

        if args.workflow == "analyze_dataset":
            outputs = analyze_dataset(dataset_path=dataset_path, output_dir=dataset_output_dir)
            logger.info(f"Dataset analysis saved to {outputs}")
            sys.exit(0)

        if args.workflow == "compare_baselines":
            comparison_path = compare_baselines(
                dataset_path=dataset_path,
                output_dir=dataset_output_dir,
                alpha=args.learning_alpha,
            )
            logger.info(f"Baseline comparison saved to {comparison_path}")
            sys.exit(0)

        logger.error(f"Invalid workflow: {args.workflow}")
        sys.exit(1)

    # Create Builder -> Holds variables and functions to create the combinatorial problem.
    problem = call_builder(config, args.trials)
    mobility_sim = EnvSim(trials=args.trials)
    progress = tqdm(total=args.trials, desc="Progress")

    map_generator = MapGenerator(
        num_areas=args.num_areas,
        users_per_area=args.num_users,
        lon=LONGITUDE_COORDS,
        lat=LATITUDE_COORDS,
        low=LOW_BOUND,
        high=HIGH_BOUND,
        seed=SEED_COUNT,
    )

    logger.debug("✅ Map Generator Initialized")

    regions, centroids, user_points, depots, distance_matrix, all_users = map_generator.create_environment(show_map=False, show_3d_map=False)

    ground_users = GroundUserGroup(
        mobility_env=mobility_sim,
        map_obj=map_generator,
        alpha=0.85,
        mean_velocity=10.0,
        sigma=0.5,
    )

    logger.debug("✅ Ground Users Group Initialized")

    ground_users.get_generated_users(user_points=user_points)

    logger.debug(f"✅Ground Users Loaded: {len(ground_users.group)} users")

    if map_generator.vor_map is None:
        raise ValueError("Voronoi map is not initialized. Ensure `voronoi_tessellation` is called successfully.")

    # all_user_points = [point for points in user_points.values() for point in points]

    mobility_sim.fig, mobility_sim.ax = ground_users.plot_users(map_generator.vor_map)

    data = problem.preprocess_generated_data(
        distance_matrix=distance_matrix,
        centroids=centroids,
        depots=depots if not isinstance(depots, list) else np.array(depots),
        num_of_agents=args.num_agents,
        v_hor=HORIZONTAL_VELOCITY,
        v_ver=VERTICAL_VELOCITY,
        altitude=ALTITUDE,
        coverage_time=args.max_coverage_time,
        user_points=user_points,
    )

    logger.debug("✅ Preprocessed Data Completed successfully")

    vor_map = map_generator.vor_map
    if not os.path.exists(ANIMATION_DIR):
        os.makedirs(ANIMATION_DIR)

    animation_filename = f"{ANIMATION_DIR}/simulation_output_{uuid.uuid4()}.mp4"

    try:
        ani = FuncAnimation(
            mobility_sim.fig,
            mobility_sim.simulations,
            frames=frame_generator(),
            fargs=(
                problem,
                ground_users,
                vor_map,
                distance_matrix,
                data,
                regions,
                centroids,
                user_points,
                ALTITUDE,
                args.trials,
            ),
            interval=100,
            blit=False,
            cache_frame_data=False,
        )
        ani.save(animation_filename, writer='ffmpeg', fps=10)
        print(f"Animation Saved to {animation_filename} with ffmpeg")
    except KeyboardInterrupt as kb:
        plt.close(mobility_sim.fig)
        logger.exception(f"KeyboardInterrupt: {kb}")
        sys.exit(1)

    except Exception as e:
        plt.close(mobility_sim.fig)
        logger.exception(f"Exception: {e}")
        sys.exit(1)

    finally:
        print("Program Terminated Gracefully...")
        if progress is not None:
            progress.close()


if __name__=="__main__": 
    main()
