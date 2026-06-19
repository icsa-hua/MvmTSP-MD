from dummy_app.tools.logger import logger 
from dummy_app.core.exceptions import ValidationOptimalityConfirmed
from dummy_app.models.RL import analyze_dataset, build_instance_specs, compare_baselines, generate_dataset, validate_action_catalog
from dummy_app.designs.envsim import EnvSim
from dummy_app.designs.mobility import GroundUserGroup
from dummy_app.designs.voronoi_map import MapGenerator
from dummy_app.tools.common import call_builder
from dummy_app.visualization.playback import find_latest_playback_artifact, render_playback
from dummy_app.program_config import * 

import os
import sys
import uuid 
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer

from dummy_app.pipeline.artifacts import persist_playback_artifacts

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
transformer_to_latlon = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)


def parse_csv_ints(raw_value):
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_csv_strings(raw_value):
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def build_runtime_config(args):
    warm_start_mode = args.warm_start_mode if args.warm_start_mode else WARM_START_MODE
    return {
        "model_name": args.model_name,
        "genetic_algorithm": warm_start_mode == "ga",
        "env_type": args.env, 
        "max_battery":args.max_battery, 
        "max_coverage_time":args.max_coverage_time,
        "scenario":args.scenario, 
        "enable_ga": "yes" if warm_start_mode == "ga" else "no",
        "objective_function":args.objective,
        "stage_solution":args.stage_solution, 
        "priority":args.priority,
        "validate":args.validate,
        "solver_backend": args.solver_backend,
        "subtour_mode": args.subtour_mode,
        "subtour_strategy": args.subtour_mode,
        "objective_strategy": args.objective_strategy,
        "scenario_constraint_set": args.scenario_constraint_set,
        "warm_start_mode": warm_start_mode,
        "solver_time_limit_seconds": None if args.solver_time_limit_seconds == 0 else args.solver_time_limit_seconds,
        "solver_fallback_gap_rel": SOLVER_FALLBACK_GAP_REL,
        "solver_watchdog_grace_seconds": SOLVER_WATCHDOG_GRACE_SECONDS,
        "random_seed": args.seed,
        "solver_seed": args.seed,
        "NUMBER_OF_AGENTS":args.num_agents,
        "NUMBER_OF_USERS":args.num_users,
        "NUMBER_OF_AREAS":args.num_areas,
        "altitude": ALTITUDE,
        "learning_enabled": args.enable_learning,
        "learning_alpha": args.learning_alpha,
        "learning_output_dir": RL_LEARNING_OUTPUT_DIR,
    }


def build_problem_context(args):
    config = build_runtime_config(args)
    problem = call_builder(config, args.trials)
    mobility_sim = EnvSim(trials=args.trials, render=False)
    map_generator = MapGenerator(
        num_areas=args.num_areas,
        users_per_area=args.num_users,
        lon=LONGITUDE_COORDS,
        lat=LATITUDE_COORDS,
        low=LOW_BOUND,
        high=HIGH_BOUND,
        seed=args.seed,
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
    return problem, mobility_sim, ground_users, data, distance_matrix, regions, user_points


def run_solve_mode(args):
    problem, mobility_sim, ground_users, data, distance_matrix, regions, user_points = build_problem_context(args)
    constructor = mobility_sim.run_headless(
        constructor=problem,
        cues=ground_users,
        distance_matrix=distance_matrix,
        data=data,
        altitude=ALTITUDE,
        trials=args.trials,
    )
    persist_combined_playback(problem, mobility_sim, regions, user_points)
    run_result = getattr(constructor, "latest_model_run_result", None)
    run_summary = dict(getattr(constructor, "latest_run_summary", {}))
    if run_result is not None:
        logger.info(
            "Solve completed. "
            f"status={run_result.normalized_status}, "
            f"objective_value={run_result.objective_value}, "
            f"best_bound={run_result.best_bound}, "
            f"optimality_gap_percent={run_summary.get('optimality_gap_percent')}, "
            f"artifacts={constructor.latest_artifact_dir}"
        )
    else:
        logger.info(f"Solve completed. Artifacts: {constructor.latest_artifact_dir}")
    return constructor.latest_artifact_dir


def resolve_artifact_dir(args):
    if args.artifact_dir:
        return args.artifact_dir
    return str(find_latest_playback_artifact(f"{PROJECT_ASSETS}/results/metrics"))


def run_animation_mode(args, artifact_dir: str):
    if not os.path.exists(ANIMATION_DIR):
        os.makedirs(ANIMATION_DIR)

    output_path = None
    if args.save_animation:
        output_path = args.animation_output if args.animation_output else f"{ANIMATION_DIR}/simulation_output_{uuid.uuid4()}.mp4"

    rendered_path = render_playback(
        artifact_dir=artifact_dir,
        save_path=output_path,
        fps=args.animation_fps,
        interval_ms=args.animation_interval_ms,
    )

    if rendered_path:
        logger.info(f"Animation saved to {rendered_path}")
    else:
        logger.info(f"Playback rendered from artifact {artifact_dir}")
    return rendered_path


def persist_combined_playback(problem, mobility_sim, regions, user_points) -> None:
    if not problem.latest_artifact_dir:
        return

    combined_rows = []
    for agent_id, coordinate_path in mobility_sim.combined_paths.items():
        for (x0, y0), (x1, y1), timestep in coordinate_path:
            lon0, lat0 = transformer_to_latlon.transform(x0, y0)
            lon1, lat1 = transformer_to_latlon.transform(x1, y1)
            combined_rows.append(
                {
                    "agent": agent_id,
                    "from_x": lon0,
                    "from_y": lat0,
                    "to_x": lon1,
                    "to_y": lat1,
                    "time_step": timestep,
                }
            )

    if not combined_rows:
        return

    pd.DataFrame(combined_rows).to_csv(CENTROIDS_PATH, index=False)
    playback_rows, playback_metadata = problem.build_playback_timeline(combined_rows)
    playback_metadata["num_sessions"] = mobility_sim.completed_sessions
    playback_metadata["voronoi_regions"] = []
    for region in regions:
        if not hasattr(region, "exterior"):
            continue
        x_coords, y_coords = region.exterior.xy
        polygon = []
        for x_coord, y_coord in zip(x_coords, y_coords):
            lon, lat = transformer_to_latlon.transform(float(x_coord), float(y_coord))
            polygon.append([lon, lat])
        playback_metadata["voronoi_regions"].append(polygon)

    cue_points = []
    for area_points in user_points.values():
        for x_coord, y_coord in area_points:
            lon, lat = transformer_to_latlon.transform(float(x_coord), float(y_coord))
            cue_points.append({"x": lon, "y": lat})
    playback_metadata["cue_points"] = cue_points
    persist_playback_artifacts(problem.latest_artifact_dir, playback_rows, playback_metadata)


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
    parser.add_argument("--mode", type=str, default="solve", help="Execution mode: solve, animate, solve_and_animate.")
    parser.add_argument("--artifact_dir", type=str, default="", help="Artifact directory to replay for animate mode.")
    parser.add_argument("--save_animation", action="store_true", help="Persist the animation as an MP4.")
    parser.add_argument("--animation_output", type=str, default="", help="Optional output path for saved animation.")
    parser.add_argument("--animation_fps", type=int, default=10, help="Frames per second when saving animation.")
    parser.add_argument("--animation_interval_ms", type=int, default=100, help="Playback interval between frames in milliseconds.")
    parser.add_argument("--enable_ga", type=str, default=ENABLE_GA, help="Initialize solver with Genetic Algorithm")
    parser.add_argument(
        "--warm_start_mode",
        type=str,
        default=WARM_START_MODE,
        help="MILP initializer: auto, none, ga, alns, greedy_nn, or greedy_partition_nn.",
    )
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
    parser.add_argument(
        "--subtour_mode",
        "--subtour_strategy",
        dest="subtour_mode",
        type=str,
        default=SUBTOUR_MODE,
        help="Subtour elimination mode: mtz, dfj_iter, or flow.",
    )
    parser.add_argument("--objective_strategy", type=str, default=OBJECTIVE_STRATEGY, help="Objective orchestration strategy.")
    parser.add_argument("--scenario_constraint_set", type=str, default=SCENARIO_CONSTRAINT_SET, help="Constraint set variant to use.")
    parser.add_argument("--solver_time_limit_seconds", type=int, default=SOLVER_TIME_LIMIT, help="Optional MILP solver time limit in seconds; 0 disables the override.")
    parser.add_argument("--seed", type=int, default=SEED_COUNT, help="Seed for instance generation, heuristics, and GLPK.")
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

    valid_modes = {"solve", "animate", "solve_and_animate"}
    if args.mode not in valid_modes:
        logger.error(f"Invalid mode. Please choose from: {sorted(valid_modes)}")
        sys.exit(1)

    if args.workflow != "simulate":
        dataset_output_dir = args.dataset_output_dir
        dataset_path = args.dataset_path if args.dataset_path else f"{dataset_output_dir}/solver_dataset.csv"

        if args.workflow == "validate_actions":
            config = build_runtime_config(args)
            validation = validate_action_catalog(config, args.trials)
            os.makedirs(dataset_output_dir, exist_ok=True)
            validation_path = f"{dataset_output_dir}/action_validation.csv"
            validation.to_csv(validation_path, index=False)
            logger.info(f"Action validation saved to {validation_path}")
            sys.exit(0)

        if args.workflow == "dataset":
            config = build_runtime_config(args)
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

    try:
        artifact_dir = ""
        if args.mode in {"solve", "solve_and_animate"}:
            artifact_dir = run_solve_mode(args)
        if args.mode == "animate":
            artifact_dir = resolve_artifact_dir(args)
            run_animation_mode(args, artifact_dir)
        elif args.mode == "solve_and_animate":
            run_animation_mode(args, artifact_dir)
    except KeyboardInterrupt as kb:
        logger.exception(f"KeyboardInterrupt: {kb}")
        sys.exit(1)

    except ValidationOptimalityConfirmed as exc:
        logger.info(str(exc))
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Exception: {e}")
        sys.exit(1)

    finally:
        print("Program Terminated Gracefully...")
        plt.close("all")


if __name__=="__main__": 
    main()
