import os

# --- Project INFO --- # 
PROJECT_DIR = os.getcwd() 
PROJECT_ASSETS = f"{PROJECT_DIR}/assets" 
ANIMATION_DIR = f"{PROJECT_ASSETS}/animations"
RL_LEARNING_OUTPUT_DIR = f"{PROJECT_ASSETS}/results/rl"
RL_EXISTING_DATASET = "" 
RL_DATASET_OUTPUT_DIR = f"{PROJECT_DIR}/results/rl_dataset"
CENTROIDS_PATH = f"{PROJECT_DIR}/dummy_app/drone_centroids_path.csv"

# --- Simulation Config --- # 
TRIALS = 1

NUMBER_OF_AGENTS = 3 # Number of UAVs 
NUMBER_OF_AREAS = 21 # Areas used for Voronoi map genertion
NUMBER_OF_USERS = 1 # Users to cover per Area

LATITUDE_COORDS = 37.961322948559
LONGITUDE_COORDS = 23.708232317542667
ALTITUDE = 100 # Optimal Coverage Altitude 

LOW_BOUND = 75 # Considered in meters 
HIGH_BOUND = 120 # Considered in meters 

# --- UAV Config --- # 
MAX_BATTERY = 362 #Wh 
MAX_COVERAGE_TIME = 3
MASS = 5.4 # In kg with payload for coverage

HORIZONTAL_VELOCITY = 10 # m/s 
VERTICAL_VELOCITY = 2.5 # m/s 

# --- Scenario Options  --- # 
SCENARIO_OPTIONS = ['cooperative', 'individual']
MODEL_NAME = "milp"
OBJECTIVE_OPTIONS = ['energy', 'coverage', 'sum_of_times', 'pareto' ]
ENVIRONMENT_OPTIONS = ['urban', 'rural', 'forest']
STAGE_OPTIONS = [1, 2, 3] 
AGENTS_OPTIONS = [3, 4, 5, 6, 7, 8, 12, 15] 
FAIRNESS_TOLERANCE = [0, 1, 2, 3, 5]
TIME_STEP_SEC = [30, 60, 120, 300, 600]

# --- Performance Admissions --- #
MAX_MEMORY = 2 * 1024 * 1024 * 1024 # 2GB
SOLVER_TIME_LIMIT = 7200  # seconds — per-cluster ceiling for interactive (execution_script) runs
RUN_TIME_LIMIT_SECONDS = 7200  # seconds — total wall-clock cap across all clusters in one run

# --- Mechanisms Enabled --- # 
ENABLE_GA = 'yes'
WARM_START_MODE = "auto"
SCENARIO_CONSTRAINT_SET = "default"
PRIORITY = 'yes' 
SOLVER_BACKEND = 'gurobi'
SUBTOUR_MODE = 'mtz'
SUBTOUR_STRATEGY = SUBTOUR_MODE
OBJECTIVE_STRATEGY = 'legacy_stage'
WORKFLOW = 'simulate' 

# The solver still targets certified optimality (0% gap).  If the configured
# ceiling is reached, a feasible incumbent at or below this gap is explicitly
# identified as the near-optimal fallback rather than being discarded.
SOLVER_FALLBACK_GAP_REL = 0.02
SOLVER_WATCHDOG_GRACE_SECONDS = 60

# Feasibility-oriented search settings. These affect search order, not the
# mathematical formulation or the configured cluster time ceiling.
GUROBI_MIP_FOCUS = 1
GUROBI_HEURISTICS = 0.15
GUROBI_PRESOLVE = 2

# --- RL Parameters --- # 
ENABLE_LEARNING = 'no'
LEARNING_ALPHA = 0.75 # Exploration factor for the contextual bandit
SEED_COUNT = 42 
AREA_VALUES = "21, 30, 40"
AGENT_VALUES = "3, 5, 7"
USER_VALUES = "1, 2, 4"
ENV_VALUES = "urban, rural, forest"
SCENARIO_VALUES = "cooperative, individual"
SPREAD_VALUES = "60, 90, 120" 
ACTION_IDS = "all"

# --- Experiment defaults --- #
EXPERIMENT_RESULTS_DIR = f"{PROJECT_DIR}/results"
EXPERIMENT_DEFAULT_SEEDS = [SEED_COUNT]
EXPERIMENT_DEFAULT_SCENARIO = "cooperative"
EXPERIMENT_DEFAULT_OBJECTIVE = "energy"
EXPERIMENT_DEFAULT_ENV = ENVIRONMENT_OPTIONS[0]
EXPERIMENT_DEFAULT_PRIORITY = PRIORITY
EXPERIMENT_DEFAULT_MEMORY_LIMIT = MAX_MEMORY
EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS = 3600  # seconds — per-cluster ceiling for experiment runs
EXPERIMENT_DEFAULT_RUN_TIME_LIMIT_SECONDS = RUN_TIME_LIMIT_SECONDS  # total wall-clock cap per experiment run

EXPERIMENT_COVERAGE_TIME_PROFILES = {
    "low": 1,
    "medium": max(1, int(MAX_COVERAGE_TIME) - 1),
    "high": int(MAX_COVERAGE_TIME),
}

EXPERIMENT_OBJECTIVE_WEIGHT_PROFILES = {
    "balanced": {"distance": 0.33, "energy": 0.33, "travel_time": 0.34},
    "energy_focused": {"distance": 0.20, "energy": 0.60, "travel_time": 0.20},
    "distance_focused": {"distance": 0.60, "energy": 0.20, "travel_time": 0.20},
    "time_focused": {"distance": 0.20, "energy": 0.20, "travel_time": 0.60},
}

EXPERIMENT_A_GRID = {
    "areas": [25, 50, 100, 150, 200],
    "users_per_area": [1, 5, 10],
    "uavs": [3, 4, 6, 8],
    "coverage_time_profile": ["low", "medium", "high"],
}

EXPERIMENT_B_GRID = {
    "areas": [25, 50, 100, 200],
    "uavs": [3, 4, 6, 8],
}
EXPERIMENT_B_SOLVERS = ["glpk", "cbc", "gurobi", "cplex"]

EXPERIMENT_C_GRID = {
    "areas": [50, 100, 150, 200],
    "uavs": [3, 4, 6, 8],
}

EXPERIMENT_C_WARM_STARTS = ["none", "ga", "alns", "greedy_nn", "greedy_partition_nn"]

EXPERIMENT_D_GRID = {
    "areas": [50, 100, 150, 200],
    "uavs": [3, 4, 6, 8],
}

EXPERIMENT_D_FORMULATIONS = {
    "single_stage_milp": 1,
    "two_stage_milp": 2,
}

EXPERIMENT_SUBTOUR_AREAS = [50, 100, 150, 200]
EXPERIMENT_SUBTOUR_UAVS = [3, 5]
EXPERIMENT_SUBTOUR_MODES = ["mtz", "dfj_iter", "flow"]
EXPERIMENT_SUBTOUR_SOLVER = "cplex"

EXPERIMENT_E_BASELINE = {
    "areas": 100,
    "users_per_area": 3,
    "uavs": 4,
    "coverage_time_profile": "medium",
}

EXPERIMENT_E_FAIRNESS_THRESHOLDS = [1, 2, 3, 4]
EXPERIMENT_E_TIME_STEP_SEC = [30, 60, 120]
EXPERIMENT_E_OBJECTIVE_WEIGHT_PROFILES = [
    "balanced",
    "energy_focused",
    "distance_focused",
    "time_focused",
]

EXPERIMENT_H_BASELINE = {
    "areas": 100,
    "users_per_area": 1,
    "uavs": 4,
    "coverage_time_profile": "medium",
}
EXPERIMENT_H_REQUIRED_VISITS = [1, 2, 3, 4, 5]

# --- Energy configuration --- # 
ROTOR_AREA = 0.45 # Rotor disk area in m 
LAMBDA_COEF = 0.08 # Coeff for the drag profile depending on the type of UAV 
ASCENT_FACTOR = 1.0 
DESCENT_FACTOR = [0.2, 0.4, 0.6, 0.8]
G = 9.81 # Acceleration of gravity in m/s^2 
P = 1.225 # Air density of gravity in kg/m^3 
MIN_HOVER = 30 # Least power required to minimally hover over the grouind
NUMBER_OF_ROTORS = 4 
P_CHARGE = 350 # WATTS 
P_BS = 200 # In W is the power to operate the drone as a low level base station
MOTOR_SPEED_MULTIPLIER = 10.5
BANK_ANGLE_DEG = 0.0 
