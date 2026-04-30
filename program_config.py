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
TRIALS = 2

NUMBER_OF_AGENTS = 3 # Number of UAVs 
NUMBER_OF_AREAS = 21 # Areas used for Voronoi map genertion
NUMBER_OF_USERS = 1 # Users to cover per Area

LATITUDE_COORDS = 37.961322948559
LONGITUDE_COORDS = 23.708232317542667
ALTITUDE = 1250 # Optimal Coverage Altitude 

LOW_BOUND = 75 # Considered in meters 
HIGH_BOUND = 120 # Considered in meters 

# --- UAV Config --- # 
MAX_BATTERY = 355.2 #Wh 
MAX_COVERAGE_TIME = 3
MASS = 6.4 # In kg with payload for coverage

HORIZONTAL_VELOCITY = 15.56 # m/s 
VERTICAL_VELOCITY = 2.78 # m/s 

# --- Scenario Options  --- # 
SCENARIO_OPTIONS = ['cooperative', 'individual']
MODEL_NAME = "milp"
OBJECTIVE_OPTIONS = ['energy', 'coverage', 'sum_of_times', 'pareto' ]
ENVIRONMENT_OPTIONS = ['urban', 'rural', 'forest']
STAGE_OPTIONS = [1, 2, 3] 
AGENTS_OPTIONS = [3, 4, 5, 6, 7] 
FAIRNESS_TOLERANCE = [0, 1, 2, 3, 5]
TIME_STEP_SEC = [30, 60, 120, 300, 600]

# --- Performance Admissions --- # 
MAX_MEMORY = 2 * 1024 * 1024 * 1024 # 2GB
SOLVER_TIME_LIMIT = 30 # Considered in mins

# --- Mechanisms Enabled --- # 
ENABLE_GA = 'yes'
SCENARIO_CONSTRAINT_SET = "default"
PRIORITY = 'yes' 
SOLVER_BACKEND = 'glpk'
SUBTOUR_STRATEGY = 'mtz' 
OBJECTIVE_STRATEGY = 'legacy_stage'
WORKFLOW = 'simulate' 

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

# --- Energy configuration --- # 
ROTOR_AREA = 0.2 # Rotor disk area in m 
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




