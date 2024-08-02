
import sys 
import resource
import geopandas
import psutil
import os 
import time
import pdb #For debugging 
import time 
import gc
import warnings
import pulp as pl
import numpy as np 
import pandas as pd
sys.path.append('/Matlab Projects/Swarms (Refined)/interface')
import matplotlib.pyplot as plt
import timeout_decorator 
from sklearn.preprocessing import robust_scale 
from k_means_constrained import KMeansConstrained
from pulp import LpProblem, LpMinimize, lpSum, LpVariable 
from interface.formulation_module import MvmTSP
from dummy_mvmtsp.ga_solver import GASOL 
from memory_profiler import profile

import logging
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

class ConstrainedMVMTSP(MvmTSP): 

    def __init__(self, data, config, max_memory=None): 
        
        """
        Initializes the ConstrainedMVMTSP class, which is a subclass of the MvmTSP class. This class is responsible for creating and solving a constrained multi-vehicle multi-depot traveling salesman problem (MVMTSP).
        
        The __init__ method takes the following parameters:
        - data: a dataset containing information about the problem, such as the locations of the depots and customers.
        - use_GA: a boolean flag indicating whether to use a genetic algorithm to solve the problem.
        - regionalization: a boolean flag indicating whether to use regionalization to solve the problem.
        - dictionary_flag: a boolean flag indicating whether to use a dictionary-based approach to handle the constraints.
        - max_memory: the maximum amount of memory to use for each agent.
        The method sets up the optimization problem, including the objective function and the constraints. It also sets a memory limit for the problem.
        """
                
        super().__init__(data, config['genetic_algorithm'], config['regionalization'])
        self.model = LpProblem("ConstrainedMvmTSP",LpMinimize)
        self.start_time = time.time()
        self.config = config

        if not self.regionalization:
            self.config['individual_solution'] = False
        

        if self.config['individual_solution']: 
            self.constraints = {
                        "const_1": self.add_constraint_1_dict,
                        "const_2": self.add_constraint_2_dict,
                        "const_3": self.add_constraint_3_dict,
                        "const_4": self.add_constraint_4_dict,
                        "const_5": self.add_constraint_5_dict,
                        "const_6": self.add_constraint_6_dict,
                        "const_7": self.add_constraint_7_dict,
                        "const_8": self.add_constraint_8_dict,
                        "const_9": self.add_constraint_9_dict,
            }
        else:
            self.constraints = {
                "const_1": self.add_constraint_1,
                "const_2": self.add_constraint_2,
                "const_3": self.add_constraint_3,
                "const_4": self.add_constraint_4,
                "const_5": self.add_constraint_5,
                "const_6": self.add_constraint_6,
                "const_7": self.add_constraint_7,
                "const_8": self.add_constraint_8,
                "const_9": self.add_constraint_9,
            }

        self.set_memory_limit(max_memory)

    def createProblem(self, V=None):
        
        """
        Creates the optimization problem for the constrained many-visits multi traveling salesman problem (MVMTSP).
        
        This method initializes the optimization problem and defines the decision variables:
        - `x`: a binary variable indicating whether agent k travels from area i to area j
        - `u`: an integer variable used for the subtour elimination constraints
        - `t`: a binary variable to handle the execution based on timing
        - `e`: a continuous variable that holds information about the energy consumption between two nodes
        - `z`: a binary variable to have information about the customer service
        
        """
        
        self.model = LpProblem("ConstrainedMvmTSP",LpMinimize)
        self.x = LpVariable.dicts("x", ((i,j,k) for i in V for j in V for k in self.agents), cat='Binary') #Variable to show if agent k traveld from area i to j. 
        self.u = LpVariable.dicts("u", ((i,k) for i in V for k in self.agents), lowBound=0, upBound=len(V)-1, cat='Integer') #Variable to use for the subtour elimination constraints. 
        self.t = LpVariable.dicts("t", ((i,j,k,ts) for i in V for j in V for k in self.agents for ts in self.TimeFrame), cat='Binary') #Variable to handle the execution based on timing. 
        self.e = LpVariable.dicts("e", ((i,k) for i in V for k in self.agents),lowBound=0,upBound=self.max_battery, cat='Continuous') #Variable that holds information about the energy consumption between two nodes.  
        self.z = LpVariable.dicts("z_ik", ((i,k) for i in V for k in self.agents), lowBound=0,upBound=1,cat='Binary') #Varable to have information about the customer service. 

    def set_objective(self, distances=[], energy=[], V_nodes=[], V=None):
        
        """
        Sets the objective function for the constrained MVMTSP.
        
        The objective function is to minimize the total distance and energy consumption for all agents traveling between the nodes in the problem.
        The `distances` and `energy` lists contain the distances and energy consumption between each pair of nodes, respectively.
        The `V_nodes` list contains the indices of the nodes in the problem, and `V` is a list of the node IDs.
        """
                
        self.model.setObjective(lpSum(distances[i][V[j]-1] * self.x[i,j,k] + energy[i][V[j]-1] * self.x[i,j,k] for i in V_nodes for j in V_nodes if i!=j for k in self.agents)) 

    def createProblem_dict(self, V=None):
       
        """
        Creates the optimization problem for the constrained many-visits multi traveling salesman problem (MVMTSP).
        
        This method initializes the optimization problem and defines the decision variables:
        - `x`: a binary variable indicating whether agent k travels from area i to area j
        - `u`: an integer variable used for the subtour elimination constraints
        - `t`: a binary variable to handle the execution based on timing
        - `e`: a continuous variable that holds information about the energy consumption between two nodes
        - `z`: a binary variable to have information about the customer service
        
        """

        self.model = LpProblem("ConstrainedMvmTSP",LpMinimize)
        self.x = LpVariable.dicts("x", ((i,j,k) for i in V for j in V for k in self.agents), cat='Binary') #Variable to show if agent k traveld from area i to j. 
        self.u = LpVariable.dicts("u", ((i,k) for i in V for k in self.agents), lowBound=0, upBound=len(V)-1, cat='Integer') #Variable to use for the subtour elimination constraints. 
        self.t = LpVariable.dicts("t", ((i,j,k,ts) for i in V for j in V for k in self.agents for ts in self.TimeFrame), cat='Binary') #Variable to handle the execution based on timing. 
        self.e = LpVariable.dicts("e", ((i,k) for i in V for k in self.agents),lowBound=0,upBound=self.max_battery, cat='Continuous') #Variable that holds information about the energy consumption between two nodes.  
        self.z = LpVariable.dicts("z_ik", ((i,k) for i in V for k in self.agents), lowBound=0,upBound=1,cat='Binary') #Varable to have information about the customer service. 

    def set_objective_dict(self, distances=[], energy=[], V_nodes=[], V=None):
       
        """
        Sets the objective function for the constrained MVMTSP.
        
        The objective function is to minimize the total distance and energy consumption for all agents traveling between the nodes in the problem.
        The `distances` and `energy` lists contain the distances and energy consumption between each pair of nodes, respectively.
        The `V_nodes` list contains the indices of the nodes in the problem, and `V` is a list of the node IDs.
        """

        self.model.setObjective(lpSum(distances[V[i]][j] * self.x[i,j,k] + energy[V[i]][j-1] * self.x[i,j,k] for i in V_nodes for j in V_nodes if i!=j for k in self.agents)) 

    def assign_agents_to_areas(self, num_agents: int, depots: list)->dict:
       
        """ 
        Assigns agents to areas based on the number of agents and the list of depots.
        
        Args:
            num_agents (int): The total number of agents.
            depots (list): A list of the depot areas.
        
        Returns:
            dict: A dictionary mapping each agent to the area they are assigned to.
        """
        
        agents_per_area = num_agents // len(depots)
        remaining_areas = num_agents % len(depots)
        area_agents = {area: agents_per_area for area in depots}
        agents_assignments = {agent : 0 for agent in range(1,num_agents+1)}
        agent_counter = 1 
        for area in depots: 
            num_agents = area_agents[area]
            for _ in range(num_agents): 
                agents_assignments[agent_counter] = area 
                agent_counter += 1 

        for i in range(remaining_areas): 
            area = depots[i] 
            agents_assignments[agent_counter] = area
            agent_counter += 1 
            
        return agents_assignments

    
    def createGeoDataset(self, data)->geopandas.GeoDataFrame:
        
        """
        Creates a GeoDataFrame from the provided data, with the geometry set to points based on the X_coords and Y_coords columns.
        
        Args:
            data (pandas.DataFrame): The input data containing the X_coords and Y_coords columns.
        
        Returns:
            geopandas.GeoDataFrame: A GeoDataFrame with the geometry set to points based on the X_coords and Y_coords columns.
        """

        return geopandas.GeoDataFrame(data, geometry=geopandas.points_from_xy(data.X_coords, data.Y_coords), crs="EPSG:4326")

    
    def set_memory_limit(self, max_memory: int): 
        """
        Sets the maximum memory limit for the process.
        
        Args:
            max_memory (int): The maximum amount of memory (in bytes) that the process can use.
        
        This function sets the maximum memory limit for the current process using the `resource.setrlimit()` function. This can be useful for preventing the process from consuming too much memory and causing issues on the system.
        """
                
        resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))


    def preprocess(self, cost_path_1:str, cost_path_2:str, nodes_path:str, agents:int, customers_path:str, max_battery:int): 
        """
        Preprocesses the data required for the constrained multi-vehicle multi-TSP (MVMTSP) problem.
        
        This method reads in the necessary data files, including the node locations, distance matrix,
        and energy consumption matrix. It also assigns agents to depots and sets up the necessary
        variables and data structures for the optimization problem.
        
        Args:
            cost_path_1 (str): The file path for the distance matrix.
            cost_path_1 (str): The file path for the energy consumption matrix.
            nodes_path (str): The file path for the node locations.
            agents (int): The number of agents to be used in the optimization.
            customers_path (str): The file path for the customers locations.
            max_battery (int): The maximum battery capacity for the agents.
        
        Returns:
            pandas.DataFrame: The preprocessed data, including the distance matrix, energy consumption matrix, and other relevant information.
        """
                
        self.V = pd.read_csv(nodes_path)
        self.v = len(self.V)
        
        self.distances = pd.read_csv(cost_path_1)
        dist_columns = [f'dist_{i}' for i in range(1,self.v+1)]
        self.distances.columns = dist_columns

        self.energy = pd.read_csv(cost_path_2)
        ee_columns = [f'ee_{i}' for i in range(1,self.v+1)]
        self.energy.columns = ee_columns

        self.agents = list(range(1,agents+1))

        self.customers = pd.read_csv(customers_path)
        self.customers = self.customers.to_numpy()
        self.max_battery = max_battery

        self.R_points = agents * np.ones(self.v).astype(int)
        rv = pd.DataFrame({"R": self.R_points[:]})

        if self.config['regionalization']:
            self.dist_columns = dist_columns
            self.energy_columns = ee_columns
            self.R_column = "R"

        self.TimeFrame = list(range(1,self.v*2,1))
        self.moment = 1 
        data = [self.distances, self.energy, rv, self.V]
        data = pd.concat(data,axis=1,join="inner")

        self.V = list(self.V['Area_id'])
        depots = [self.V[7], self.V[8]]
        self.depots_for_agents = self.assign_agents_to_areas(agents,depots)
        self.paths = {k:[] for k in self.agents}

        return data 


    @timeout_decorator.timeout(3600) #1 hour limit
    def solve_problem(self):
        
        """
        Solves the optimization problem using the GLPK solver with a maximum MIP gap of 0.05.
        """
                
        self.model.solve(pl.GLPK(msg=False,options=['--mipgap','0.05'] ))
         
    
    def call_genetic_algorithm(self, V_nodes, cost, agent=None) -> list: 
        """
        Args:
            V_nodes (list): A list of node IDs.
            cost (dict): A dictionary mapping node IDs to their associated costs.
        
        Returns:
            list: A list of the best paths found by the genetic algorithm.
        """
                
        population_size = 200 
        generations = 100 
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
           
            if self.config['individual_solution'] and isinstance(self.initial_population,dict):
                ga = GASOL(population_size,generations,V_nodes,self.agents,self.depots_for_agents[agent])
                best_paths, hof = ga.run(cthr=0.7, mthr=0.05, cost=cost, enable_individual_fitness=True)
                return best_paths 
            
            initial_population = [] 
            ga = GASOL(population_size,generations,V_nodes,self.agents,self.depots_for_agents)
            best_paths, hof = ga.run(cthr=0.7, mthr=0.05, cost=cost, enable_individual_fitness=False)
            if len(best_paths) < len(self.agents): 
                initial_population = best_paths
                additional_path = next(iter(best_paths)) 
                for _ in range(len(self.agents)-len(best_paths)): 
                    initial_population.append(additional_path)
            else: 
                initial_population = best_paths

            return initial_population

    
    def regionOptimize(self, GDF): 
    
        cluster_variables=[self.dist_columns + self.energy_columns + list(self.R_column)]
        max_nodes = int(self.max_battery/np.average(self.energy))
        charge_points = int(np.floor(len(self.V)/max_nodes))
        db_scaled = robust_scale(GDF[cluster_variables[0]])
        kmeans = KMeansConstrained(n_clusters=charge_points+1, 
                                   size_min=charge_points, 
                                   size_max=max_nodes-3, 
                                   random_state=0)
        
        np.random.seed(1234)
        k5cls = kmeans.fit_predict(db_scaled)
        GDF['k5cls'] = k5cls 
        clusters = GDF.groupby("k5cls")
        return clusters


    def process_with_dictionary(self, context):
        
        cost_d = dict(zip(context['area_ids'],context['dists']))
        cost_e = dict(zip(context['area_ids'],context['ees']))
        R_points = dict(zip(context['area_ids'],context['R']))
        nodes = {agent : np.copy(context['area_ids'].astype(int)) for agent in self.agents}
        virtual_nodes = list(context['area_ids'].astype(int))
        for k in self.agents: 
            depot = self.depots_for_agents[k]

            if depot not in nodes[k]:
                nodes[k] = np.append(nodes[k],depot)
                cost_d[depot] = self.distances.loc[depot].to_numpy()
                cost_e[depot] = self.energy.loc[depot].to_numpy()   
                R_points[depot] = self.R_points[depot]

            if depot not in virtual_nodes:
                virtual_nodes.append(depot)

        if any(len(R_points) != len(dictionary) for dictionary in (cost_d, cost_e)):
            warnings.warn("Sizes of area attribute arrays do not match.")
            return

        self.initial_population = {agent: [] for agent in self.agents}
        nodes_dict = {i: n for i, n in enumerate(virtual_nodes)}
        V = {agent: {i: nodes[agent][i] for i in range(len(nodes[agent]))} for agent in self.agents}                    # cost_d = {area_id : dists for area_id, dists in zip(cluster[1]['Area_id'], cluster[1][self.dist_columns].to_numpy().astype(float))}
        
        if self.config['genetic_algorithm']:
            logging.info("Genetic Algorithm Enabled...")
            for k in self.agents:
                self.initial_population[k] = self.call_genetic_algorithm(V[k], cost_d, k)


        V_nodes = list(range(len(virtual_nodes)))
        return cost_d, cost_e, R_points, V_nodes, nodes_dict


    def process_without_dictionary(self,context): 
        nodes = list(context['area_ids'].astype(int))
        cost_d = context['dists']
        cost_e = context['ees']
        R_points = list(context['R'])
        
        for k in self.depots_for_agents:
            
            if self.depots_for_agents[k] not in nodes: 
                nodes.append(self.depots_for_agents[k])
                cost_d = np.append(cost_d, self.distances.loc[self.depots_for_agents[k]].to_numpy().reshape(1, -1), axis=0)
                cost_e = np.append(cost_e, self.energy.loc[self.depots_for_agents[k]].to_numpy().reshape(1, -1), axis=0)
                R_points.append(self.R_points[self.depots_for_agents[k]])

        if any(len(R_points) != len(dictionary) for dictionary in (cost_d, cost_e)):
            warnings.warn("Sizes of area attribute arrays do not match.")
            return

        V_nodes = list(range(len(nodes)))
        V = {V_nodes[i] : nodes[i] for i in range(len(nodes))}
            
        if self.config['genetic_algorithm']:
            logging.info("Genetic Algorithm Enabled...")
            self.initial_population = self.call_genetic_algorithm(V, cost_d)
            # print(self.initial_population)
        
        return cost_d, cost_e, R_points, V_nodes, V
    

    def extract_content(self): 
        return {
            "dists": self.distances.to_numpy().astype(float),
            "ees": self.energy.to_numpy().astype(float),
            "R": self.R_points,
            "area_ids": self.V,
        }


    def extract_context_cluster(self, cluster):
        return {
            "dists": cluster[1][self.dist_columns].to_numpy().astype(float),
            "ees": cluster[1][self.energy_columns].to_numpy().astype(float),
            "R": cluster[1][self.R_column].to_numpy().astype(int),
            "area_ids": cluster[1]['Area_id'].to_numpy(),
        }


    def clustering(self, cluster): 
        context = self.extract_context_cluster(cluster)
        
        if self.config['individual_solution']: 
            
            cost_d, cost_e, R_points, V_nodes, nodes_dict = self.process_with_dictionary(context)
            self.createProblem_dict(V_nodes)
            self.set_objective_dict(cost_d, cost_e, V_nodes, nodes_dict)
            self.apply_constraints_dict(self.config['enabled_constraints'], V_nodes, nodes_dict, cost_d, cost_e, R_points)
            self.run()
            self.create_solution_dict(V_nodes, nodes_dict)
        
        else: 

            cost_d, cost_e, R_points, V_nodes, V = self.process_without_dictionary(context)
            self.createProblem(V_nodes)
            self.set_objective(cost_d, cost_e, V_nodes,V)
            self.apply_constraints(self.config['enabled_constraints'], V_nodes,V, cost_d, cost_e, R_points)
            self.run()
            self.create_solution(V_nodes, V)


    def validation(self): 
        total_steps = 4 
        with tqdm(total=total_steps, desc="Validation", unit="step") as pbar:
            
            context = self.extract_content()

            nodes = context['area_ids']
            cost_d = context['dists']
            cost_e = context['ees'] 
            R_points = list(context['R'])
            
            V_nodes = list(range(len(nodes)))
            V = {V_nodes[i] : nodes[i] for i in range(len(nodes))}
            pbar.update(1)
            if self.config['genetic_algorithm']: 
                logging.info("Genetic Algorithm Enabled...")
                self.initial_population = self.call_genetic_algorithm(V, cost_d)
            
            self.createProblem(V_nodes)
            self.set_objective(cost_d, cost_e, V_nodes,V)
            
            self.apply_constraints(self.config['enabled_constraints'], V_nodes, V, cost_d, cost_e, R_points)
            self.get_statistics()

            memory_usage = self.get_memory_usage()
            print(f"Memory Usage: {memory_usage:.2f} MB")
            pbar.update(1)
            self.run()
            pbar.update(1)
            self.create_solution(V_nodes, V)
            pbar.update(1)
            pdb.set_trace()


    def run(self): 
        try:
            self.solve_problem()
        except timeout_decorator.TimeoutError:
            print("Time limit exceeded")
            exit(0)
        
    @profile
    def run_model(self,data): 
        """
        Runs the optimization model based on the provided data.
        
        If the 'regionalization' configuration is enabled,
        the method creates a GeoDataset from the data,
        optimizes the regions, and then runs the optimization for each cluster.
        
        If the 'regionalization' configuration is disabled,
        the method calls the 'validation' method to run the optimization.
        """
                
        if self.config['regionalization']:
            gdf = self.createGeoDataset(data)
            logging.info("Regionalization initiation...")
            clusters = self.regionOptimize(gdf)
            logging.info("Running optimization...")
            total_steps = len(clusters) 
            with tqdm(total=total_steps, desc='Processing', unit="step") as pbar: 
                for cluster in clusters: 
                    self.clustering(cluster)
                    pbar.update(1)
            
        else: 
            logging.info("Validation Execution...")
            self.validation()

                        
    def apply_constraints(self, constraints_to_apply, V_nodes,ordered_nodes, cost_d, cost_e, R_points): 

        for key in constraints_to_apply:
            if key in self.constraints:
                self.constraints[key](V_nodes,ordered_nodes, cost_d, cost_e, R_points)  # Calling the constraint method directly


    def apply_constraints_dict(self, constraints_to_apply, V_nodes, ordered_nodes, cost_d, cost_e, R_points): 

        for key in constraints_to_apply:
            if key in self.constraints:
                self.constraints[key](V_nodes,ordered_nodes, cost_d, cost_e, R_points)  # Calling the constraint method directly


    def add_constraint_1(self, V_nodes,ordered_nodes, cost_d, cost_e, R_points): 

        for k in self.agents: 
            axx = [i for i,value in enumerate(ordered_nodes.values()) if value == self.depots_for_agents[k]]
            d = axx[0]
            for j in V_nodes: 
                if j != d: 
                    self.model += lpSum(self.x[i,j,k] for i in V_nodes if i != j) == 1, f"Enter_{j}_by_{k}"
                    self.model += lpSum(self.x[j,i,k] for i in V_nodes if i != j) == 1, f"Exit_{j}_by_{k}"


    def add_constraint_1_dict(self, V_nodes, ordered_nodes, cost_d, cost_e, R_points): 
        for k in self.agents: 
            axx = [i for i,value in enumerate(ordered_nodes.values()) if value == self.depots_for_agents[k]]
            d = axx[0]
            for j in V_nodes: 
                if j != d: 
                    self.model += lpSum(self.x[i,j,k] for i in V_nodes if i != j) == 1, f"Enter_{j}_by_{k}"
            for i in V_nodes: 
                if i != d:
                    self.model += lpSum(self.x[i,j,k] for j in V_nodes if i != j) == 1, f"Exit_{i}_by_{k}"


    def add_constraint_2(self, V_nodes,ordered_nodes, cost_d, cost_e, R_points): 

        for k in self.agents: 
            axx = [i for i,value in enumerate(ordered_nodes.values()) if value == self.depots_for_agents[k]]
            d = axx[0]
            for i in V_nodes[1:]: 
                self.model += self.x[i, d, k] + self.x[d,i,k] <=1

        for k in self.agents:
            axx = [i for i,value in enumerate(ordered_nodes.values()) if value == self.depots_for_agents[k]]
            d = axx[0]
            for i in V_nodes: 
                for j in V_nodes: 
                    if i!=j and i != d and j != d : 
                        self.model +=  self.u[i, k] - self.u[j, k] + len(V_nodes) * self.x[i, j, k] <= len(V_nodes) - 1 

        for k in self.agents: 
            for i in V_nodes: 
                if i != V_nodes: 
                    self.model += self.u[i,k] >= 1 
                    self.model += self.u[i,k] <= len(V_nodes) - 1
    

    def add_constraint_2_dict(self, V_nodes, ordered_nodes, cost_d, cost_e, R_points): 


        for i in V_nodes[1:]: 
            for j in V_nodes[1:]: 
                if i != j : 
                    for k in self.agents: 
                        axx = [i for i,value in enumerate(ordered_nodes.values()) if value == self.depots_for_agents[k]]
                        d = axx[0]
                        if i != j:
                            self.model += self.u[i,k] - self.u[j,k] + len(V_nodes) * self.x[i, j, k] <= len(V_nodes) - 1, f"Subtour Elimination_{i}_{j}_{k}"

        for k in self.agents: 
            axx = [i for i,value in enumerate(ordered_nodes.values()) if value == self.depots_for_agents[k]]
            d = axx[0]
            self.model += lpSum(self.x[d, j, k] for j in V_nodes if j != d) == 1, f"Agent_start_{d}_by_{k}"
        
        for k in self.agents: 
            self.model += lpSum(self.x[i, d, k] for i in V_nodes if i !=d)==1, f"Agent_end_{d}_by_{k}"

        for k1 in self.agents:
            for k2 in self.agents: 
                if k1 < k2 : 
                    self.model += lpSum(self.x[i,j,k1] - self.x[i,j,k2] for i in V_nodes for j in V_nodes if i != j ) != 0, f"Agent_order_{k1}_{k2}"
    

    def add_constraint_3(self, V_nodes,ordered_nodes, cost_d, cost_e, R_points): 
        for j in V_nodes : 
            self.model += lpSum(self.x[i,j,k] for i in V_nodes if i!=j for k in self.agents) <= R_points[j] , f'Min_visits_{j}' #Constraint 2 For Visits
    

    def add_constraint_3_dict(self, V_nodes, ordered_nodes, cost_d, cost_e, R_points): 
        for j in V_nodes : 
            self.model += lpSum(self.x[i,j,k] for i in V_nodes if i!=j for k in self.agents) <= R_points[ordered_nodes[j]] , f'Min_visits_{j}' #Constraint 2 For Visits


    def add_constraint_4(self, V_nodes,ordered_nodes, cost_d, cost_e, R_points): 
        reverse_dict = {v:k for k,v in ordered_nodes.items()}
        for step in self.TimeFrame:
            self.model += lpSum(self.t[i,j,k,step] for k in self.agents for i in V_nodes for j in V_nodes if i!=j and j != reverse_dict[self.depots_for_agents[k]]) <= 1,f"Single_Agent_at_at_time_{step}"
    

    def add_constraint_4_dict(self, V_nodes, ordered_nodes, cost_d, cost_e, R_points): 
        reverse_dict = {v:k for k,v in ordered_nodes.items()}
        
        for step in self.TimeFrame[self.moment:self.moment+len(V_nodes)]:
            self.model += lpSum(self.t[i,j,k,step] for k in self.agents for i in V_nodes for j in V_nodes if i!=j and j !=reverse_dict[self.depots_for_agents[k]]) <= 1,f"Single_Agent_at_at_time_{step}"
        
        for step in self.TimeFrame[self.moment:self.moment+len(V_nodes)]: 
            for j in V_nodes : 
                self.model += lpSum(self.t[i,j,k,step] for k in self.agents for i in V_nodes if i!=j ) <= 1,f"Entry_Agent_at_at_time_{j}_{step}"
                self.model += lpSum(self.t[j,i,k,step] for k in self.agents for i in V_nodes if i!=j ) <= 1,f"Exit_Agent_at_at_time_{j}_{step}"
        
        for step in self.TimeFrame[self.moment:self.moment + len(V_nodes)]:
            for i in V_nodes: 
                for j in V_nodes: 
                    if i!=j: 
                        for k in self.agents: 
                                self.model += self.u[j,k] >= self.u[i,k] + 1 -len(V_nodes)*(1 - self.t[i,j,k,step])


    def add_constraint_5(self, V_nodes,ordered_nodes, cost_d, cost_e, R_points): 

        M = 25
        for k in self.agents:
            axx = [i for i,value in enumerate(ordered_nodes.values()) if value == self.depots_for_agents[k]]
            d = axx[0] 
            self.model += self.e[d,k] >= self.max_battery - M * (lpSum(self.z[d,k])), f"EnergyFull_at_start_{d}_{k}"

        for k in self.agents: 
            axx = [i for i,value in enumerate(ordered_nodes.values()) if value == self.depots_for_agents[k]]
            d = axx[0]
            for i in V_nodes: 
                for j in V_nodes: 
                    if i!=j: 
                        self.model += self.e[j,k] >= self.e[i,k] - (cost_e[i][ordered_nodes[j]-1] * self.x[i,j,k]),f"Update_remaining_energy_{i}_{j}_{k}"
                        self.model += self.e[i,k] >= cost_e[i][ordered_nodes[j]-1] * self.x[i,j,k],f"No_travel_if_low_energy{i}_{j}_for_{k}"
                if i == d: 
                    self.model += self.e[j,k] == self.max_battery - M * (lpSum(self.z[d,k])), f"EnergyFull_at_end_{i}_{k}"


    def add_constraint_5_dict(self, V_nodes,  ordered_nodes,cost_d, cost_e, R_points): 

        M = 25
        for k in self.agents:
            axx = [i for i,value in enumerate(ordered_nodes.values()) if value == self.depots_for_agents[k]]
            d = axx[0]
            self.model += self.e[d,k] >= self.max_battery - M * (lpSum(self.z[d,k])), f"EnergyFull_at_start_{d}_{k}"

        for k in self.agents: 
            axx = [i for i,value in enumerate(ordered_nodes.values()) if value == self.depots_for_agents[k]]
            d = axx[0]
            for i in V_nodes: 
                for j in V_nodes: 
                    if i!=j: 
                        self.model += self.e[j,k] >= self.e[i,k] - (cost_e[ordered_nodes[i]][j] * self.x[i,j,k]),f"Update_remaining_energy_{i}_{j}_{k}"
                        self.model += self.e[i,k] >= cost_e[ordered_nodes[i]][j] * self.x[i,j,k],f"No_travel_if_low_energy{i}_{j}_for_{k}"
                if i == d: 
                    self.model += self.e[j,k] == self.max_battery - M * (lpSum(self.z[d,k])), f"EnergyFull_at_end_{i}_{k}"



    def add_constraint_6(self, V_nodes, ordered_nodes, cost_d, cost_e, R_points): 

        depot_values = [self.depots_for_agents[key] for key in self.depots_for_agents.keys()]
        depot_values = set(depot_values)
        for k in self.agents: 
            axx = [i for i,value in enumerate(ordered_nodes.values()) if value == self.depots_for_agents[k]]
            d = axx[0]
            self.model += lpSum(self.x[d,j,k] for j in V_nodes if ordered_nodes[j] not in self.depots_for_agents) ==1, f"Start_at_depot_{k}"
            self.model += lpSum(self.x[j,d,k] for j in V_nodes if ordered_nodes[j] not in self.depots_for_agents) ==1, f"Finish_at_depot_{k}"
            self.model += self.x[d,d,k] == 0, f"No_loop_at_depot_{k}_{d}"
            for i in depot_values:
                if i != d: 
                    axx = [ind for ind,value in enumerate(ordered_nodes.values()) if i == value]
                    ind = axx[0]
                    self.model += self.x[d,ind,k] == 0, f"No_travel_from_{i}_at_depot_{d}_for_{k}"
                    self.model += self.x[ind,d,k] == 0, f"No_travel_to_{i}_at_depot_{d}_for_{k}"
                    self.model += self.x[d,ind,k] + self.x[ind,d,k] <= 0, f"No_travel_{i}_at_{d}_for_{k}"



    def add_constraint_6_dict(self, V_nodes,  ordered_nodes,cost_d, cost_e, R_points): 
        
        depot_values = [self.depots_for_agents[key] for key in self.depots_for_agents.keys()]
        depot_values = set(depot_values)
        for k in self.agents: 
            axx = [i for i,value in enumerate(ordered_nodes.values()) if value == self.depots_for_agents[k]]
            d = axx[0]
            self.model += lpSum(self.x[d,j,k] for j in V_nodes if ordered_nodes[j] not in self.depots_for_agents) ==1, f"Start_at_depot_{k}"
            self.model += lpSum(self.x[j,d,k] for j in V_nodes if ordered_nodes[j] not in self.depots_for_agents) ==1, f"Finish_at_depot_{k}"
            self.model += self.x[d,d,k] == 0, f"No_loop_at_depot_{k}_{d}"
            for i in depot_values:
                if i != d: 
                    axx = [ind for ind,value in enumerate(ordered_nodes.values()) if i == value]
                    ind = axx[0]
                    self.model += self.x[d,ind,k] == 0, f"No_travel_from_{i}_at_depot_{d}_for_{k}"
                    self.model += self.x[ind,d,k] == 0, f"No_travel_to_{i}_at_depot_{d}_for_{k}"
                    self.model += self.x[d,ind,k] + self.x[ind,d,k] <= 0, f"No_travel_{i}_at_{d}_for_{k}"



    def add_constraint_7(self, V_nodes,ordered_nodes, cost_d, cost_e, R_points): 
        depot_values = [self.depots_for_agents[key] for key in self.depots_for_agents.keys()]
        depot_values = set(depot_values)
        
        for k in self.agents: 
            for step in self.TimeFrame[self.moment:self.moment+len(V_nodes)]: 
                    axx = [i for i,value in enumerate(ordered_nodes.values()) if value == self.depots_for_agents[k]]
                    d = axx[0]
                    reverse = {i for i,value in enumerate(ordered_nodes.values()) if value in depot_values}

                    self.model += self.t[d,d,k,step] == 0 
                    self.model += self.t[d,d,k,step] + self.t[d,d,k,step+1] <= 1, f"No_loop_at_depot_{d}_{step}_{k}"
                    self.model += lpSum(self.t[d,j,k,step] for j in reverse) == 0, f"No_unnecessary_visits_at_depot_{d}_{step}_{k}"
        self.moment += len(V_nodes)


    def add_constraint_7_dict(self, V_nodes,  ordered_nodes, cost_d, cost_e, R_points): 
        depot_values = [self.depots_for_agents[key] for key in self.depots_for_agents.keys()]
        depot_values = set(depot_values)
        
        for k in self.agents: 
            for step in self.TimeFrame[self.moment:self.moment+len(V_nodes)]: 
                    axx = [i for i,value in enumerate(ordered_nodes.values()) if value == self.depots_for_agents[k]]
                    d = axx[0]
                    reverse = {i for i,value in enumerate(ordered_nodes.values()) if value in depot_values}
                    self.model += self.t[d,d,k,step] == 0 
                    self.model += self.t[d,d,k,step] + self.t[d,d,k,step+1] <= 1, f"No_loop_at_depot_{d}_{step}_{k}"
                    self.model += lpSum(self.t[d,j,k,step] for j in reverse ) == 0, f"No_unnecessary_visits_at_depot_{d}_{step}_{k}"
        self.moment += len(V_nodes)


    def add_constraint_8(self, V_nodes, ordered_nodes, cost_d, cost_e, R_points):

        if self.config['genetic_algorithm']:
            if self.initial_population is not None:
                #Set initial solution based on the GA 
                for a in range(len(self.initial_population)): 
                        for i in range(len(self.initial_population[a])-1): 
                            self.x[self.initial_population[a][i],self.initial_population[a][i+1],a+1].setInitialValue(1)


    def add_constraint_8_dict(self, V_nodes, ordered_nodes, cost_d, cost_e, R_points):

        if self.config['genetic_algorithm']:
            if self.initial_population is not None:
                #Set initial solution based on the GA 
                for a in self.agents: 
                        for i in range(len(*self.initial_population[a])-1): 
                            self.x[self.initial_population[a][0][i],self.initial_population[a][0][i+1],a].setInitialValue(1)


    def add_constraint_9(self, V_nodes,ordered_nodes, cost_d, cost_e, R_points):
        reverse_nodes = {v:k for k,v in ordered_nodes.items()}
        for k in self.agents: 
            for j in self.customers:
                if j[0] in ordered_nodes.values():
                    self.model += lpSum(self.x[i,reverse_nodes[j[0]],k] for i in V_nodes ) >= self.z[reverse_nodes[j[0]],k], f"Service_constraint{j[0]}_for_{k}"
 

    def add_constraint_9_dict(self, V_nodes,ordered_nodes, cost_d, cost_e, R_points):
        reverse_nodes = {v:k for k,v in ordered_nodes.items()}
        for k in self.agents: 
            for j in self.customers:
                if j[0] in ordered_nodes.values():
                    self.model += lpSum(self.x[i,reverse_nodes[j[0]],k] for i in V_nodes ) >= self.z[reverse_nodes[j[0]],k], f"Service_constraint{j[0]}_for_{k}"


    def create_solution(self, V_nodes, V): 
        reverse_dict = {v: k for k, v in V.items()}

        if pl.LpStatus[self.model.status] == 'Optimal':
            for k in self.agents: 
                start_node = self.depots_for_agents[k]
                current_node = start_node
                route = [start_node]

                while True: 
                    next_node = None 
                    for j in V_nodes: 
                        if V[j] != current_node and self.x[reverse_dict[current_node], j, k].varValue == 1:
                            next_node = V[j]
                            break
                    
                    if next_node is None :
                        break
                    
                    if next_node == start_node:
                        route.append(start_node)
                        break
                    
                    route.append(next_node)
                    current_node = next_node
                
                self.paths[k].append(route)
            print("Optimal solution found.")
            memory_usage = self.get_memory_usage()
            print(f"Memory Usage: {memory_usage:.2f} MB")
            

    def create_solution_dict(self, V_nodes, V): 
        reverse_dict = {v: k for k, v in V.items()}
        if pl.LpStatus[self.model.status] == 'Optimal':
            for k in self.agents: 
                start_node = self.depots_for_agents[k]
                current_node = start_node
                route = [start_node]

                while True: 
                    next_node = None 
                    for j in V_nodes:
                        if V[j] != current_node and self.x[reverse_dict[current_node], j, k].varValue == 1:
                            next_node = V[j]
                            break
                    
                    if next_node is None :
                        break
                    
                    if next_node == start_node:
                        route.append(start_node)
                        break
                    
                    route.append(next_node)
                    current_node = next_node
                self.paths[k].append(route)
            print("Optimal solution found.")
            memory_usage = self.get_memory_usage()
            print(f"Memory Usage: {memory_usage:.2f} MB")
            

    def get_solution(self) -> list: 

        for k in self.paths.keys():
                tmp = [] 
                for tour in range(len(self.paths[k])):
                    tmp.extend(self.paths[k][tour])
                self.paths[k] = []
                self.paths[k].append(tmp)

        for k in self.paths:
            print(f"Agent {k} path: {' -> '.join(map(str, self.paths[k]))}")
            tmp = len(set(self.paths[k][0]))
            print(tmp == self.v, tmp)
            if tmp != self.v: 
                print(sorted(set(self.paths[k][0])))

        return self.paths 


    def get_performance(self): 

        end_time = time.time()
        execution_time = end_time - self.start_time
        print(f" Program Completion Time: {execution_time}s")


    def get_memory_usage(self): 

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 ** 2) 


    def get_statistics(self): 
        
        print(f"Number of Variables: {len(self.model.variables())}")
        print(f"Number of Constraints: {len(self.model.constraints)}") 