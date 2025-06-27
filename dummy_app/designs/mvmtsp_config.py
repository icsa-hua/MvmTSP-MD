from abc import ABC, abstractmethod
from dummy_app.tools.logger import logger
from dummy_app.tools.common import deallocate_memory
from dummy_app.models.genetic_algorithm import GASolution
from dummy_app.models.topsis import TOPSISPriority
from dummy_app.models.energy_model import DroneEnergyModel

import os
import joblib
import geopandas 
import pandas as pd 
import numpy as np 
import pulp as pl 
import timeout_decorator 
import resource
import random 
import networkx as nx

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler 
from collections import defaultdict
from k_means_constrained import KMeansConstrained
from typing import Any, Union, List, Dict, Mapping, Tuple, Optional



class MVMTSPConfig(ABC): 

    @abstractmethod
    def __init__(self, env_type:str, max_battery:int, max_coverage_time:int, enable_ga:bool, scenario:str, objective_function:str, stage_solution:int, priority:str,validate:bool )->None: 

        self.problem = pl.LpProblem() 
        self.V:pd.DataFrame = pd.DataFrame()
        self.agents:List[int] = [] 
        self.travel_cost:np.ndarray = np.empty((0,0))
        self.distance_columns:List[str] = []
        self.energy_columns:List[str] = []
        self.travel_time_columns:List[str] = []
        self.move_energy:np.ndarray = np.ndarray((0,0))
        self.average_energy:float = 0.0
        self.depots:Optional[np.ndarray] = None
        self.average_coverage_energy:float=0.0
        self.user_points = defaultdict(list)  # User points for regionalization
        self.scenario:str = scenario
        self.ascend_energy = pd.DataFrame
        self.descend_energy= pd.DataFrame
        self.max_battery:int = max_battery
        self.coverage_time = max_coverage_time
        self.enable_ga = enable_ga
        self.objective_function:str = objective_function
        self.env_type:str = env_type
        self.stage_solution:int = stage_solution
        self.priority = priority 
        self.validate = validate
        

    @abstractmethod
    def assign_agents_to_areas(self, plethos:int, depots:Any)->Dict[int,int]:
        """ 
            Assign agents randomly and equally to depot areas.

            Args:
                num_agents (int): Total number of agents.
                depots (list): List of depot areas.

            Returns:
                dict: Mapping of each agent to their assigned area.
            """
        if plethos <= 0 or depots is None: 
            logger.error("Number of agents and depots must be specified")
            raise ValueError("Number of agents and depots must be specified")


        random.shuffle(self.agents)
        base_agents_per_area = plethos // len(depots) 
        extra_agents = plethos % len(depots) 

        assignments = {} 
        agent_index = 0 

        for depot in depots: 
            n_assigned = base_agents_per_area + (1 if extra_agents > 0 else 0)

            for _ in range(n_assigned): 
                if agent_index < plethos: 
                    assignments[self.agents[agent_index]] = depot 
                    agent_index += 1 

            if extra_agents > 0 : 
                extra_agents -= 1

        return assignments 
    

    @abstractmethod 
    def createGeoDataset(self, data:pd.DataFrame)->geopandas.GeoDataFrame:

        if data is None:
            logger.error("Data is not initialized")
            raise ValueError("Data is not initialized")
        return geopandas.GeoDataFrame(data, geometry=geopandas.points_from_xy(data.X_coords,data.Y_coords), crs="EPSG:4326")


    @abstractmethod
    def set_memory_limit(self, max_memory:int=1024)->None:
        resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))


    @abstractmethod
    def preprocess_generated_data(
            self, 
            distance_matrix:np.ndarray, 
            centroids:list,   
            depots:np.ndarray, 
            num_of_agents:int, 
            v_ver:float, 
            v_hor:float, 
            altitude:int, 
            coverage_time:int,
            user_points=defaultdict(), 
    )->pd.DataFrame:
        
        def normalize_data(df:pd.DataFrame, name:str='')->pd.DataFrame:
            scalers_path = f"{os.getcwd()}/assets/scalers"
            scaler_file_name = f"scaler_{name}.pkl"
            if not os.path.exists(scalers_path):
                os.mkdir(scalers_path)
            scaler_file_path = os.path.join(scalers_path,scaler_file_name)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df.values)
            joblib.dump(scaler, scaler_file_path) 
            
            return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

        data_path = f"{os.getcwd()}/assets/data"
        if not os.path.exists(data_path):
            os.mkdir(data_path)


        distances = pd.DataFrame(distance_matrix, columns=[f'dist_{i}' for i in range(1, len(distance_matrix)+1)])
        distances.to_csv(f"{data_path}/distances.csv")

        energy_model = DroneEnergyModel(
            v_hor=v_hor, 
            v_ver=v_ver,
            max_battery=self.max_battery)
        
        energy_matrix = np.ndarray(distance_matrix.shape)
        
        for i in range(len(distance_matrix)):
            for j in range(len(distance_matrix)):
                energy_matrix[i,j] = energy_model.move_energy(current_node=i,next_node=j,distance_matrix=distance_matrix)
        logger.debug(f"Average_energy expend for Move: {energy_matrix.mean()}")

        # NOTE: Energies here are in JOULE 
        energies = pd.DataFrame(energy_matrix, columns=[f'ee_{i}' for i in range(1, len(energy_matrix)+1)])

        # NOTE: To convert it to Wh 
        energies = energies / 3600.0
        energies.to_csv(f"{data_path}/energies.csv")
        self.coverage_time = coverage_time 
        self.move_energy = energies.values.astype(np.float32)
        self.average_coverage_energy = energy_model.coverage_energy(altitude, 1) # In J for a single time step 
        self.average_coverage_energy = self.average_coverage_energy / 3600.0  # Convert to Wh

        logger.debug(f"Average coverage energy: {self.average_coverage_energy} Wh")

        area_ids = list(user_points.keys()) 
        centroids_x = [centroids[i][0] for i in range(len(centroids))]
        centroids_y = [centroids[i][1] for i in range(len(centroids))]
        nodes = {
            'X_coords': centroids_x, 
            'Y_coords': centroids_y, 
            'Area_id': area_ids
        }

        nodes = pd.DataFrame(nodes)
        self.V = nodes 
        self.v = len(self.V)
        self.agents = list(range(1,num_of_agents+1))
        self.average_energy = float(np.average(energies))

        # Calculate Average Ascend and Descend 
        ascend_energy = defaultdict(list)
        descend_energy = defaultdict(list)
        for depot in depots: 
            for i in self.V['Area_id']: 
                
                ascend_energy[(int(depot))].append(np.float32((energy_model.ascend_energy(
                    current_node=depot, 
                    next_node=i,
                    altitude=altitude,
                    distance_matrix=distance_matrix
                )) / 3600))

                descend_energy[(int(depot))].append(np.float32((energy_model.descend_energy(
                    current_node=i, 
                    next_node=depot, 
                    altitude=altitude, 
                    distance_matrix=distance_matrix
                )) / 3600))
        
        self.ascend_energy = pd.DataFrame.from_dict(ascend_energy).T
        self.descend_energy = pd.DataFrame.from_dict(descend_energy).T

        # Here distance must be in meters
        travel_times = (distance_matrix * 1e3) / v_hor / 60.0  # Convert to minutes
        travel_times = pd.DataFrame(travel_times, columns=[f'tt_{i}' for i in range(1, len(travel_times)+1)])
        self.travel_cost = travel_times.values
        travel_times.to_csv(f'{data_path}/times.csv')
        assert distances.shape == energies.shape == travel_times.shape, "Distances, energies, and travel times must have the same shape"

        distances = normalize_data(distances,name='distance')
        energies = normalize_data(energies, name='energy')
        travel_times = normalize_data(travel_times, name='travel_time')

        self.distance_columns = distances.columns.tolist()
        self.energy_columns = energies.columns.tolist()
        self.travel_time_columns = travel_times.columns.tolist()

        self.depots = depots 
        # combine al normalized data

        data = pd.concat([distances, energies, travel_times, nodes], axis=1, join='inner')
        return data 


    @abstractmethod 
    @timeout_decorator.timeout(3600)
    def solve_problem(self, cluster:Any)->None: 
        # self.problem.solve(pl.GLPK_CMD(msg=False, options=['--mipgap', '0.05']))
        pass


    @abstractmethod 
    def call_genetic_algorithm(self, nodes_dict:Dict[int,int], cost:Dict[str,float], depot:int, verbose:bool=False, population_size:int=200, generations:int=100)->Tuple[List[int],Any]: 
        
        ga = GASolution(
            population=population_size, 
            generations=generations, 
            nodes_dict=nodes_dict,
            depot=depot
        )

        best_paths, hof = ga.run(
            crossover_rate=0.7, 
            mutation_rate=0.05,
            cost=cost, 
            enable_indi_fitness=True, 
            verbose=verbose
        )

        logger.debug(f"Best Paths: {best_paths} with depot {depot}")
        return best_paths, hof


    @abstractmethod
    def run_model(self, distance_matrix:np.ndarray, data:pd.DataFrame, cue_groups:Dict[int,List[Any]])->Dict:
        pass 


    @abstractmethod 
    def regionalization(self, GDF:geopandas.GeoDataFrame)->Any:
        """
        Cluster nodes (excluding depots) into constrained regions based on agent capacity.

        Args:
            GDF (pd.DataFrame): Geospatial or feature DataFrame of all nodes.

        Returns:
            clusters (pd.core.groupby.generic.DataFrameGroupBy): Grouped clusters by label.
        """

        # Determine the maximum number of nodes per cluster based on battery

        # TODO: Calculate Maximum nodes based on Hover.
        max_nodes = 0 
        
        # Reserve 10–15% for emergency return
        if self.scenario == 'cooperative': 
            reserve = self.max_battery * 0.35
        elif self.scenario == 'individual': 

            reserve = self.max_battery * 0.60
        else: 
            reserve = self.max_battery * 0.30

        adjusted_energy = self.average_energy + self.average_coverage_energy * self.coverage_time
        
        if self.scenario == 'cooperative':
            max_nodes = int((self.max_battery-reserve) / adjusted_energy) - 2
        elif self.scenario == 'individual':
            max_nodes = int((self.max_battery-reserve) / adjusted_energy) - 3
            if max_nodes == 0: raise ValueError("Insufficient battery capacity for the given coverage time and coverage energy.")
        else:
            max_nodes = int((self.max_battery-reserve) / adjusted_energy) - 2

        logger.debug(f"Maximum nodes per cluster based on battery: {max_nodes}")
        # charge_points = int(np.floor(self.v/max_nodes))

        if hasattr(self, 'depots') and self.depots is not None:
            non_depot_gdf = GDF[~GDF['Area_id'].isin(self.depots)].copy() 
        else: 
            non_depot_gdf = GDF.copy() 

        features = pd.concat([
            non_depot_gdf[self.distance_columns], 
            non_depot_gdf[self.energy_columns],
            non_depot_gdf[self.travel_time_columns]
        ], axis=1)
        
        # n_clusters = len(self.agents) # NOTE: Why is this n_clusters = 4 e.g.? 
        # Determine total demand (total nodes to cover)
        total_nodes = len(GDF)
        n_clusters = int(np.ceil(total_nodes / max_nodes))

        logger.info(f"Total Nodes: {total_nodes}, Clusters: {n_clusters}")
        
        kmeans = KMeansConstrained(
            n_clusters=n_clusters, 
            size_min=2, 
            size_max=max_nodes,
            random_state=42
        )
        cluster_labels = kmeans.fit_predict(features)
        
        # Update GDF with cluster labels (only for non-depots)
        non_depot_gdf['cluster'] = cluster_labels

        # Merge cluster labels back to full GDF
        GDF = GDF.copy() 
        GDF = GDF.merge(non_depot_gdf[['Area_id','cluster']], on='Area_id', how='left')

        GDF['cluster'] = GDF['cluster'].fillna(-1).astype(int)  # Fill NaN with -1

        clusters = GDF.groupby('cluster')

        logger.debug("Region Clustering Complete...")

        return clusters


    @abstractmethod
    def cluster_prioritization(self, clusters:Any, cue_groups:Mapping[int,Any], distance_matrix)->pd.DataFrame:
        topsis = TOPSISPriority()
        cluster_criteria = {} 
         
        for cluster_id, cluster_df in clusters:
            cluster_criteria[cluster_id] = topsis.gather_criteria(cluster_df, cue_groups=cue_groups, distance_matrix=distance_matrix)

        priority = topsis.run_model(cluster_criteria, np.ndarray(0))
        logger.debug("Cluster Prioritization (TOPSIS) Complete...")
        
        return priority
    

    @abstractmethod
    def clustering(self, cluster:pd.DataFrame, cluster_id:int, assignment:List[int], depot_id:int)->Dict: 
        pass 



     


        





