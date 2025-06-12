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
from typing import Any, Union, List, Dict, Mapping, Tuple, Optional
import pulp as pl 
import timeout_decorator 
import resource
import random 
import networkx as nx
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler 
from collections import defaultdict
import pdb
from k_means_constrained import KMeansConstrained



class MVMTSPConfig(ABC): 

    @abstractmethod
    def __init__(self)->None: 

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
        self.distance_metric:str = "euclidean"
        self.average_coverage_energy:float=0.0
        self.user_points:Dict[int,Tuple[float,float]] = {}  # User points for regionalization
        self.scenario:str = ""
        self.ascend_energy = pd.DataFrame
        self.descend_energy= pd.DataFrame
        self.coverage_time = 0 
      

    @abstractmethod
    def assign_agents_to_areas(self, plethos:int=0, depots:List[int]=[])->Dict[int,int]:
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
    def preprocess(
        self,
        distances_path:Union[str,Path],
        energies_path:Union[str,Path], 
        nodes_path:Union[Path, str], 
        num_of_agents:int, 
        v_ver:float, 
        v_hor:float, 
        customers_path:Union[Path,str], 
        user_points:Any, 
        max_battery:int, 
        altitude:int
    )->pd.DataFrame:

        def normalize_data(df:pd.DataFrame)->pd.DataFrame:
            
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df.values)
            
            return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
        
        def ensure_str_path(path:Union[Path,str])->str:
            return str(path) if isinstance(path, Path) else path
        
        # Load Data 
        distances = pd.read_csv(ensure_str_path(distances_path))
        energies = pd.read_csv(ensure_str_path(energies_path))
        nodes = pd.read_csv(ensure_str_path(nodes_path))
        customers = pd.read_csv(ensure_str_path(customers_path))

        # Assign Nodes and Agents 
        self.V = nodes 
        self.v = len(self.V)
        
        self.agents = list(range(1,num_of_agents+1))
        self.max_battery = max_battery

        energy_model = DroneEnergyModel()
        
        self.average_coverage_energy = energy_model.coverage_energy(1250) # In J 
        self.average_coverage_energy = self.average_coverage_energy / 3600.0  # Convert to Wh
        self.normalized_coverage_energy = self.average_coverage_energy / max_battery  # Normalize coverage energy
        logger.debug(f"Average coverage energy: {self.average_coverage_energy} Wh")


        # Prepare Matrices 
        dist_columns = [f'dist_{i}' for i in range(1, self.v + 1)]
        energy_columns = [f'ee_{i}' for i in range(1, self.v + 1)]
        tt_columns = [f'tt_{i}' for i in range(1, self.v + 1)] 

        distances.columns = dist_columns 
        energies.columns = energy_columns

        self.average_energy = float(np.average(energies))
        
        # Velocity in m/s
        velocity = 5.5555555555555 
        travel_times = distances/velocity/60
        travel_times.columns = tt_columns 
        self.travel_cost = travel_times.values
        # Assert matrix shapes 
        assert distances.shape == energies.shape == travel_times.shape, "Distances, energies, and travel times must have the same shape"

        # Normalize matrices 
        distances = normalize_data(distances)
        energies = normalize_data(energies)
        travel_times = normalize_data(travel_times)

        # Setup visits allowed 
        # self.allowed_visits = np.full(self.v, len(self.agents), dtype=int)

        # Final preperation 
        self.distance_columns = dist_columns 
        self.energy_columns = energy_columns
        self.travel_time_columns = tt_columns
        

        if self.v >= 10: 
            self.depots = self.V['Area_id'].iloc[np.array([7, 8])].values
        else: 
            raise ValueError("Not enough nodes to select default depots at positions 7 and 8.")
        self.distance_metric = "euclidean"
        # combine al normalized data
        data = pd.concat([distances, energies, travel_times, nodes], axis=1, join='inner')
        return data 


    def preprocess_generated_data(
            self, 
            distance_matrix:np.ndarray, 
            centroids:list, 
            user_points:dict, 
            depots:np.ndarray, 
            num_of_agents:int, 
            v_ver:float, 
            v_hor:float, 
            max_battery:float, 
            altitude:int, 
            coverage_time:int
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

        self.max_battery = max_battery

        distances = pd.DataFrame(distance_matrix, columns=[f'dist_{i}' for i in range(1, len(distance_matrix)+1)])
        distances.to_csv(f"{data_path}/distances.csv")

        energy_model = DroneEnergyModel(
            v_hor=v_hor, 
            v_ver=v_ver,
            max_battery=max_battery)
        
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
        self.average_coverage_energy = energy_model.coverage_energy(altitude, self.coverage_time) # In J 
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

        distances = normalize_data(distances,name='distance_km')
        energies = normalize_data(energies, name='energies_Wh')
        travel_times = normalize_data(travel_times, name='times_min')

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
    def run_model(self, data:pd.DataFrame,  cue_groups:Dict[int,List[Any]])->Dict: 
        pass 


    @abstractmethod
    def create_solution(self, cluster:Any)->Dict[str,List[int]]: 
        pass 


    @abstractmethod 
    def regionalization(self, GDF:geopandas.GeoDataFrame)->pd.core.groupby.generic.DataFrameGroupBy:
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
        reserve = self.max_battery * 0.15
        adjusted_energy = self.average_energy + self.average_coverage_energy 
        
        max_nodes = int((self.max_battery-reserve) / adjusted_energy) - 1

        logger.debug(f"Maximum nodes per cluster based on battery: {max_nodes}")
        charge_points = int(np.floor(self.v/max_nodes))

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
        total_nodes = len(GDF) - len(self.depots) if self.depots is not None and len(self.depots) != 0 else len(GDF)
        n_clusters = int(np.ceil(total_nodes / max_nodes))

        logger.info(f"Total Nodes: {total_nodes}, Clusters: {n_clusters}")
        
        kmeans = KMeansConstrained(
            n_clusters=n_clusters, 
            size_min=charge_points, 
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
    def cluster_prioritization(self, clusters:pd.core.groupby.generic.DataFrameGroupBy, cue_groups:Mapping[int,Any], distance_matrix)->pd.DataFrame:
        topsis = TOPSISPriority()
        cluster_criteria = {} 
         
        for cluster_id, cluster_df in clusters:
            cluster_criteria[cluster_id] = topsis.gather_criteria(cluster_df, cue_groups=cue_groups, distance_matrix=distance_matrix)

        priority = topsis.run_model(cluster_criteria, [])
        logger.debug("Cluster Prioritization (TOPSIS) Complete...")
        
        return priority
    

    @abstractmethod
    def clustering(self, cluster:pd.DataFrame, cluster_id:int, assignment:List[int], depot_id:int)->Dict: 
        pass 



     


        





