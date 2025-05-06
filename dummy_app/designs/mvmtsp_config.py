from abc import ABC, abstractmethod
from dummy_app.tools.logger import logger
from dummy_app.models.genetic_algorithm import GASolution
from dummy_app.models.topsis import TOPSISPriority
import geopandas 
import pandas as pd 
import numpy as np 
from typing import Any, Union, List, Dict
import pulp as pl 
import timeout_decorator 
import resource
import random 
import networkx as nx
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler 
from k_means_constrained import KMeansConstrained



class MVMTSPConfig(ABC): 

    @abstractmethod
    def __init__(self, config:Dict[str, Any])->None: 

        self.problem = None 
        self.V:pd.DataFrame = None
        self.agents:List[int] = [] 
        self.moment:int = 0 
        self.paths:Dict[str, List[int]] = {}
        self.travel_cost:np.ndarray = np.empty((0,0))
        self.distance_columns:List[str] = []
        self.energy_columns:List[str] = []
        self.travel_time_columns:List[str] = []
        self.average_energy:float = 0.0
        self.customers:np.ndarray = np.empty((0,0)) 
        self.STEPS:list = [range(0,3600,1)]
        self.graph:nx.Graph = nx.Graph()

    @abstractmethod 
    def set_objective(self, alpha:Any, beta:Any, gamma:Any, weight:object):

        self.problem.setObjective(
            pl.lpSum(alpha[i][j] * weight[i,j] + beta[i][j] * weight[i,j] + gamma[i][j] * weight[i,j] for i in self.V for j in self.V)
        ) 


    @abstractmethod
    def assign_agents_to_areas(self, plethos:int=0, depots:Union[List[int], Dict[int,int]]=None)->Dict[int,int]:
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
    def preprocess(self, distances_path:Union[str,Path], energies:Union[str,Path], nodes_path:Union[Path, str], agents:int, customers_path:Union[Path,str], ground_users:Union[Path,str], max_battery:int)->pd.DataFrame:

        def normalize_data(df:pd.DataFrame)->pd.DataFrame:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df.values)
            return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
        
        def ensure_str_path(path:Union[Path,str])->str:
            return str(path) if isinstance(path, Path) else path
        
        # Load Data 
        distances = pd.read_csv(ensure_str_path(distances_path))
        energies = pd.read_csv(ensure_str_path(energies))
        nodes = pd.read_csv(ensure_str_path(nodes_path))
        customers = pd.read_csv(ensure_str_path(customers_path))

        # Assign Nodes and Agents 
        self.V = nodes 
        self.v = len(self.V)
        self.agents = list(range(1,agents+1))
        self.max_battery = max_battery 

        # Prepare Matrices 
        dist_columns = [f'dist_{i}' for i in range(1, self.v + 1)]
        energy_columns = [f'ee_{i}' for i in range(1, self.v + 1)]
        tt_columns = [f'tt_{i}' for i in range(1, self.v + 1)] 

        distances.columns = dist_columns 
        energies.columns = energy_columns
        self.average_energy = np.average(energies)
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

        # Format customers 
        self.customers = customers.to_numpy() 

        # Setup visits allowed 
        self.allowed_visits = np.full(self.v, len(self.agents), dtype=int)

        # Final preperation 
        self.distance_columns = dist_columns 
        self.energy_columns = energy_columns
        self.travel_time_columns = tt_columns
        self.paths = {agent:[] for agent in self.agents}
        

        if self.v >= 10: 
            self.depots = self.V['Area_id'].iloc[[7,8]].values.tolist()
        else: 
            raise ValueError("Not enough nodes to select default depots at positions 7 and 8.")
        
        # combine al normalized data
        data = pd.concat([distances, energies, travel_times, nodes], axis=1, join='inner')
        
        return data 


    @abstractmethod 
    @timeout_decorator.timeout(3600)
    def solve_problem(self)->None: 
        self.problem.solve(pl.GLPK_CMD(msg=False, options=['--mipgap', '0.05']))


    @abstractmethod 
    def call_genetic_algorithm(self, V_nodes:List[int], cost:Dict[str,float], depot:int, verbose:bool=False, population_size:int=200, generations:int=100)->List[int]: 
        
        ga = GASolution(
            population=population_size, 
            generations=generations, 
            nodes=V_nodes,
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
    def run(self): 

        try: 
            self.solve_problem() 
            return tuple(("Problem solved successfully", True))
        except timeout_decorator.TimeoutError:
            logger.exception("Timeout reached. Problem could not be solved within the time limit.")
            return tuple(("Timeout occurred. Exiting...", False))
        
    

    @abstractmethod
    def run_model(self, data:pd.DataFrame, enabled_constraints:List[str])->None: 
        pass 



    @abstractmethod
    def create_solution(self, nodes:List[int], V:Dict[int, int])->None: 
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
        max_nodes = int(self.max_battery / self.average_energy)
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


        n_clusters = len(self.agents)
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
    def cluster_prioritization(self, clusters:pd.core.groupby.generic.DataFrameGroupBy, cue_groups:Dict[int,object])->pd.DataFrame:
        topsis = TOPSISPriority()
        cluster_criteria = {} 
         
        for cluster_id, cluster_df in clusters: 
            cluster_criteria[cluster_id] = topsis.gather_criteria(cluster_df, cue_groups=cue_groups)

        priority = topsis.run_model(cluster_criteria, None)
        logger.debug("Cluster Prioritization (TOPSIS) Complete...")
        
        return priority
    

    @abstractmethod
    def clustering(self, cluster:pd.DataFrame, cluster_id:int, assignment:Dict[int, int])->None: 
        pass 



     


        





