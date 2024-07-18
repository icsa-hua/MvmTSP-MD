from abc import ABC, abstractmethod
import pandas as pd
import geopandas 
class MvmTSP(ABC): 

    @abstractmethod
    def __init__(self, data: pd.DataFrame, use_GA: bool, regionalization: bool):
        """
        Initialize the MvmTSP formulation, what components to use. 
        """
        self.model=None 
        self.data = data
        self.use_GA = use_GA
        self.regionalization = regionalization
        pass

    @abstractmethod
    def createProblem(self):
        pass

    @abstractmethod
    def set_objective(self): 
        pass

    @classmethod 
    def assign_agents_to_areas(self, num_agents: int, depots: list)->dict:
        """
        Assign the k agent to one of the depot. 
        Each agent must have only one
        """
        pass

    @abstractmethod
    def createGeoDataset(self): 
        pass

    @abstractmethod
    def set_memory_limit(self, max_memory: int): 
        pass 

    @abstractmethod
    def preprocess(self, distance: str, energy: str, areas: str, num_agents:int) -> pd.DataFrame: 
        pass

    @abstractmethod
    def solve_problem(self):
        pass

    @abstractmethod
    def call_genetic_algorithm(self, nodes:list, cost:list) -> list: 
        """
        Call the Genetic algorithm class to create the object. 
        """
        pass

    @abstractmethod
    def regionOptimize(self, GDF: geopandas.GeoDataFrame)->pd.core.groupby.generic.DataFrameGroupBy: 
        """
        Call the regionalization procedure to create the clusters
        """
        pass 

    @abstractmethod
    def apply_constraints(self,constraints_to_apply): 
        pass
    
    @abstractmethod
    def run_model(self):
        pass

    @abstractmethod
    def create_solution(self, nodes: list, Cities: dict)->list:
        pass

    @classmethod
    def get_solution(self)->list: 
        pass

    @classmethod
    def get_performance(self):
        pass