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
        """
        Create the optimization problem with the PuLP library. 
        Initialize the optimization variables (e.g. x_ij, y_ij, etc.)
        """
        pass

    @abstractmethod
    def set_objective(self): 
        """
        Set the objective function for the optimization problem.
        Utilizing the PuLP library we can add this as a constraint to 
        the formulation based on the cost valuator. 
        """
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
        """
        Create a GeoDataFrame object for regionalization. 
        """
        pass

    @abstractmethod
    def set_memory_limit(self, max_memory: int): 
        """
        Explicitly set the memory limit for the solver.
        If the memory consumed is larger than the limit, the solver will stop.
        """
        pass 

    @abstractmethod
    def preprocess(self, distance: str, energy: str, areas: str, num_agents:int) -> pd.DataFrame: 
        """
        Initialize the main entities for the optimization problem, 
        such as nodes, agents, and the distance matrix. 

        Returns:
            pd.DataFrame : A data representation of the problem in a single entity.
        """
        pass

    @abstractmethod
    def solve_problem(self):
        """
        Run the specified solver 
        """
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
        """
        Set the constraints for the optimization problem, 
        and apply them to the optimization model.
        """
        pass
    
    @abstractmethod
    def run_model(self):
        """
        Execute the problem constructor. 
        """
        pass

    @abstractmethod
    def create_solution(self, nodes: list, Cities: dict)->list:
        """
        Process and validate the solution returned by the solver.
        """
        pass

    @classmethod
    def get_solution(self)->list: 
        """
        Return the solution of the optimization problem.
        """
        pass

    @classmethod
    def get_performance(self):
        """
        Return the computational time in seconds.
        """
        pass