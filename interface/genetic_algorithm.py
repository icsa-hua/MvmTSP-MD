from abc import ABC, abstractmethod
import networkx
import numpy as np
from typing import Tuple


class GA(ABC): 

    @abstractmethod
    def __init__(self, population_size: int, generations: int, nodes: list, agents:list, depots:dict): 
        """
        Initialize the GA heuristic with the given parameters
        """

        pass

    @abstractmethod
    def create_graph(self, weights) -> networkx.Graph: 
        """
        Create the graph for the nodes, with the associated weights for the edges
        """
        pass

    @abstractmethod
    def initialize_tours(self):
        """
        Modify the tours created, usually common in multiTSP
        """
        pass

    @abstractmethod
    def initialize_population(self):
        """
        Specify the initialization of population. 
        """
        pass

    @abstractmethod
    def crossover(self, ind1, ind2): 
        """
        Create custom crossover 
        """
        pass 

    @abstractmethod
    def mutation(self, individual, indpb):
        """
        Create custom mutation 
        """
        pass

    @abstractmethod
    def fitness_evaluation(self, individual, cost:np.ndarray) -> Tuple: 
        """
        Calculation of the objective to optimize (e.g. distance, energy consumption). 
        """
        pass

    @abstractmethod
    def toolbox_config(self):
        """
        Create the configuration about population, crossover, mutation
        """
        
        pass 

    @abstractmethod
    def run(self, cthr: float, mthr: float, enable_ind_fit:bool)->list: 

        """
        Execute the GA to find the optimal path
        """
        pass

        