import networkx as nx 
from typing import Dict 


class CentralHub: 

    def __init__(self, weights)->None: 
        self.betweeness:float = 0.0  
        self.distance_centroid:float = 0.0 
        self.weights:Dict[str,float] = weights