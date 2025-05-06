import networkx as nx
import numpy as np 
from typing import Dict, List
from sklearn.preprocessing import MinMaxScaler

class CentralHub: 

    # NOTE: This class finds the central node for a cluster to act as a bridge or hub, 
    # that the agent will be able to pass through multiple times 

    def __init__(self)->None: 
        self.betweeness:float = 0.0  
        self.distance_centroid:float = 0.0 
        self.number_allowed_visits:int = 0

    def calculate_betweeness(self, G:nx.Graph, cluster_nodes:List[int], nodes_dict, cost_dist)->float:

        if len(cluster_nodes) < 25: 
            centrality = nx.closeness_centrality(G, distance='cost')
        else: 
            # TODO: Change this one to walk centrality for larger graphs. 
            centrality = nx.betweenness_centrality(G, weight='cost')
        
        if not nx.is_connected(G):
            print("Graph is not connected.")
            return
        
        self.betweeness = np.array([centrality[nodes_dict[n]] for n in cluster_nodes]).reshape(-1, 1)
        

    def get_node_with_min_total_distance(self, cluster_nodes, dist_matrix, nodes_dict):
        total_dists = {i: sum(dist_matrix[nodes_dict[i]][nodes_dict[j]] for j in cluster_nodes if i != j) for i in cluster_nodes}
        self.distance_centroid = np.array([total_dists[i] for i in cluster_nodes]).reshape(-1, 1)


    def normalize_values(self):
        scaler = MinMaxScaler() 
        self.betweeness = scaler.fit_transform(self.betweeness).flatten()
        self.distance_centroid = scaler.fit_transform(self.distance_centroid).flatten()
    

    def aggregate_scores(self, cluster_nodes): 
        composite_score = {cluster_nodes[i]: (self.betweeness[i] + (1 - self.distance_centroid[i])) / 2 for i in range(len(cluster_nodes))}
        return composite_score
    

    def allowed_visits(self, cluster_nodes, n_agents): 
        max_visits = n_agents 
        self.number_allowed_visits = {k: int(1 + self.betweeness[k] * (max_visits - 1)) for k in range(len(cluster_nodes))}
        

    def get_bridge_nodes(self, graph, cluster_nodes, cost_dist, nodes_dict, n_agents ): 
        cluster_nodes = list(cluster_nodes)
        self.calculate_betweeness(graph, cluster_nodes, nodes_dict,cost_dist)
        self.get_node_with_min_total_distance(cluster_nodes, cost_dist, nodes_dict)
        self.normalize_values() 
        self.allowed_visits(cluster_nodes, n_agents)
        composite_score = self.aggregate_scores(cluster_nodes)
        bridge_nodes = max(1, int(0.1 * len(composite_score)))

        return sorted(composite_score, key=composite_score.get, reverse=True)[:bridge_nodes]
    

