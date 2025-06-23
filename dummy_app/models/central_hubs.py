import networkx as nx
import numpy as np 
import pandas as pd 

from typing import Dict, List, Any
from sklearn.preprocessing import MinMaxScaler


class CentralHub: 

    """
        This class finds the central node for a cluster to act as a bridge or hub, 
        that the agent will be able to pass through multiple times. The node is determined 
        through the betweeness centrality and minimum distance to the centroid.
    """ 
    # 

    def __init__(self)->None: 
        self.betweeness:np.ndarray = np.array([])  
        self.distance_centroid:np.ndarray = np.array([])   
        self.number_allowed_visits:Dict[int,int] = {}


    def calculate_betweeness(self, G:nx.DiGraph, cluster_nodes:List[int], nodes_dict, cost_dist, depot_id)->None:
        if not nx.is_strongly_connected(G):
            print("Directed graph is not strongly connected.")
            return

        central_candidates = [n for n in G.nodes if n != depot_id]
        reverse_dict = {v:k for k,v in nodes_dict.items()}
        if len(cluster_nodes) < 25: 
            centrality = nx.closeness_centrality(G, distance='cost')
            centrality = {k: v for k, v in centrality.items() if k in central_candidates}

        else: 
            # TODO: Change this one to walk centrality for larger graphs. 
            centrality = nx.betweenness_centrality(G, weight='cost')

        self.betweeness = np.array([centrality[nodes_dict[n]] for n in cluster_nodes if n != reverse_dict[depot_id]]).reshape(-1, 1)
        

    def get_node_with_min_total_distance(self, cluster_nodes, dist_matrix, nodes_dict, depot_id):
        reverse_dict = {v:k for k,v in nodes_dict.items()}

        total_dists = {i: sum(dist_matrix[nodes_dict[i]][nodes_dict[j]-1] for j in cluster_nodes if i != j) for i in cluster_nodes if i != reverse_dict[depot_id]}
        self.distance_centroid = np.array([total_dists[i] for i in cluster_nodes if i != reverse_dict[depot_id]]).reshape(-1, 1)


    def normalize_values(self):
        scaler = MinMaxScaler(feature_range=(0,1)) 
        self.betweeness = scaler.fit_transform(self.betweeness).flatten()
        self.distance_centroid = scaler.fit_transform(self.distance_centroid).flatten()
    

    def aggregate_scores(self, cluster_nodes,alpha=0.5): 
        composite_score = {
            cluster_nodes[i]: alpha * self.betweeness[i] + (1 - alpha) * (1 - self.distance_centroid[i])
            for i in range(len(cluster_nodes) - 1)
        }        
        return composite_score
    

    def allowed_visits(self, cluster_nodes, n_agents, bridge_nodes, composite_score): 
        max_visits = n_agents 

        self.number_allowed_visits = {
            k: int(1 + composite_score[k] * (max_visits - 1))
            for k in range(len(cluster_nodes) - 1)
        }

        composite_dict = {
            'composite_score': composite_score.values() 
        }
        composite_df = pd.DataFrame.from_dict(composite_dict)
        bridge_nodes = int(composite_df.idxmax().iloc[0])

        # Ensure no node has zero allowed visits
        for k in self.number_allowed_visits:
            if self.number_allowed_visits[k] == 0:
                self.number_allowed_visits[k] = 1

        # Ensure at least one node gets more than 1 visit
        if all(v == 1 for v in self.number_allowed_visits.values()):
            self.number_allowed_visits[bridge_nodes] = n_agents + 1 

            
    def get_bridge_nodes(self, graph:nx.DiGraph, cluster_nodes:List[int], cost_dist:Any, nodes_dict:Dict[int,int], n_agents:int )-> List[int]: 
        depot_id = nodes_dict[len(nodes_dict)-1]
    
        self.calculate_betweeness(graph, cluster_nodes, nodes_dict,cost_dist,depot_id=depot_id)
        self.get_node_with_min_total_distance(cluster_nodes, cost_dist, nodes_dict,depot_id=depot_id)
        
        self.normalize_values() 
        
        composite_score = self.aggregate_scores(cluster_nodes)
        
        bridge_nodes = max(1, int(0.1 * len(composite_score))) # consider only the top 10% of nodes (by composite score) to act as bridge or central nodes.
        
        self.allowed_visits(cluster_nodes, n_agents, bridge_nodes, composite_score)

        return [node for node, _ in sorted(composite_score.items(), key=lambda item: item[1], reverse=True)[:bridge_nodes]]
    

