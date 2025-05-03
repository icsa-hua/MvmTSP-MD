import numpy as np 
import pandas as pd 
import networkx as nx 
from dummy_app.models.opa import opa_weights
from typing import Dict, List
from scipy.spatial.distance import pdist, squareform, euclidean 
from sklearn.preprocessing import robust_scale, MinMaxScaler


class TOPSISPriority:


    def __init__(self): 
        self.criteria = {} 
        self.criteria_names = ['Customers', 'Avg_distance', 'Density', 'Avg_Shortest_path_length', 'Avg_Clustering_coeff']
        self.topsis_scores = None 
        self.weights = None 


    def calculate_cluster_connectivity(self, cluster:pd.DataFrame)->Dict[str,float]: 

        coords = cluster[['X_coords','Y_coords']].values 
        dist_matrix = squareform(pdist(coords,metric='euclidean'))
        num_nodes = len(coords)

        # Average Distance between nodes 
        avg_distance = np.sum(dist_matrix) / (num_nodes * (num_nodes-1)) / 1000 

        G = nx.Graph() 
        for i in range(num_nodes): 
            for j in range(i + 1, num_nodes):  
                G.add_edge(i,j, weight=dist_matrix[i,j])

        density = nx.density(G) 

        # Average Shortest path 
        if nx.is_connected(G): 
            avg_shortest_path = nx.average_shortest_path_length(G, weight='weight')
        else: 
            avg_shortest_path = np.mean([
                nx.average_shortest_path_length(comp)
                for comp in (G.subgraph(c).copy() for c in nx.connected_components(G))
                if len(comp) > 1
            ])

        avg_clustering_coeff = nx.average_clustering(G, weight='weight')

        return {
            'Avg_distance': avg_distance,
            'Density': density,
            'Avg_Shortest_path_length': avg_shortest_path,
            'Avg_Clustering_coeff': avg_clustering_coeff
        }
    
    
    def gather_criteria(self, cluster:pd.DataFrame, cue_groups:Dict[int,List[object]]) -> Dict[str,float]: 

        # find the users inside the areas of the cluster. 
        # the areas outside the cluster are not considered. 
        num_customers = 0 
        for area_id in cluster['Area_id']:
            if area_id in cue_groups.keys(): 
               num_customers += len(cue_groups[area_id])

        conn_metrics = self.calculate_cluster_connectivity(cluster)
        return {
            'Customers': num_customers,
            **conn_metrics
        }
    

    def run_model(self, cluster_criteria:Dict[int,Dict], ranks:List[int])-> pd.DataFrame: 

        
        df = pd.DataFrame.from_dict(cluster_criteria, orient='index', columns=self.criteria_names)

        # normalize criteria with vector normalization 
        # norm_df = df / np.sqrt((df**2).sum())
        # if norm_df["Customers"].sum() == 0:
        #     norm_df["Customers"] = 0
        # else:
        #     norm_df["Customers"] = norm_df["Customers"] / norm_df["Customers"].sum()
        
        # NOTE: In this case it seems to be better to normalize using a MinMaxScaler 
        # to preserve scale and direction of the values. 

        scaler = MinMaxScaler() 
        norm_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index) 

        # Apply OPA for weight
        if ranks is None: 
            mean_val = df.mean().tolist() 
            ranks = np.argsort(-np.array(mean_val)) + 1 # Higher value = better 

        self.weights = opa_weights(ranks, self.criteria_names)
        weights_vector = pd.Series(self.weights) 

        # Weighted normalized devision matrix 
        weighted_df = norm_df * weights_vector 
        ideal = weighted_df.max() 
        nadir = weighted_df.min()

        dist_to_label = np.sqrt(((weighted_df - ideal) ** 2).sum(axis=1))
        dist_to_nadir = np.sqrt(((weighted_df - nadir) ** 2).sum(axis=1))

        scores = dist_to_nadir / (dist_to_label + dist_to_nadir)

        df['TOPSIS'] = scores 
        df['Rank'] = scores.rank(ascending=False)

        self.topsis_scores = scores 
        
        return df 








