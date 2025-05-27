import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from typing import Union, Dict 
from pathlib import Path 
from scipy.spatial import Voronoi, voronoi_plot_2d 
from shapely.geometry import Polygon, box 
from geopy.distance import geodesic
from shapely.ops import triangulate
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix 
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform 
from geopy.distance import distance
from geopy import Point 




class Map: 

    def __init__(self, data_path:Union[Path, str], incremental:int)->None: 

        try: 
            self.points = pd.read_csv(data_path)[['X_coords','Y_coords']]
        except IndexError: 
            raise ValueError('Invalid data path or invalid columns names [X_coord, Y_coord]')
        
        self.incremental = incremental 
        self.vor_map = None 

        self.MAX_X = self.points['X_coords'].max() + 250
        self.MAX_Y = self.points['Y_coords'].max() + 250
        self.MIN_X = self.points['X_coords'].min() - 250
        self.MIN_Y = self.points['Y_coords'].min() - 250


    def voronoi_tessellation(self)->None: 
        self.vor_map = Voronoi(self.points)

    
    def plot_voronoi(self)->None: 
        voronoi_plot_2d(self.vor_map) 
        plt.show() 

    def extract_regions(self)->Dict[int,Polygon]: 
        regions = {} 

        if self.vor_map is None:
            raise ValueError("Voronoi map is not initialized. Call 'voronoi_tessellation()' first.")
        
        for i, region_idx in enumerate(self.vor_map.point_region): 
            vertices = self.vor_map.regions[region_idx]
            if -1 not in vertices and vertices: 
                polygon = Polygon(self.vor_map.vertices[v] for v in vertices) 
                regions[i] = polygon 
        
        return regions 
    

    def clip_voronoi_to_box(self)->Dict[int, Polygon]: 
        bounding_box = box(
            self.points['X_coords'].min() - 100, self.points['Y_coords'].min() - 100,
            self.points['X_coords'].max() + 100, self.points['Y_coords'].max() + 100
        )
        if self.vor_map is None:
            raise ValueError("Voronoi map is not initialized. Call 'voronoi_tessellation()' first.")
        
        regions = {} 
        for i, region in enumerate(self.vor_map.regions): 
            if -1 not in region and region: 
                polygon = Polygon(self.vor_map.vertices[v] for v in region)
                clipped = polygon.intersection(bounding_box)
                if not clipped.is_empty:
                    regions[i] = clipped
        
        return regions 
    

    def get_point_region(self, point, regions)->Union[int, None]: 
        for region_id, polygon in regions.items(): 
            if polygon.contains(point): 
                return region_id
            

        
class MapGenerator(Map): 

    def __init__(self, num_areas:int=30, users_per_area:int=5, low_lat:float=52.5, high_lat:float=52.6, low_long:float=13.5, high_long:float=13.6, seed:int=0): 

        self.num_areas = num_areas 
        self.low_lat = low_lat
        self.high_lat = high_lat
        self.low_long = low_long
        self.high_long = high_long
        self.seed = seed 
        self.vor_map = None
        self.users_per_area = users_per_area


    def generate_points(self): 
        np.random.seed(self.seed) 
        latitudes = np.random.uniform(self.low_lat, self.high_lat, self.num_areas)
        longitudes = np.random.uniform(self.low_long, self.high_long, self.num_areas)
        self.points = np.column_stack((longitudes, latitudes))
        self.vor_map = Voronoi(self.points) 
        self.points = pd.DataFrame(self.points, columns=['X_coords', 'Y_coords'])
        self.set_boundaries()

    
    def generate_bb(self): 
        low_lat = self.low_lat-0.05
        low_long = self.low_long - 0.05 
        high_lat = self.high_lat+0.05 
        high_long = self.high_long+0.05 
        return Polygon([
            (low_long, low_lat), 
            (high_long, low_lat), 
            (high_long, high_lat), 
            (low_long, high_lat)
        ])
    

    def voronoi_polygons(self): 

        if self.vor_map is None:
           raise ValueError("Voronoi map is not initialized. Call 'voronoi_tessellation()' first.")
        

        bbox = self.generate_bb() 
        regions = [] 
        centroids = [] 
        user_points = defaultdict(list)
        unified_id = 0 
        for region_idx in self.vor_map.point_region:
            region = self.vor_map.regions[region_idx]
            if not -1 in region and len(region) > 0:
                poly_points = [self.vor_map.vertices[i] for i in region]
                poly = Polygon(poly_points)
                poly = poly.intersection(bbox)
                if poly.is_empty or not poly.is_valid or poly.area == 0: continue

                regions.append(poly)
                centroids.append(poly.centroid.coords[0])
                user_points[unified_id]= self.generate_users(poly, user_points[unified_id]) 
                unified_id += 1 
        return regions, centroids, user_points


    def calculate_centroid_distance(self, centroids): 

        dist_matrix = squareform(pdist(centroids, lambda u, v: geodesic(u, v).km))
        return dist_matrix


    def generate_users(self, poly, user_points):

        triangles = triangulate(poly) 
        areas = np.array([tri.area for tri in triangles])
        probs = areas / areas.sum()

        for _ in range(self.users_per_area):
            tri_index = np.random.choice(range(len(triangles)), p=probs)
            tri = triangles[tri_index]
            a, b, c = tri.exterior.coords[:3]
            u, v = np.random.rand(2)
            if u + v > 1:
                u, v = 1 - u, 1 - v
            px = a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0])
            py = a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1])
            user_points.append((px, py))

        return user_points
        
        
    def plot_map(self, ax, regions, centroids, user_points): 
        # Plot results
        # fig, ax = plt.subplots(figsize=(10, 10))

        if self.vor_map is None:
           raise ValueError("Voronoi map is not initialized. Call 'voronoi_tessellation()' first.")
        

        for poly in regions:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, alpha=0.3, edgecolor='black')
        x = [point[0] for point in centroids]
        y = [point[1] for point in centroids]
        labels = [f"A{i}" for i,_ in enumerate(centroids) ]
        ax.scatter(*zip(*centroids), c='red', marker='x', label='Centroids')
        ax.scatter(*zip(*self.vor_map.points), c='blue', marker='o', label='Original Nodes')
        ax.scatter(*zip(*user_points), c='green', s=10, label='Users')
        for i in range(len(centroids)): 
            ax.text(x[i],y[i], labels[i], c='black')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Voronoi Diagram with Triangle-Based User Sampling")
        ax.legend()
        plt.grid(True)
        plt.show()


    def plot_map_3D(self, regions, centroids, user_points):

        # Assign synthetic altitudes to centroids (e.g., 100m ± 20m)
        altitudes = np.random.uniform(80, 120, len(centroids))

        # 3D Plotting
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot Voronoi region edges in 3D at z=0
        for poly in regions:
            x, y = poly.exterior.xy
            z = np.zeros_like(x)
            ax.plot(x, y, z, color='gray', alpha=0.5)

        # Plot centroids with altitude
        for i, (lon, lat) in enumerate(centroids):
            ax.scatter(lon, lat, altitudes[i], c='red', marker='x', s=50)
            ax.text(lon, lat, altitudes[i] + 5, f'A{i}', color='black')

        # Plot user points at ground level (z=0)
        for (x, y) in user_points:
            ax.scatter(x, y, 0, c='green', s=10)

        # Plot drone paths between centroids
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                x_vals = [centroids[i][0], centroids[j][0]]
                y_vals = [centroids[i][1], centroids[j][1]]
                z_vals = [altitudes[i], altitudes[j]]
                ax.plot(x_vals, y_vals, z_vals, 'blue', alpha=0.3)

        # Axis labels
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Altitude (m)')
        ax.set_title('3D Voronoi Areas with Drone Paths and Users')
        plt.tight_layout()
        plt.show()


    def get_central_depots(self, sites, number_of_areas=2): 
        sites = np.array(sites)
        n = len(sites)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = geodesic(sites[i], sites[j]).km
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist  # symmetry
        
        avg_distances = np.mean(dist_matrix, axis=1)
        central_indices = np.argsort(avg_distances)[:number_of_areas]
        return central_indices


    def set_boundaries(self, buffer_m:int=200): 

        lat_min = self.points['Y_coords'].min()
        lat_max = self.points['Y_coords'].max()
        lon_min = self.points['X_coords'].min()
        lon_max = self.points['X_coords'].max()

        # Create bounding points
        south = distance(meters=buffer_m).destination(Point(lat_min, (lon_min + lon_max) / 2), bearing=180)
        north = distance(meters=buffer_m).destination(Point(lat_max, (lon_min + lon_max) / 2), bearing=0)
        west  = distance(meters=buffer_m).destination(Point((lat_min + lat_max) / 2, lon_min), bearing=270)
        east  = distance(meters=buffer_m).destination(Point((lat_min + lat_max) / 2, lon_max), bearing=90)

        self.MAX_X = east.longitude
        self.MAX_Y = north.latitude
        self.MIN_X = west.longitude
        self.MIN_Y = south.latitude
       
       

