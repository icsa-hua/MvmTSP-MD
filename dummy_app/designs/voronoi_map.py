from dummy_app.tools.logger import logger 

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
from pyproj import Transformer 

# Functions to transform coordinates from EPSG to UTM 
transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
transformer_to_latlon = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)


class Map: 
    """
    This class is used to capture or provide the interface to generate the map based on real 
    geographical latitude and longitude. The data are transformed between coordinate 
    systems to accommodate for realistic distances and for visualization and user movement. 
    """

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
    """
    Map generator that takes the center of a location and creates the Voronoi 
    tessellation and generates the users based on Delaunay triangulation. 
    """

    def __init__(self, num_areas:int=30, users_per_area:int=5, lat:float=13.5, lon:float=33.3, seed:int=0, low:int=1, high:int=1): 
        
        logger.debug("Map Generator initialized...")
        # Transform lon and lat into UTM for better point management. 
        # print("----------------------------artemis-----------------------------lat/lon", lat, lon)
        self.lat, self.lon = transformer_to_utm.transform(lat,lon)
<<<<<<< HEAD
        logger.info(transformer_to_latlon.transform(self.lat,self.lon)) 
=======
>>>>>>> ff64c65 (Add files via upload)
        self.lat_wgs84 = lat 
        self.lon_wsg84 = lon 
        self.num_areas = num_areas 
        self.seed = seed 
        self.vor_map = None
        self.users_per_area = users_per_area
        self.low = low
        self.high = high
        
        
    
    def create_environment(self, show_map:bool=False, show_3d_map:bool=False): 
        regions, centroids, all_user_points = [],[],[]
        depots, user_points = [], defaultdict(list) 
        distance_matrix_wgs84 = np.ndarray((0,0))
        try: 
            # Generate points in both utm
            points_utm = self.generate_points()

            # Fixed plot window based on the UTM coords 
            self.set_boundaries(points_utm)

            # Voronoi Map returns applicable regions to generate user points 
            regions, centroids, user_points = self.voronoi_polygons()
            # print("-------voronoi_polygons-------------",centroids)

            # Transform centroids into WGS84 
            centroids_wgs84 = np.array([transformer_to_latlon.transform(lon,lat) for lon, lat in centroids])
            
            # calculate distances in both UTM and WGS84
            distance_matrix_utm = distance_matrix(centroids, centroids)
            distance_matrix_wgs84 = squareform(pdist(centroids_wgs84, lambda u, v: geodesic(u, v).km))
            
            # Get central areas based on WGS84 coordinates 
            depots = self.get_central_depots(sites=centroids_wgs84)
            all_user_points = [point for points in user_points.values() for point in points]
        
        except Exception as E: 
            logger.exception("Raised exception {E}.")

        transformer = Transformer.from_crs("epsg:32633", "epsg:4326", always_xy=True)
        # x0, y0 = transformer.transform(centroids[0][0], centroids[0][1])
        # print("lon/lat", x0, y0)
        return regions, centroids, user_points, depots, distance_matrix_wgs84, all_user_points
        

    def generate_points(self): 
        # 1. Δημιουργείς κατευθείαν το NumPy array
        np.random.seed(self.seed) 
        latitudes  = self.lat + np.random.uniform(-self.low, self.high, self.num_areas)
        longitudes = self.lon + np.random.uniform(-self.low, self.high, self.num_areas)        
        points = np.column_stack((latitudes, longitudes))

        # 2. Φτιάχνεις τη Voronoi πάνω στο καθαρό array
        self.vor_map = Voronoi(points) 

        # 3. Φτιάχνεις το DataFrame για να το έχεις διαθέσιμο μετά
        points_utm = pd.DataFrame(points, columns=['X_coords','Y_coords'])
        self.points = points_utm

        # print("-------------------------------", points_utm)
        return points_utm
    

    def generate_bb(self):
        # Χρησιμοποιούμε το DataFrame self.points που ήδη περιέχει όλα τα σημεία
        minx = self.points['X_coords'].min()
        maxx = self.points['X_coords'].max()
        miny = self.points['Y_coords'].min()
        maxy = self.points['Y_coords'].max()
        margin = max(maxx - minx, maxy - miny) * 0.1  # 10% έξτρα περιθώριο

        return box(
            minx - margin, 
            miny - margin, 
            maxx + margin, 
            maxy + margin
        )
    

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
            # print(f"Region {region_idx} → vertices indices: {region}")
            if not -1 in region and len(region) > 0:
                
                poly_points = [self.vor_map.vertices[i] for i in region]
                
                poly = Polygon(poly_points)
                poly = poly.intersection(bbox)
                if poly.is_empty or not poly.is_valid or poly.area == 0: continue

                regions.append(poly)
                # print("  → good")
                centroids.append(poly.centroid.coords[0]) # This is in UTM 
                user_points[unified_id]= self.generate_users(poly, user_points[unified_id]) 
                unified_id += 1 

        return regions, centroids, user_points


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
        
        
    def plot_map(self, ax, regions, centroids): 

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

        for i in range(len(centroids)): 
            ax.text(x[i],y[i], labels[i], c='black')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()
        

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
    
    
    def clip_voronoi_to_box(self)->Dict[int, Polygon]: 

        bounding_box = self.generate_bb()
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


    def set_boundaries(self, points_utm): 
        self.MAX_X = points_utm['X_coords'].max() 
        self.MAX_Y = points_utm['Y_coords'].max() 
        self.MIN_Y = points_utm['Y_coords'].min()
        self.MIN_X = points_utm['X_coords'].min()
        


       

