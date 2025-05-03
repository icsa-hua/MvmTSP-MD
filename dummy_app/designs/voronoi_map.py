import pandas as pd 
import matplotlib.pyplot as plt
from typing import Union, Dict 
from pathlib import Path 
from scipy.spatial import Voronoi, voronoi_plot_2d 
from shapely.geometry import Polygon, box 


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
            

        















