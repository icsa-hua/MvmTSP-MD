import numpy as np 
import pandas as pd 
import pdb
from typing import Union, Any, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from dummy_app.designs.voronoi_map import Map
from scipy.spatial import Voronoi, voronoi_plot_2d 
from shapely.geometry import Point
from geopy.distance import distance
from geopy import Point as GeoPoint
from pyproj import Transformer

""" 
This function generates uniform random values between MIN and MAX for each element 
in `SAMPLES.shape`. It is used to initialize the starting position and directions of nodes. 
"""
U = lambda MIN, MAX, SAMPLES: np.random.rand(*SAMPLES.shape) * (MAX-MIN) + MIN 
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)

class GroundUser: 

    def __init__(self, x, y, mean_velocity=1.0)->None:
        self.x:Union[int,float] = x 
        self.y:Union[int,float] = y 
        self.velocity:float = mean_velocity 
        self.theta:float = 0.0 
        self.angle_mean:float = 0.0
        self.current_area:int = 0 
        self.distance_metric:str = 'euclidean'  # Default distance metric
    
    def move(self, map_obj:Map)->Point: 

        self.x += self.velocity * np.cos(self.theta)
        self.y += self.velocity * np.sin(self.theta)    

        # reflect boundaries 
        if self.x < map_obj.MIN_X or self.x > map_obj.MAX_X:
            self.x = np.clip(self.x, map_obj.MIN_X, map_obj.MAX_X)
            self.theta = np.pi - self.theta
            self.angle_mean = np.pi - self.angle_mean 

        if self.y < map_obj.MIN_Y or self.y > map_obj.MAX_Y:
            self.y = np.clip(self.y, map_obj.MIN_Y, map_obj.MAX_Y)
            self.theta = -self.theta
            self.angle_mean = -self.angle_mean

        # Update region 
        point = Point(self.x, self.y) 
        return point 
    

class GroundUserGroup: 

    def __init__(self, mobility_env, map_obj:Map, alpha:float, mean_velocity:float=1.0, sigma:float=0.5)->None:
        self.mobility_env = mobility_env
        self.map_obj = map_obj
        self.alpha = alpha
        self.mean_velocity = mean_velocity
        self.sigma = sigma

        self.alpha2 = 1.0 - self.alpha
        self.alpha3 = np.sqrt(1.0 - self.alpha * self.alpha) * self.sigma

        self.group = {}
        self.fig = self.mobility_env.fig 
        self.ax = self.mobility_env.ax
        self.scatter = None 
        self.process = self.mobility_env.env.process(self.simulate())
        self.distance_metric = 'euclidean'


    def load_users_from_csv(self, data_path:Union[str,Path], customers_path:Union[Path,str])->None: 
        df = pd.read_csv(data_path)
        customers = pd.read_csv(customers_path)

        df = df.set_index(customers['Customers ids'])
        df.columns = [i for i in range(1, len(df.columns)+1)]

        self.group = {area_id : [] for area_id in df.index}
        theta = U(0, 2*np.pi, df)
        theta = [theta[i][0] for i in range(len(theta))]
    
        for idx in df.index: 
            counter = 0 
            for x_val, y_val in zip(df.loc[idx, 1:10], df.loc[idx,11:20]): 
                user = GroundUser(x_val, y_val, self.mean_velocity)
                user.theta = theta[counter]
                user.angle_mean = user.theta

                user.current_area = idx 
                self.group[idx].append(user)
                counter += 1  


    def get_generated_users(self, user_points)->None:
        self.distance_metric = 'geodesic'

        df = pd.DataFrame(user_points)
        df = df.T

        self.group = {area_id : [] for area_id in user_points.keys()}
        theta = U(0, 2*np.pi, df)

        for area_id, points in user_points.items():
            for i, (x_val, y_val) in enumerate(points):
                user = GroundUser(x_val, y_val, self.mean_velocity)
                user.theta = theta[area_id][i]
                user.angle_mean = user.theta
                user.distance_metric = self.distance_metric
                user.current_area = area_id
                self.group[area_id].append(user)


    def get_coords(self)->np.ndarray: 

        if self.distance_metric == 'geodesic':
            # Convert to GeoPoint for geodesic distance calculations
            x = [float(user.x) for area in self.group.values() for user in area]
            y = [float(user.y) for area in self.group.values() for user in area]

            return np.column_stack((x, y))
        
        x = [user.x for area in self.group.values() for user in area]
        y = [user.y for area in self.group.values() for user in area]
        return np.column_stack((x,y))
    

    def plot_users(self, vor_map:Voronoi)->Tuple: 

        voronoi_plot_2d(
            vor_map, 
            ax=self.ax, 
            show_vertices=False, 
            line_colors='black',
            line_width=3, 
            line_alpha=0.6
        )
        self.scatter = self.ax.scatter(
            [],[],
            c='red',
            s=50,
            label='Ground Users',
            edgecolors='black'
        )

        self.scatter.set_offsets(self.get_coords())
        return self.fig, self.ax

    
    def plot_generated_users(self, map_obj:Any, regions, centroids, user_points)->Tuple: 

        if hasattr(map_obj, 'plot_map'):
            map_obj.plot_map(ax=self.ax,regions=regions, centroids=centroids, user_points=user_points)

            self.scatter = self.ax.scatter(
                [],[],
                c='red',
                s=50,
                label='Ground Users',
                edgecolors='black'
            )

            self.scatter.set_offsets(self.get_coords())
            return self.fig, self.ax

        return None, None


    def simulate(self): 

        regions = self.map_obj.clip_voronoi_to_box() 
        while True: 
            for area_users in self.group.values() : 
                for user in area_users:
                    point = user.move(self.map_obj)
                    new_region = self.map_obj.get_point_region(point, regions)
                    if new_region is not None: 
                        user.current_area = new_region  
                    
                    # Update motion 
                    user.velocity = (self.alpha * user.velocity +
                                     self.alpha2 * self.mean_velocity +
                                     self.alpha3 * np.random.normal(0.0))
                    
                    user.theta = (self.alpha * user.theta +
                                  self.alpha2 * user.angle_mean +
                                  self.alpha3 * np.random.normal(0.0))

            self.update_plot() 
            plt.draw()  
            yield self.mobility_env.env.timeout(1)


    def update_plot(self)->None:
        if self.scatter: 
            self.scatter.set_offsets(self.get_coords())
            

    # def run(self, duration:int=3600)->None: 
    #     self.env.run(until=duration)
