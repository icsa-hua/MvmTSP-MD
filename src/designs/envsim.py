from dummy_app.tools.logger import logger
from dummy_app.tools.common import get_session_duration, add_session_time
from dummy_app.designs.agents import TSPAgents

import simpy
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from collections import defaultdict
from typing import Any, List, Dict, Optional

# import pyvista as pv # This was initially tested for 3D animation 


class EnvSim:
    """
    This class is the core of the simulation that calls the optimization 
    every time it is needed (a.k.a. at the start and when the agents finish their
    mission. 
    """
    def __init__(self, trials, render: bool = True)->None: 

        self.env = simpy.Environment()
        self.ready_event = self.env.event()
        self.TimeWindow = list(range(0,trials,1)) # discrete time window for simulation 
        self.timestep:int = self.TimeWindow[0] 
        self.session_duration = 0 
        self.trials = trials
        self.render = render
        self.completed_sessions = 0
        self.combined_paths = defaultdict(list)
        
        self.optimization_guard_flag:bool = False
        self.processes_initialized:bool = False 
        
        self.fig, self.ax = self.__init_plot() if self.render else (None, None)
        
        self.agent_group:Optional[TSPAgents] = TSPAgents(self,{}, empty=True, render=self.render)
        
        #--- PYVISTA Setup --- self.plotter = pv.Plotter(window_size=[1200,800]) self.user_actors = {} self.agent_actors = {} self.path_actors = {} 

    
    def optimization_process(self, constructor:Any, distance_matrix:np.ndarray, data:pd.DataFrame, cue_groups:Dict[int,List[Any]], altitude:int):
        logger.debug("Running optimization process...")
        
        raw_detailed_log = constructor.run_model(distance_matrix, data, cue_groups)
        constructor.gather_results() 
        local_duration = get_session_duration(raw_detailed_log) if raw_detailed_log else 0
        detailed_log = add_session_time(raw_detailed_log, self.session_duration) if self.session_duration > 0 else raw_detailed_log
        self._append_combined_paths(detailed_log)
        self.agent_group = TSPAgents(self, detailed_log, altitude=altitude, render=self.render)
        self.session_duration += local_duration
        self.completed_sessions += 1

        constructor.Time = self.session_duration 
        self.ready_event.succeed()
        self.optimization_guard_flag = False 

        logger.debug(f"Session duration: {self.session_duration}")
        

    def user_movement_process(self, cues:Any, trials:int):
        yield self.ready_event 
        cues.simulate()  # no timeout here
        

    def agent_movement_process(self, trials:int): 
        yield self.ready_event
        self.agent_group.simulate()


    def simulations(
        self,
        frame,
        constructor:Any,
        cues:Any, 
        map_obj:Any,
        distance_matrix:np.ndarray,
        data:pd.DataFrame,
        regions:list, 
        centroids:list, 
        user_points:list,
        altitude:int,
        trials:int=500, 

    )->Any:
        
        if not self.processes_initialized: 
            self.env.process(self.user_movement_process(cues,trials))  # SimPy coroutine, scheduled once
            self.env.process(self.agent_movement_process(trials))  # SimPy coroutine, scheduled once
            self.processes_initialized = True

        if (
            constructor.Time == self.timestep
            and not self.optimization_guard_flag
        ):           
            self.optimization_guard_flag = True
            self.ready_event = self.env.event()
            logger.info(f"Re-optimizing at simulator time {self.env.now}")

            if self.ax.legend_:
                self.ax.legend_.remove() # Remove old legend if needed
            
            self.optimization_process(constructor, distance_matrix, data, cues.group, altitude=altitude)

        self.ax.legend()
        self.env.step()
        return cues, self.agent_group


    def run_headless(
        self,
        constructor: Any,
        cues: Any,
        distance_matrix: np.ndarray,
        data: pd.DataFrame,
        altitude: int,
        trials: int,
    ) -> Any:
        self.completed_sessions = 0
        self.session_duration = 0
        self.combined_paths = defaultdict(list)
        self.optimization_process(constructor, distance_matrix, data, cues.group, altitude=altitude)

        while self.completed_sessions < trials:
            if constructor.Time == self.timestep and not self.optimization_guard_flag:
                self.optimization_guard_flag = True
                self.ready_event = self.env.event()
                logger.info(f"Re-optimizing at simulator time {self.env.now}")
                self.optimization_process(constructor, distance_matrix, data, cues.group, altitude=altitude)
                continue

            try:
                self.env.step()
            except simpy.core.EmptySchedule:
                break

        return constructor


    def _append_combined_paths(self, shifted_paths: Dict[Any, List[Any]]) -> None:
        for agent_id, agent_path in shifted_paths.items():
            self.combined_paths[agent_id].extend(agent_path)
    

    def __init_plot(self) -> tuple[Figure, Axes]:
        fig, ax = plt.subplots(figsize=(8,8))
        # Set 3D background color
        ax.set_facecolor('white')  # plot area
        fig.patch.set_facecolor('grey')  
        ax.set_title("2D S&R Simulation")
        ax.grid(True)
        # plt.close()
        return fig, ax
    

    def init_animation(self, map_obj, regions, centroids, cues, altitude): 
        """
        Initializes the plot for animation.
        This function was tested in combination with the PYVISTA. 
        NOTE: It should not be considered for the plot nor the simulation 
        visualization. 
        """

        self.fig = plt.figure(figsize=(12,10)) 
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title("UAV and Ground User Simulation") 
        
        area_alt = np.zeros(len(centroids))
        for poly in regions:
            x, y = poly.exterior.xy
            z = np.zeros_like(x)
            self.ax.plot(x, y, z, color='gray', alpha=0.5)

        for i, (lon, lat) in enumerate(centroids):
            self.ax.scatter(lon, lat, area_alt[i], c='blue', marker='o', s=50)
            self.ax.text(lon, lat, area_alt[i] + 0.5, f'A{i}', color='black')
                
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_zlabel('Altitude')
        
        plt.tight_layout()

        cues.scatter = self.ax.scatter(
            [],[],[],
            c='red',
            s=50,
            label='Ground Users',
            edgecolors='black'
        )

        coords = cues.get_coords() 
        xs,ys,zs = zip(*coords) 
        cues.scatter._offsets3d = (xs, ys, zs)

        # For agent positions
        self.agent_scatter = self.ax.scatter(
            [], [], [],
            c='red',
            s=50,
            marker='^',
            depthshade=True, 
            label='Agents'
        )

        self.agent_path_lines = {}
        if self.agent_group:
            
            for i, agent in enumerate(self.agent_group.agents):
                # Create a line artist for each agent and store it
                line, = self.ax.plot([], [], [], linestyle='--', color=self.agent_group.agent_colors[i], label=f'Path Agent {agent.agent_id}')
                self.agent_path_lines[agent.agent_id] = line

        self.ax.legend()

        # FuncAnimation requires returning the artists that will be updated
        artists = [cues.scatter, self.agent_scatter] + list(self.agent_path_lines.values())
        return artists
