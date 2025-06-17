from dummy_app.tools.logger import logger
from dummy_app.tools.common import get_session_duration
from dummy_app.designs.agents import TSPAgents

import simpy
import numpy as np
import pandas as pd 
# import pyvista as pv
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Any, List, Dict, Optional



class EnvSim:

    def __init__(self, trials)->None: 

        self.env = simpy.Environment()
        self.ready_event = self.env.event()
        self.TimeWindow = list(range(0,trials,1)) # discrete time window for simulation 
        self.timestep:int = self.TimeWindow[0] 
        self.session_duration = 0 
        
        self.optimization_guard_flag:bool = False
        self.processes_initialized:bool = False 
        
        self.fig, self.ax = self.__init_plot()    
        
        self.agent_group:Optional[TSPAgents] = TSPAgents(self.env,{}, empty=True)
        
        #--- PYVISTA Setup ---
        # self.plotter = pv.Plotter(window_size=[1200,800])
        # self.user_actors = {} 
        # self.agent_actors = {} 
        # self.path_actors = {} 

        logger.debug("Initialized EnvSim and main figure...")
    
    
    # def setup_scene(self, regions,  centroids, cues): 
    #     # Initialize the scene. NOTE: call this once. 

    #     # 0. Add the lines of the voronoi regions
    #     # for poly in regions:
    #     #     x, y = poly.exterior.xy
    #     #     z = np.zeros_like(x)
    #     #     self.plotter.add_mesh(pv.Line([x, y, z], [x, y, z]), color='grey', line_width=5)

    #     # 1. Add static map elements
    #     for i, (lon, lat) in enumerate(centroids):
    #         self.plotter.add_mesh(pv.Sphere(radius=1, center=(lon, lat, 0)), color='black')
    #         self.plotter.add_text(f'N{i}', position=(lon,lat,0+2), font_size=10, color='black')

    #     # 2. Add initial actors for dynamic objects 
    #     for user_group in cues.group: 
    #         for user in cues.group[user_group]:
    #             actor = self.plotter.add_mesh(pv.Sphere(radius=0.5, center=(user.x, user.y, user.z)), color='green')
    #             self.user_actors[user.user_id] = actor

    #     # Agents (larger red pyramids) 
    #     if self.agent_group is not None: 
    #         for agent in self.agent_group.agents: 
    #             actor = self.plotter.add_mesh(pv.Cone(center=(agent.z, agent.y, agent.z), direction=(0,0,1), radius=2, height=4), color='red')
    #             self.agent_actors[agent.agent_id] = actor 

    #             # Create actor for path line 
    #             path_actor = self.plotter.add_mesh(pv.Line([agent.x, agent.y, agent.z], [agent.x, agent.y, agent.z]), color='blue', line_width=5)
    #             self.path_actors[agent.agent_id] = path_actor 

    #     self.plotter.add_title("3D SAR Simulation") 

    
    # def update_frame(self, cues): 
    #     # advance the simulation by one step and update visualization 

    #     #1. Advance simpy simulation 
    #     self.env.step() 

    #     #2. Update user actors 
    #     for user_group in cues.group: 
    #         for user in cues.group[user_group]: 
    #             actor = self.user_actors[user.user_id]
    #             actor.position = (user.x, user.y, user.z) # This is how we move the users. 

    #     # 3. Update agent actors and their paths
    #     if self.agent_group is not None: 
    #         for agent in self.agent_group.agents: 
    #             actor = self.agent_actors[agent.agent_id]
    #             actor.position = (agent.x, agent.y, agent.z) # How we move the users. 

    #             # Update path actor
    #             agent.history.append(actor.position)
    #             path_points = agent.history 
    #             if len(path_points) > 1: 
    #                 # Update the points of the line mesh 
    #                 self.path_actors[agent.agent_id].points = path_points 



    # def run_simulation(self, trials, constructor, cues, regions, centroids, output_filename="pyvista.mp4"):
    #     """The main application loop."""

    #     if not self.processes_initialized: 
    #         self.env.process(self.user_movement_process(cues,trials))  # SimPy coroutine, scheduled once
    #         self.env.process(self.agent_movement_process(trials))  # SimPy coroutine, scheduled once
    #         self.processes_initialized = True


    #     self.setup_scene(
    #         regions=regions,
    #         centroids=centroids,
    #         cues=cues
    #     )     

    #     logger.info('Starting Visualization...')
    #     self.plotter.show(interactive_update=True, auto_close=False) 

    #     # Open a movie file 
    #     self.plotter.open_movie(output_filename)

    #     # The main hearbeat loop 
    #     for i in range(trials): 
    #         if (
    #             constructor.Time == self.timestep
    #             and not self.optimization_guard_flag
    #         ):           
    #             self.optimization_guard_flag = True
    #             self.ready_event = self.env.event()
    #             logger.info(f"Re-optimizing at simulator time {self.env.now}")

    #         self.update_frame(cues) 
    #         self.plotter.write_frame() 

    #     logger.info("Simulation finished. Closing visualization...")
    #     self.plotter.close()




    def optimization_process(self, constructor:Any, distance_matrix:np.ndarray, data:pd.DataFrame, cue_groups:Dict[int,List[Any]], altitude:int):
        logger.debug("Running optimization process...")
        
        detailed_log = constructor.run_model(distance_matrix, data, cue_groups)
        constructor.gather_results() 

        self.agent_group = TSPAgents(self, detailed_log, altitude=altitude)

        self.session_duration += get_session_duration(detailed_log)
        constructor.Time += self.session_duration 
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

        # self.env.process(self.user_movement_process(cues, trials))
        if (
            constructor.Time == self.timestep
            and not self.optimization_guard_flag
        ):           
            self.optimization_guard_flag = True
            self.ready_event = self.env.event()
            logger.info(f"Re-optimizing at simulator time {self.env.now}")

            # --- SOLUTION FOR PLOTTING ---
            # 1. Clear the entire axes object. This removes all lines, scatters, legends, etc.
            # self.ax.cla()

            if self.ax.legend_:
                self.ax.legend_.remove() # Remove old legend if needed
            
            # if hasattr(self,"agent_group") and self.agent_group is not None: 
            #     for line in self.agent_group.path_lines:
            #         line.remove()

            # 2. Re-initialize the plot's appearance
            # self.init_animation(map_obj=map_obj, regions=regions, centroids=centroids, user_points=user_points, altitude=altitude)
            # You would also redraw your static map elements (voronoi cells, etc.) here
            

            # 3. Now run the optimization, which will create a new TSPAgents object.
            # This new object will draw fresh paths and a clean legend on the newly cleared axes.
            self.optimization_process(constructor, distance_matrix, data, cues.group, altitude=altitude)

        self.ax.legend()
        self.env.step()
        # if self.agent_group:
        #     agent_coords= self.agent_group.get_coords()
        #     self.agent_group.scatter.set_offsets(agent_coords)
        #     self.agent_group.scatter.set_color(self.agent_group.agent_colors)
            
        #     for i,agent in enumerate(self.agent_group.agents):
        #         # Get the agent's travel history up to the current point
        #         history = agent.history
        #         if len(history) > 1:
        #             # Unzip the list of tuples into separate lists for x, y, z
        #             x_hist, y_hist,t  = zip(*history)
                    
        #             # Get the correct line artist for this agent
        #             line = self.agent_group.path_lines[i]
                    
        #             # --- THIS IS THE FIX ---
        #             # Use the correct 3D method to update the line data
        #             line.set_data(x_hist, y_hist)
        # cues.simulate()

        # if self.agent_group: 
        #     self.agent_group.simulate() 

        # user_coords_3d = cues.get_coords() 
        # self.user_scatter._offsets3d = user_coords_3d

        # # Update agent positions and path lines
        # if self.agent_group:
        #     agent_coords_3d = self.agent_group.get_coords()
        #     self.agent_scatter._offsets3d = agent_coords_3d
            
            # for agent in self.agent_group.agents:
            #     path_history_3d = agent.get_path_history_3d() # returns (xs, ys, zs)
            #     line = self.agent_path_lines[agent.agent_id]
            #     line.set_data_3d(path_history_3d) # Update the line with the full history
        

        return cues, self.agent_group
    
    def __init_plot(self) -> tuple[Figure, Axes]:
        fig, ax = plt.subplots(figsize=(8,8))
        # Set 3D background color
        ax.set_facecolor('grey')  # plot area
        fig.patch.set_facecolor('grey')  
        ax.set_title("2D S&R Simulation")
        ax.grid(True)
        # plt.close()
        return fig, ax
    

    def init_animation(self, map_obj, regions, centroids, cues, altitude): 
        """
        Initializes the 3D plot for animation. Called only once 
        """

        self.fig = plt.figure(figsize=(12,10)) 
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title("UAV and Ground User Simulation") 
        
        # 1. Plot all STATIC background elements 

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