import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import colorsys 

from typing import Any 
from matplotlib.colors import to_rgb, to_hex


def clamp_rgb(rgb):
    """Ensure RGB values are within [0, 1] range."""
    return tuple(min(1.0, max(0.0, c)) for c in rgb)

def generate_agent_colormap(n_agents: int):
    cmap = plt.get_cmap("tab10")  # Up to 20 visually distinct colors
    agent_colors = []
    path_colors = []

    for i in range(n_agents):
        base_rgb = cmap(i % cmap.N)[:3]  # Get RGB tuple
        base_rgb = clamp_rgb(base_rgb)
        agent_colors.append(base_rgb)

        # Lighten color by increasing brightness in HSV space
        h, l, s = colorsys.rgb_to_hls(*base_rgb)
        lighter_rgb = colorsys.hls_to_rgb(h, min(1, l + 0.3), s)
        lighter_rgb = clamp_rgb(lighter_rgb)
        path_colors.append((*lighter_rgb, 0.9))  # Transparent version

    return agent_colors, path_colors



class TSPAgent: 

    def __init__(self, agent_id, path, altitude):
        
        self.agent_id = agent_id 
        # Ensure each element in path is a tuple/list with at least 3 elements
        self.plan = sorted(
            [tuple(item) for item in path if hasattr(item, '__getitem__') and len(item) >= 3],
            key=lambda x: x[2]
        )  # list of (x, y, t)
        self.index = 0 
        self.x = self.plan[0][0] if self.plan is not None else 0 
        self.y = self.plan[0][1] if self.plan is not None else 0
        self.z = altitude
        self.history = [] 


    def update_position(self, current_sim_time):
        while self.index < len(self.plan) and self.plan[self.index][2] <= current_sim_time:
            self.x, self.y, _ = self.plan[self.index]
            # print(f"Agent _ id {self.agent_id}({self.x},{ self.y})")
            self.index += 1

    def append_history(self, current_sim_time):
        self.history.append((self.x, self.y, current_sim_time)) 


class TSPAgents: 

    def __init__(self, mobility_env:Any, agent_paths:dict, empty:bool=False, altitude:int=1250):
        
        if empty: 
            self.mobility_env = mobility_env
            self.ax = None 
            self.agents = []
            self.path_lines = [] 
            self.agent_colors = [] 
            self.path_colors = []
            self.scatter:Any = None 
            self.altitude = altitude


        else: 
            self.mobility_env = mobility_env
            self.ax = self.mobility_env.ax
            agent_paths = agent_paths or {}
            self.altitude = altitude
            self.agents = [TSPAgent(id, path, self.altitude) for (id), path in agent_paths.items()]
            
            self.path_lines = []  # To store line objects for each agent
            n_agents = len(self.agents)
            self.agent_colors, self.path_colors = generate_agent_colormap(n_agents)
           
            for i, agent in enumerate(self.agents):
                # Initially empty line plot for each agent
                line, = self.ax.plot([], [], color=self.path_colors[i],linestyle='--', linewidth=2.5 , zorder=5)
                self.path_lines.append(line)

            self.scatter = self.ax.scatter([], [], s=200,  marker='^', label='Agents',  edgecolors='black', alpha=1 )
            
            for i, agent in enumerate(self.agents):
                self.ax.plot([], [], color=self.agent_colors[i], label=f'Agent {agent.agent_id}')
            # self.ax.legend()

            # This is necessary for the visualization of the agents. Otherwise nothing shows on the same plot 
            self.process = self.mobility_env.env.process(self.simulate())


    def get_coords(self):
        x = [agent.x for agent in self.agents]
        y = [agent.y for agent in self.agents]
        return np.column_stack((x,y))

        # z = [self.altitude] * len(x)

        # return np.column_stack((x,y,z))


    def update_plot(self):
        if self.scatter: 
           self.scatter.set_offsets(self.get_coords())
           self.scatter.set_color(self.agent_colors)
         
        # coords = self.get_coords()
        # xs, ys, zs = coords[:,0], coords[:,1], coords[:,2]
        # if self.scatter: 
        #     self.scatter._offsets3d = (xs, ys, zs)
        #     self.scatter.set_color(self.agent_colors)
         
         

    def simulate(self):

        # This process now just triggers updates. The agent itself knows what to do.
        while True:
            timestep = self.mobility_env.env.now 
            for i, agent in enumerate(self.agents):
                agent.update_position(timestep)
                agent.append_history(timestep)
                path = agent.plan  # Assume path is a list of (x, y) coordinates
                if path:
                    history = agent.history
                    x_hist, y_hist,t  = zip(*history)
                    self.path_lines[i].set_data(x_hist, y_hist)

            self.update_plot()
            plt.draw()
            if hasattr(self.mobility_env, "timestep"):
                self.mobility_env.timestep += 1 
            yield self.mobility_env.env.timeout(1)