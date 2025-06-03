import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Any 


class TSPAgent: 

    def __init__(self, id, path):
        self.path = path  # list of (x, y, t)
        self.index = 0
        self.x = path[0][0]
        self.y = path[0][1]
        self.agent_id = id 


    def update_position(self, timestep):
        # print(f"Agent {self.agent_id} with path {self.path[self.index]} with index {self.index} at {timestep}")
        while self.index < len(self.path) and self.path[self.index][2] <= timestep:
            self.x, self.y, _ = self.path[self.index]
            # print(f"Agent _ id {self.agent_id}({self.x},{ self.y})")
            self.index += 1


class TSPAgents: 

    def __init__(self, mobility_env:Any, agent_paths:dict, empty:bool=False):
        
        if empty: 
            self.mobility_env = mobility_env
            self.ax = None 
            self.agents = []
            self.path_lines = [] 
            self.agent_colors = [] 
            self.scatter:Any = None 

        else: 
            self.mobility_env = mobility_env
            self.ax = self.mobility_env.ax
            agent_paths = agent_paths or {}
            
            self.agents = [TSPAgent(id, path) for (id), path in agent_paths.items()]
            
            self.path_lines = []  # To store line objects for each agent

            for agent in self.agents:
                # Initially empty line plot for each agent
                line, = self.ax.plot([], [], linestyle='--', linewidth=2.5 , zorder=5)
                self.path_lines.append(line)

            n_agents = len(self.agents)
            self.agent_colors = cm.get_cmap('tab10', n_agents)(range(n_agents))
            self.scatter = self.ax.scatter([], [], s=100, label='Agents', edgecolors='black')
            
            for i, agent in enumerate(self.agents):
                self.ax.plot([], [], color=self.agent_colors[i], label=f'Agent {agent.agent_id}')
            self.ax.legend()

            # This is necessary for the visualization of the agents. Otherwise nothing shows on the same plot 
            self.process = self.mobility_env.env.process(self.simulate())


    def get_coords(self):
        x = [agent.x for agent in self.agents]
        y = [agent.y for agent in self.agents]
        return np.column_stack((x,y))


    def update_plot(self):

        if self.scatter: 
            self.scatter.set_offsets(self.get_coords())
            self.scatter.set_color(self.agent_colors)
         

    def simulate(self):
        while True:
            timestep = self.mobility_env.timestep 
            for i, agent in enumerate(self.agents):
                agent.update_position(timestep)

                path = agent.path  # Assume path is a list of (x, y) coordinates
                if path:
                    x_vals, y_vals, _ = zip(*path)
                    self.path_lines[i].set_data(x_vals, y_vals)
            
            self.update_plot()
            plt.draw()
            if hasattr(self.mobility_env, "timestep"):
                self.mobility_env.timestep += 1 
            yield self.mobility_env.env.timeout(1)
