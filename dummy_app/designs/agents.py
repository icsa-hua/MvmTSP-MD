import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Any 


class TSPAgent: 

    def __init__(self, agent_id, path):
        
        self.agent_id = agent_id 
        # Ensure each element in path is a tuple/list with at least 3 elements
        self.plan = sorted(
            [tuple(item) for item in path if hasattr(item, '__getitem__') and len(item) >= 3],
            key=lambda x: x[2]
        )  # list of (x, y, t)
        self.index = 0 
        self.x = self.plan[0][0] if self.plan is not None else 0 
        self.y = self.plan[0][1] if self.plan is not None else 0
        

    def update_position(self, current_sim_time):
        if not self.plan:
            return


        start_event = None 
        end_event = None 

        for i in range(len(self.plan)-1): 
            if self.plan[i][2] <= current_sim_time < self.plan[i+1][2]: 
                start_event = self.plan[i] 
                end_event = self.plan[i+1]
                break 

        
        if start_event is None : 
            final_node = self.plan[-1]
            self.x, self.y = final_node[0], final_node[1] 
            return
       
        if end_event is None:
            self.x, self.y = self.plan[-1][0], self.plan[-1][1]
            return

        start_time = start_event[2] 
        end_time = end_event[2] 

        start_pos = (start_event[0], start_event[1])
        end_pos = (end_event[0], end_event[1])

        if start_event[0] == start_event[1] : 
            self.x, self.y = start_pos 
            return 
        
        travel_duration = end_time - start_time 
        if travel_duration <= 0: 
            self.x, self.y = end_pos 
            return 
        
        fraction = (current_sim_time - start_time) / travel_duration 

        self.x = start_pos[0] + fraction * (end_pos[0] - start_pos[0])
        self.y = start_pos[1] + fraction * (end_pos[1] - start_pos[1])
        












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
            self.agent_colors = cm.get_cmap('tab10', n_agents)(np.arange(n_agents))
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

        # This process now just triggers updates. The agent itself knows what to do.
        while self.mobility_env.env.now < self.mobility_env.session_duration[-1]:
            current_time = self.mobility_env.env.now
            for agent in self.agents:
                agent.update_position(current_time)
            
            # This part remains the same to update the plot
            self.update_plot()
            plt.draw()
            
            yield self.mobility_env.env.timeout(1) # Advance simulation by one step
            
