import numpy as np
import matplotlib.pyplot as plt


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

    def __init__(self, mobility_env, agent_paths):
        self.mobility_env = mobility_env
        self.ax = self.mobility_env.ax
        agent_paths = agent_paths or {}

        
        self.agents = [TSPAgent(id, path) for (id), path in agent_paths.items()]
        self.scatter = self.ax.scatter([], [], c='blue', s=100, label='Agents', edgecolors='black')
        
        # This is necessary for the visualization of the agents. Otherwise nothing shows on the same plot 
        self.process = self.mobility_env.env.process(self.simulate())
        import pdb; pdb.set_trace()


    def get_coords(self):
        x = [agent.x for agent in self.agents]
        y = [agent.y for agent in self.agents]
        return np.column_stack((x,y))


    def update_plot(self):

        if self.scatter: 
            self.scatter.set_offsets(self.get_coords())
         

    def simulate(self):
        while True:
            timestep = self.mobility_env.env.now
            for agent in self.agents:
                agent.update_position(timestep)
            self.update_plot()
            plt.draw()
            yield self.mobility_env.env.timeout(1)
