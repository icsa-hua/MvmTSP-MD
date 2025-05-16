import simpy
import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
from typing import Any, List, Dict
from matplotlib.animation import FuncAnimation
from dummy_app.tools.common import get_session_duration
from dummy_app.designs.agents import TSPAgents

class EnvSim:

    def __init__(self)->None: 
        self.env = simpy.Environment()
        self.ready_event = self.env.event()
        self.session_duration = 0 
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title("Shared Plot")
        self.ax.grid(True)        
        self.agent_group = TSPAgents(self, {})
        

    def optimization_process(self, constructor:Any, data:pd.DataFrame, cue_groups:Dict[int,List[Any]]):
        agents_paths_clusters = constructor.run_model(data, cue_groups)
        self.agent_group = TSPAgents(self, agents_paths_clusters)
        self.session_duration = get_session_duration(agents_paths_clusters)
        self.ready_event.succeed()

        # while True: 
        #     yield self.env.timeout(self.session_duration)
        #     constructor.run_model(data,cue_groups) 


    def user_movement_process(self, cues:Any, trials:int):
        yield self.ready_event 
        # while True:
        cues.simulate()  # no timeout here
            # yield self.env.timeout(0)
        


    def agent_movement_process(self, agents:Any, trials:int): 
        yield self.ready_event
        agents.simulate()
        # yield self.env.timeout(1)


    def update(self, cues, trials):
        self.user_movement_process(cues,trials)
        self.agent_movement_process(self.agent_group, trials)


    def simulations(self, frame, constructor:Any, cues:Any, map:Any, data:pd.DataFrame, trials:int=500):
        if not hasattr(self, "processes_initialized"):
            self.env.process(cues.simulate())  # SimPy coroutine, scheduled once
            self.env.process(self.agent_group.simulate())
            self.processes_initialized = True

        # self.env.process(self.user_movement_process(cues, trials))
        if self.env.now == self.session_duration : 
            self.optimization_process(constructor, data, cues.group)
        
        self.env.step()

        # return cues
        # self.fig,self.ax = cues.plot_users(map)

        # self.env.process(self.optimization_process(constructor, data, cues.group))
        
        # # Initialize agents AFTER ready
        # def launch_agents(env):
        #     yield self.ready_event
        #     agent_group = TSPAgents(env, self.ax, self.paths)
        #     self.env.process(self.agent_movement_process(agent_group, trials))
        
        # cues.simulate()
        # self.ready_event.succeed()
        
        # # self.env.process(self.optimization_process(constructor, data, cues.group))
        # self.env.process(self.user_movement_process(cues, trials))
        
        # # self.env.process(self.agent_movement_process(trials))
        # self.env.process(launch_agents(self.env))
        # self.env.run(until=trials)


    
