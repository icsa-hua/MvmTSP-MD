import simpy
import pandas as pd 
import matplotlib.pyplot as plt
from typing import Any, List, Dict
from dummy_app.tools.logger import logger
from dummy_app.tools.common import get_session_duration
from dummy_app.designs.agents import TSPAgents


class EnvSim:

    def __init__(self)->None: 
        self.env = simpy.Environment()
        self.ready_event = self.env.event()
        self.session_duration = 0 
        self.optimization_guard_flag = False 
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title("Shared Plot")
        self.ax.grid(True)        
        self.agent_group = TSPAgents(self, {})
        logger.debug("Initialized EnvSim and main figure...")
        

    def optimization_process(self, constructor:Any, data:pd.DataFrame, cue_groups:Dict[int,List[Any]]):
        logger.debug("Running optimization process...")
        agents_paths_clusters = constructor.run_model(data, cue_groups)
        self.agent_group = TSPAgents(self, agents_paths_clusters)

        self.session_duration += get_session_duration(agents_paths_clusters)
        self.ready_event.succeed()
        self.optimization_guard_flag = False 
        logger.debug(f"Session duration: {self.session_duration}")


    def user_movement_process(self, cues:Any, trials:int):
        yield self.ready_event 
        cues.simulate()  # no timeout here
        

    def agent_movement_process(self, agents:Any, trials:int): 
        yield self.ready_event
        agents.simulate()


    def simulations(self, frame, constructor:Any, cues:Any, map:Any, data:pd.DataFrame, trials:int=500):
        if not hasattr(self, "processes_initialized"):
            self.env.process(self.user_movement_process(cues,trials))  # SimPy coroutine, scheduled once
            self.env.process(self.agent_movement_process(self.agent_group, trials))  # SimPy coroutine, scheduled once
            self.processes_initialized = True

        # self.env.process(self.user_movement_process(cues, trials))
        if self.env.now == self.session_duration and not self.optimization_guard_flag:           
            self.optimization_guard_flag = True
            self.ready_event = self.env.event()
            logger.info(f"Session duration reached: {self.session_duration} with simulator time {self.env.now}")
            self.ax.legend().remove()  # Remove old legend if needed
            self.ax.legend()    
            self.optimization_process(constructor, data, cues.group)

        self.env.step()

        return cues
    

    
