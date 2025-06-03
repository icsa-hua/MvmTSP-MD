import simpy
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Any, List, Dict, Optional
from dummy_app.tools.logger import logger
from dummy_app.tools.common import get_session_duration
from dummy_app.designs.agents import TSPAgents


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
        
        self.agent_group:Optional[TSPAgents] = TSPAgents(self.env,{},empty=True)
        
        logger.debug("Initialized EnvSim and main figure...")
    
    
    def __init_plot(self) -> tuple[Figure, Axes]:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_title("Environment Plot")
        ax.grid(True)
        # plt.close()
        return fig, ax
    

    def optimization_process(self, constructor:Any, distance_matrix:np.ndarray, data:pd.DataFrame, cue_groups:Dict[int,List[Any]]):
        logger.debug("Running optimization process...")
        
        agents_paths_clusters = constructor.run_model(distance_matrix, data, cue_groups)
        self.agent_group = TSPAgents(self, agents_paths_clusters)
        self.session_duration += get_session_duration(agents_paths_clusters)
        constructor.Time += self.session_duration 
        self.ready_event.succeed()
        self.optimization_guard_flag = False 
        
        logger.debug(f"Session duration: {self.session_duration}")
        import pdb;pdb.set_trace() 
        

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
        map:Any,
        distance_matrix:np.ndarray,
        data:pd.DataFrame, 
        trials:int=500
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
            logger.info(f"Session duration reached: {self.session_duration} with simulator time {self.env.now}")
            if self.ax.legend_:
                self.ax.legend_.remove() # Remove old legend if needed
            self.ax.legend()
            if hasattr(self,"agent_group") and self.agent_group is not None: 
                for line in self.agent_group.path_lines:
                    line.remove()
                # self.agent_group.path_lines.clear()

            self.optimization_process(constructor, distance_matrix, data, cues.group )

        self.env.step()

        return cues
    

    
