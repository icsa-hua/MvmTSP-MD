import simpy 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dummy_app.tools.autonomize import get_session_duration

class EnvSim:

    def __init__(self)->None: 
        self.env = simpy.Environment()
        self.ready_event = self.env.event()
        self.paths = {} 
        self.session_duration = 0 
        self.fig, self.ax = None,None

    def optimization_process(self, constructor:object, data:pd.DataFrame, cue_groups:dict):
        paths = constructor.run_model(data, cue_groups)
        self.session_duration = get_session_duration(paths)
        self.ready_event.succeed()
        print(self.session_duration)
        time.sleep(10)

        while True: 
            yield self.env.timeout(self.session_duration)
            constructor.run_model(data,cue_groups) 


    def user_movement_process(self, cues:object, trials:int):
        yield self.ready_event 
        cues.simulate()


    def agent_movement_process(self, trials:int): 
        yield self.ready_event


    def simulations(self, frame, constructor:object, cues:object, map:object, data:pd.DataFrame, trials:int=500):
        
        # self.fig,self.ax = cues.plot_users(map)
        self.env.process(self.optimization_process(constructor, data, cues.group))
        self.env.step()
        self.user_movement_process(cues,trials)


        # cues.simulate()

        # self.ready_event.succeed()
        
        # # self.env.process(self.optimization_process(constructor, data, cues.group))
        # self.env.process(self.user_movement_process(cues, trials))
        
        # # self.env.process(self.agent_movement_process(trials))
    
        # self.env.run(until=trials)

