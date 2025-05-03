import simpy 
import pandas as pd 

class EnvSim:

    def __init__(self)->None: 
        self.env = simpy.Environment()
        self.ready_event = self.env.event()


    def optimization_process(self, constructor:object, data:pd.DataFrame, cue_groups:dict):
        constructor.run_model(data, cue_groups)
        self.ready_event.succeed()

        while True: 
            yield self.env.timeout(100)
            constructor.run_model(data,cue_groups) 


    def user_movement_process(self, cues:object, trials:int):
        yield self.ready_event 
        cues.run(trials)


    def simulations(self, constructor:object, cues:object, data:pd.DataFrame, trials:int=3000):
        self.env.process(self.optimization_process(constructor, data, cues.group))
        self.env.process(self.user_movement_process(cues, trials))
        self.env.run(until=trials)

