from pulp import LpProblem, LpMinimize, lpSum, LpBinary, LpStatus, LpVariable
from dummy_mvmtsp.construct_mvmtsp import ConstrainedMVMTSP 
from gui.configuration import ConfigApp
import numpy as np 
import warnings
import tkinter as tk
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

def call_mvmtsp_constructor(*,enable_dictionary:bool = False, useGA:bool = True, regionalization:bool = True ,number_of_agents:int=5, max_battery:int=1500, max_memory:int=2*1024*1024*1024, constraints=[]):

    config = {'regionalization':regionalization,
              'genetic_algorithm': useGA,
              'individual_solution':enable_dictionary, 
              'enabled_constraints':constraints}
    
    problem = ConstrainedMVMTSP(data=None, config=config, max_memory=max_memory)
    
    dist_path = 'swarms/distance_cost.csv'
    energy_path = 'swarms/energy_cost.csv'
    nodes_path = 'swarms/areas.csv'
    customers = 'swarms/customers.csv'
        
    data = problem.preprocess(cost_path_1=dist_path, cost_path_2=energy_path,
                              nodes_path=nodes_path, agents=number_of_agents, 
                              customers_path=customers,
                              max_battery=max_battery)
    
    
    
    problem.run_model(data=data)
    logger.info("Problem has been solved! Extracting paths...")
    
    solution = problem.get_solution()
    problem.get_performance() 
    problem.get_statistics()
    
    return solution


def main(): 
    logger.info("Preparing the GUI setup...")
    try: 
        with warnings.catch_warnings():
            root = tk.Tk()
            app = ConfigApp(root)
            root.mainloop()
            logger.info("Reading Input configuration...")
            warnings.simplefilter("ignore", RuntimeWarning)
            max_memory = 2 * 1024 *1024 *1024
            enabled_constraints = ['const_1', 'const_2', 'const_3', 'const_4',
                                   'const_5', 'const_6', 'const_7', 'const_8',
                                   'const_9']
            logger.info("Starting Optimization process...")
            
            call_mvmtsp_constructor(enable_dictionary=app.enable_individual.get(),
                                    useGA=app.use_ga.get(),
                                    regionalization=app.regionalization.get(),
                                    number_of_agents=app.number_of_agents.get(),
                                    max_battery=app.max_battery.get(),
                                    max_memory=max_memory,
                                    constraints=enabled_constraints)
    except: 
        logging.ERROR("Calling the constructor failed!")


if __name__ == "__main__":
    main()

