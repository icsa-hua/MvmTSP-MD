from pulp import LpProblem, LpMinimize, lpSum, LpBinary, LpStatus, LpVariable
from dummy_mvmtsp.construct_mvmtsp import ConstrainedMVMTSP 
import numpy as np 
import warnings

def call_mvmtsp_constructor(*,enable_dictionary:bool = False, useGA:bool = True ,number_of_agents:int=5, max_battery:int=1500, max_memory:int=2*1024*1024*1024):
    
    problem = ConstrainedMVMTSP(data=None, use_GA=useGA, regionalization=True, dictionary_flag=enable_dictionary, max_memory=max_memory)
    
    dist_path = 'swarms/distance_cost.csv'
    energy_path = 'swarms/energy_cost.csv'
    nodes_path = 'swarms/areas.csv'
    customers = 'swarms/customers.csv'
    
    data = problem.preprocess(cost_path_1=dist_path, cost_path_2=energy_path,
                              nodes_path=nodes_path, agents=number_of_agents, 
                              customers_path=customers,
                              max_battery=max_battery)
    
    enabled_constraints = ['const_1', 'const_2', 'const_3', 'const_4',
                           'const_5', 'const_6', 'const_7', 'const_8',
                           'const_9']
    
    problem.run_model(data=data,enabled_constraints=enabled_constraints)
    print("Problem has been solved! Extracting paths...")
    
    solution = problem.get_solution()
    problem.get_performance() 
    
    return solution


def main(): 
    
    try: 
         with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            max_memory = 2 * 1024 *1024 *1024
            call_mvmtsp_constructor(enable_dictionary=False, useGA=False, number_of_agents=3, max_battery=1500,max_memory=max_memory)
    except: 
        print("Calling the constructor failed!")


if __name__ == "__main__":
    main()

