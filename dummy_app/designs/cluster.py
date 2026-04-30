from __future__ import annotations
from dummy_app.tools.logger import logger
import dummy_app.tools.common as common
from dummy_app.models.genetic_algorithm import GASolution
from dummy_app.models.milp.extract_solution import extract_cluster_solution, get_node_visits as record_node_visit
from dummy_app.models.milp.objectives import (
    set_energy_objective as apply_energy_objective,
    set_hybrid_objective as apply_hybrid_objective,
    set_makespan_objective as apply_makespan_objective,
    set_max_coverage_objective as apply_max_coverage_objective,
    set_pareto_energy_objective as apply_pareto_energy_objective,
    set_sum_return_times_objective as apply_sum_return_times_objective,
)
from dummy_app.models.milp.scenario_constraints import apply_scenario_constraints
from dummy_app.models.milp.variables import create_problem_variables

import os
import math 
import pandas as pd 
import numpy as np 
import pulp as pl 
import networkx as nx 

from copy import deepcopy
from collections import defaultdict
from typing import Dict, Tuple, List, Any


class Cluster: 

    def __init__(self, cluster:pd.DataFrame, id:int, assignment:List[int], depot_id:int, max_battery): 
        self.cluster = cluster 
        self.id = id
        self.employed_agents:List[int] = assignment 
        self.max_battery = max_battery
        self.nodes_dict = {} 
        self.timeframe = [] 
        self.bridge_nodes:List[int] = []
        self.initial_population:Dict[int, Tuple[List[int], float]] = {} 
        self.depot_id = depot_id
        self.tr_times:Dict[(Tuple[int,int],int)] = {}
        self.cost = {}
        self.virtual_nodes = defaultdict()
        self.allowed_visits = {}
        self.problem = pl.LpProblem()
        self.R = defaultdict(float) 
        self.sinr = defaultdict(float)
        self.max_durations = defaultdict(int)
        self.original_nodes_dict = self.nodes_dict 
        self.V_nodes = list() 
        self.NODES = list() 
        self.makespan_value:float = 0.0
        self.total_data_achievable:float = 0.0
        self.builder_objective_weights:Dict[str, float] = {}
        self.solve_metadata:Dict[str, Any] = {}
        

    def get_cluster_content(self, distance, energy, time, column_names )->Dict:

        # Get the available data that are assoiated to the areas of the cluster
        context = common.extract_context_for_cluster(
            cluster=self.cluster, 
            columns=[
                distance, 
                energy, 
                time, 
                ['Area_id']
            ], 
            column_names=column_names
        )

        return context 
    

    def prepare_context(self, context:Dict, builder:Any): 

        """ 
        Prepare virtual node installation and use Central-Hybs to find
        the bridge nodes and use GA for the initial path and estimate 
        Time Window for the cluster to solve. 
        """

        self.cost, self.virtual_nodes, self.bridge_nodes, self.nodes_dict,raw_population =common.process_extraction(
            problem_builder=builder, 
            extraction=context,
            depot=self.depot_id, 
            employed_agents=self.employed_agents,
        )
        
        self.initial_population = {
            k: (list(v[0]), float(v[1])) for k, v in raw_population.items()
        }

        for vn,hub in self.virtual_nodes.items(): 
            if hub in self.allowed_visits: 
                self.allowed_visits[hub] +=1 
            else:
                self.allowed_visits[hub] = 1


    def get_estimated_time_frame(self, builder:Any): 
        total_time = 0 

        # NOTE: the GA is not enabled, no initial population of paths is generated.
        # This does not account for scenario or coverage mandatory time. 
        if not self.initial_population: 
            
            G = GASolution.create_model_graph(
                cost=self.cost['travel_time'], 
                nodes=self.nodes_dict, 
                weights={'travel_time':1}
            )
            
            # Find the minimum spanning tree to consider it as a solution and estimate the time frame for the cluster
            mst = nx.minimum_spanning_tree(G, weight='weight')
            estimated_time = sum(edge[2]['weight'] for edge in mst.edges(data=True))
            total_time = math.ceil(estimated_time)

        else: 
            # Get the travel time baed on the GA paths considering the scenario and the coverage wait time.  
            best_agent = min(self.initial_population.items(), key=lambda item: item[1][1])
            best_path = best_agent[1][0]

            if builder.scenario == 'cooperative': 
                num_travels = int((len(best_path) - 1) /len(self.employed_agents))

            elif builder.scenario == 'individual':
                num_travels = len(best_path) - 1 

            else: num_travels = len(best_path) 

            if hasattr(builder, 'get_travel_time'):
                total_time = math.ceil(sum(
                    self.get_travel_times(i, i+1, best_path, builder)
                    for i in range(len(best_path)-1)
                    )) + num_travels * builder.coverage_time
 
        if total_time == 0: 
            logger.error(f"Total time is 0 for cluster {self.id}")
            raise ValueError(f"Total time is 0 for cluster {self.id}")
        
        self.timeframe = list(range(0, total_time + 1))


    def problem_formulation(self, builder, scenario:str='cooperative', objective_function:str='energy', stage_solution:int=2): 
        

        if not hasattr(builder, 'get_travel_time'):
            logger.error("Builder does not have get_travel_time method") 
            raise ValueError("Builder does not have get_travel_time method")    
        
        V_nodes = list(self.original_nodes_dict.keys())
        logger.info("Number of Cluster's vertices:: {0}".format(len(V_nodes)))

        # Get duration of each trip (arc) 
        self.tr_times = {(self.original_nodes_dict[i],self.original_nodes_dict[j]):self.get_travel_times(i, j, self.original_nodes_dict, builder) for i in V_nodes for j in V_nodes}
        self.builder_objective_weights = getattr(builder, "objective_weights", {})

        # Set the decision variables 
        self.create_problem(scenario=scenario, objective_functions=objective_function) 

        employed_agents = ["Agent_" + str(i) for i in self.employed_agents]
        list_of_agents = {name: int(name.split("_")[1]) for name in employed_agents}
        
        #NOTE: Pareto is not fully functional. 
        # if objective_function == "pareto":

        #     if scenario != 'cooperative':
        #         logger.error("Pareto objective function is only available for cooperative scenario")
        #         raise ValueError("Pareto objective function is only available for cooperative scenario")
        #     import pdb;pdb.set_trace()
        #     self.problem.setObjective(self.makespan) 
        #     cooperative_scenario_constraints(cluster=self,builder=builder, V_nodes=self.V_nodes, list_of_agents=list_of_agents)

        #     builder.solve_problem(self) 

        #     min_makespan = self.makespan.varValue
        #     logger.info(f"Minimum makespan found: {min_makespan}")

        #     self.set_pareto_energy_objective(energy=self.cost['energy'])

        #     builder.solve_problem(self) 
        #     min_spatial_cost =  self.spatial_cost.value() 

        #     max_makespan = self.makespan.varValue

        #     logger.info(f"Maximum makespan found: {max_makespan}")

        #     pareto_points = [] 

        #     num_points_on_front = 15 

        #     epsilon_values = np.linspace(min_makespan, max_makespan, num_points_on_front)

        #     for epsilon in epsilon_values: 

        #         self.problem.setObjective(self.spatial_cost) 

        #         constraint_name = "makespan_epsilon_constraint"
        #         if constraint_name in self.problem.constraints:
        #             del self.problem.constraints[constraint_name] 

        #         self.problem += self.makespan <= epsilon, constraint_name
        #         logger.info(f"Setting makespan upper bound to {epsilon}")
        #         builder.solve_problem(self)
        #         if self.problem.status != pl.LpStatusOptimal:
        #             logger.warning(f"Problem is not optimal for epsilon {epsilon}, skipping...")
        #             continue

        #         cost_result = self.spatial_cost.value() 
        #         makespan_result = self.makespan.varValue 
        #         logger.info(f"Pareto point found: (Cost: {cost_result}, Makespan: {makespan_result})")
        #         pareto_points.append((cost_result, makespan_result))


        #     import matplotlib.pyplot as plt
        #     if pareto_points:
        #         # Unzip the list of tuples into separate lists for plotting
        #         makespan_values, cost_values = zip(*sorted(pareto_points))

        #         plt.figure(figsize=(10, 6))
        #         plt.plot(makespan_values, cost_values, marker='o', linestyle='-', color='b')
        #         plt.xlabel("Makespan (Total Time)")
        #         plt.ylabel("Total Energy Cost")
        #         plt.title("Pareto Front: Trade-off between Time and Energy")
        #         plt.grid(True)
        #         plt.show()

        #     return 

        apply_scenario_constraints(
            cluster=self,
            builder=builder,
            scenario=scenario,
            list_of_agents=list_of_agents,
            subtour_strategy=getattr(builder, "subtour_strategy", "mtz"),
        )
        
        # STAGE 1 SOLUTION : ONLY Objective Functions to solve
        if stage_solution == 1: 

            if objective_function == 'energy':
                self.set_hybrid_objective(
                    distance=self.cost['distance'],
                    energy=self.cost['energy'],
                    time=self.cost['travel_time'],
                )

            elif objective_function == "coverage":
                self.set_max_coverage_objective(builder)
                # self.set_makespan_objective(
                #     distance=self.cost['distance'],
                #     energy=self.cost['energy'],
                #     time=self.cost['travel_time']
                # ) 

            builder.solve_problem(self)

            # NOTE: In the case the solution from the MILP solver is indeed optimal, this should terminate the implenetation. 
            # Remove it to execute normally. This is just a check. 
            if builder.validate: 
                self.validate_solution(objective_function=objective_function, builder=builder)

            self.makespan_value = self.makespan.varValue
            self.total_data_achievable = self.total_data_collected_main.value() 

            logger.info(f"{objective_function} optimization returned makespan: {self.makespan_value} and total data collected: {self.total_data_achievable}")


        # STAGE 2 SOLUTION : Objective Function + Second Objective Function.
        elif stage_solution == 2: 

            if objective_function == 'energy':
                self.set_hybrid_objective(
                    distance=self.cost['distance'],
                    energy=self.cost['energy'], 
                    time=self.cost['travel_time'],
                )

                builder.solve_problem(self) 

                if self.problem.status != pl.LpStatusOptimal:
                    raise Exception("Could not solve for the initial bound in Stage 1.")

                feasible_makespan = self.makespan.varValue
                C_optimal = self.total_cost.value()

                logger.info(f"Found a tight, feasible makespan bound of: {feasible_makespan}")
                self.problem += self.makespan <= feasible_makespan 

                delta = 0.20 # Percentage of huw much energy we can compromise for a better makespan.
                self.problem += self.total_cost <= C_optimal * (1 + delta)

                self.set_makespan_objective(
                    distance=self.cost['distance'],
                    energy=self.cost['energy'],
                    time=self.cost['travel_time']
                )
                builder.solve_problem(self)

            elif objective_function == "coverage":
                isolation_thr = 0.85
                self.set_max_coverage_objective(builder)
                builder.solve_problem(self)

                if self.problem.status != pl.LpStatusOptimal:
                    raise Exception("Could not solve for the initial bound in Stage 2.")

                max_possible_data = self.total_data_collected_main.value()
                feasible_makespan = self.makespan.varValue

                logger.info(f"The theoretical maximum data collection for this scenario is: {max_possible_data}")

                dynamic_quota = max_possible_data * isolation_thr

                self.problem += self.total_data_collected_main >= dynamic_quota, "Data_Quota_Constraint"
                self.problem += self.makespan <= feasible_makespan 

                self.set_makespan_objective(
                    distance=self.cost['distance'],
                    energy=self.cost['energy'],
                    time=self.cost['travel_time']
                )
                builder.solve_problem(self)

            self.makespan_value = self.makespan.varValue
            self.total_data_achievable = self.total_data_collected_main.value() 

            logger.info(f"{objective_function} optimization returned makespan: {self.makespan_value} and total data collected: {self.total_data_achievable}")

            
        # elif stage_solution == 3: 
        #     # Stage 1 
        #     self.set_hybrid_objective(
        #             distance=self.cost['distance'],
        #             energy=self.cost['energy'], 
        #             time=self.cost['travel_time'],
        #     )
            
        #     builder.solve_problem(self) 

        #     if self.problem.status != pl.LpStatusOptimal:
        #         raise Exception("Could not solve for the initial bound in Stage 1.")

        #     # This makespan bound will be MUCH tighter and more useful!
        #     feasible_makespan = self.makespan.varValue
        #     logger.info(f"Found a tight, feasible makespan bound of: {feasible_makespan}")

        #     # Stage 2
        #     isolation_thr = 0.85 # Achieve 85% of the maximum coverage 
        #     self.set_max_coverage_objective(builder)
        #     builder.solve_problem(self)

        #     if self.problem.status != pl.LpStatusOptimal:
        #         raise Exception("Could not solve for the initial bound in Stage 2.")

        #     max_possible_data = self.total_data_collected_main.value()
        #     logger.info(f"The theoretical maximum data collection for this scenario is: {max_possible_data}")

        #     # Stage 3 
        #     dynamic_quota = max_possible_data * isolation_thr
        #     makespan_bound = feasible_makespan * 0.99

        #     self.problem += self.total_data_collected_main >= dynamic_quota
        #     # self.problem += self.makespan <= makespan_bound 
            
        #     self.set_makespan_objective(
        #         distance=self.cost['distance'],
        #         energy=self.cost['energy'],
        #         time=self.cost['travel_time']
        #     )

        #     builder.solve_problem(self)

        #     makespan = self.makespan.varValue
        #     total_data_collected = self.total_data_collected_main.value() 

        #     logger.info(f"{objective_function} optimization returned makespan: {makespan} and total data collected: {total_data_collected}")


        return self.get_results(builder=builder)
    

    def create_problem(self, scenario:str='cooperative', objective_functions:str="energy")->None: 
        create_problem_variables(self, scenario=scenario)


    def set_hybrid_objective(self, distance, energy, time):
        apply_hybrid_objective(self, distance, energy, time)


    def set_energy_objective(self, distance, energy, time): 
        apply_energy_objective(self, distance, energy, time)
        
        
       
    def set_pareto_energy_objective(self,energy): 
        apply_pareto_energy_objective(self, energy)



    def set_max_coverage_objective(self, builder):
        apply_max_coverage_objective(self, builder)


    def set_sum_return_times_objective(self):
        apply_sum_return_times_objective(self)


    def set_makespan_objective(self, distance, energy, time)->None:
        apply_makespan_objective(self, distance, energy, time)


    def set_up_virtual_nodes_properties(self): 

        reverse_nodes = {v:k for k, v in self.nodes_dict.items()}
        self.original_nodes_dict = deepcopy(self.nodes_dict)
        remove_original_nodes = set(self.virtual_nodes.values()) 

        for node in remove_original_nodes: 
            self.nodes_dict.pop(reverse_nodes[node])

        self.nodes_dict = {i:v for i,(k,v) in enumerate(self.nodes_dict.items())} 
        
        self.V_nodes = list(self.nodes_dict.keys())
        reverse_nodes = {v:k for k, v in self.nodes_dict.items()}

        depot_id = reverse_nodes[self.depot_id]
        self.NODES = self.V_nodes[:depot_id] + self.V_nodes[depot_id+1:]


    def get_average_coverage(self, user_points, altitude, user_height, terrain_type='rural', filename='outuput.csv'):
        from dummy_app.models.coverage import coverage_probability, coverage_u2c

        average_R = defaultdict(float)
        average_sinr = defaultdict(float)
        R = defaultdict(list) 
        sinr = defaultdict(list) 
        user_per_area = defaultdict(int) 
        
        for i in self.nodes_dict.keys():
            
            area = self.nodes_dict[i]
            coords = self.cluster.loc[self.cluster['Area_id'] == area, ['X_coords', 'Y_coords']].values
            
            if area not in user_points: continue
            
            user_per_area[i] = 0 
            
            for user in user_points[area]:
                user_coords = (user.x, user.y)
                horizontal_distance = 0.0                 

                horizontal_distance = np.linalg.norm(np.array(coords) - np.array(user_coords))
                horizontal_distance = horizontal_distance / 1e3 # Convert to km

                height_difference = altitude - user_height
                dist = np.sqrt(horizontal_distance**2 + height_difference**2)

                r_value, sinr_value = coverage_u2c(
                    agent_to_user_dist=dist, 
                    agent_altitude=altitude, 
                    user_altitude=user_height, 
                    agent_pos=coords, 
                    terrain_type=terrain_type
                )
                
                R[i].append(r_value)
                sinr[i].append(sinr_value)
                user_per_area[i] += 1

            # Convert to numpy arrays for easier calculations
            average_R[i] = float(np.mean(R[i])/1e6) # Convert to Mbps
            average_sinr[i] = float(np.mean(sinr[i]))
            logger.debug(f"R: {average_R[i] } Mbps, SINR: {average_sinr[i]} dB for area {area}")

        coverage_summary = coverage_probability(
            self,
            num_users=dict(user_per_area),
            savefile_name=filename,
            directory=None,
            snr=average_sinr,
            save_artifacts=False,
        )

        self.R = average_R
        self.sinr = average_sinr
        return {
            "cluster_id": self.id,
            "avg_rate_mbps_by_node": dict(average_R),
            "avg_sinr_db_by_node": dict(average_sinr),
            "users_per_node": dict(user_per_area),
            "coverage_probability": coverage_summary,
        }


    def get_node_visits(self, builder:Any, node, dc):
        record_node_visit(self, builder, node, dc)

             
    def get_results(self,builder:Any): 
        return extract_cluster_solution(self, builder)


    def get_travel_times(self,i,j, nodes, builder): 
         
        source = nodes[i] 
        target = nodes[j]
        if  nodes[i] in self.virtual_nodes:
            source = self.virtual_nodes[nodes[i]]
            
        if nodes[j] in self.virtual_nodes: 
            target = self.virtual_nodes[nodes[j]]

        return math.ceil(builder.travel_cost[source, target])
             

    def check_results_file(self, name, directory, type): 
        parent_dir = os.getcwd() 
        assets_dir = os.path.join(parent_dir, 'assets') 
        results_dir = os.path.join(assets_dir, 'results')
        if not os.path.exists(results_dir): 
            os.mkdir(results_dir)

        dd = os.path.join(results_dir, directory) 
        if not os.path.exists(dd):
            os.mkdir(dd)
        accepted_files = ['csv', 'txt', 'png', 'jpg']
        
        if type not in accepted_files: 
            raise ValueError(f"Invalid file type. Accepted file types are: {', '.join(accepted_files)}")
        
        results_file = os.path.join(results_dir, name + f'.{type}')
        if os.path.exists(results_file):
            os.remove(results_file)

 
    def validate_solution(self, objective_function, builder): 
        if objective_function == 'energy':
            const = self.total_cost.value()

            self.problem += self.total_cost <= const - 0.00001

            builder.solve_problem(self)

            if self.problem.status != pl.LpStatusOptimal:
                raise Exception("Solution is Truly Optimal")

        elif objective_function == "coverage":
            total_data_collected = self.total_data_collected_main.value()
            self.problem += self.total_data_collected_main >= total_data_collected + 0.00001

            builder.solve_problem(self)

            if self.problem.status != pl.LpStatusOptimal:
                raise Exception("Solution is Truly Optimal")
