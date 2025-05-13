{
1: [],
2: [],
3: [[(8, 3, 0),
    (3, 4, 3),
    (4, 30, 4),
    (30, 29, 6),
    (29, 32, 8),
    (32, 28, 10),
    (28, 14, 11),
    (14, 3, 13),
    (3, 8, 16)]],
4: [[(8, 3, 0),
    (3, 14, 3),
    (14, 28, 4),
    (28, 4, 5),
    (4, 30, 7),
    (30, 29, 9),
    (29, 32, 11),
    (32, 3, 13),
    (3, 8, 16)]],
5: [[(8, 3, 0),
    (3, 32, 3),
    (32, 29, 5),
    (29, 30, 7),
    (30, 4, 9),
    (4, 28, 10),
    (28, 14, 12),
    (14, 3, 14),
    (3, 8, 16)]]
}


Next run (constraint 15 is the one that right now defines the paths so that they all end in the same depot)

{1: [],
 2: [[(8, 3, 0),
   (3, 4, 3),
   (4, 30, 4),
   (30, 29, 6),
   (29, 32, 8),
   (32, 28, 10),
   (28, 14, 11),
   (14, 3, 13),
   (3, 8, 16)]],
 3: [[(8, 3, 0),
   (3, 14, 3),
   (14, 28, 4),
   (28, 4, 5),
   (4, 30, 7),
   (30, 29, 9),
   (29, 32, 11),
   (32, 3, 13),
   (3, 8, 16)]],
 4: [],
 5: [[(8, 3, 0),
   (3, 32, 3),
   (32, 29, 5),
   (29, 30, 7),
   (30, 4, 9),
   (4, 28, 10),
   (28, 14, 12),
   (14, 3, 14),
   (3, 8, 16)]]}


   Without const 7 and const 8 
   {1: [],
    2: [[(8, 3, 0),
      (3, 4, 3),
      (4, 30, 4),
      (30, 29, 6),
      (29, 32, 8),
      (32, 28, 10),
      (28, 14, 11),
      (14, 3, 13),
      (3, 8, 16)]],
    3: [[(8, 3, 0),
      (3, 14, 3),
      (14, 28, 4),
      (28, 4, 5),
      (4, 30, 7),
      (30, 29, 9),
      (29, 32, 11),
      (32, 3, 13),
      (3, 8, 16)]],
    4: [[(8, 3, 0),
      (3, 32, 3),
      (32, 29, 5),
      (29, 30, 7),
      (30, 4, 9),
      (4, 28, 10),
      (28, 14, 12),
      (14, 3, 14),
      (3, 8, 16)]],
    5: []}



    const_10 without +1 -1 
    {1: [[(8, 3, 0),
        (3, 32, 2),
        (32, 14, 3),
        (14, 30, 4),
        (30, 3, 5),
        (3, 32, 6),
        (32, 14, 7),
        (14, 30, 8),
        (30, 3, 9),
        (3, 32, 10),
        (32, 14, 11),
        (14, 30, 12),
        (30, 3, 13),
        (3, 32, 14)]],
      2: [[(8, 3, 0),
        (3, 4, 2),
        (4, 28, 3),
        (28, 14, 4),
        (14, 3, 5),
        (3, 4, 6),
        (4, 28, 7),
        (28, 14, 8),
        (14, 3, 9),
        (3, 4, 10),
        (4, 28, 11),
        (28, 14, 12),
        (14, 3, 13),
        (3, 4, 14)]],
      3: [],
      4: [[(8, 3, 0),
        (3, 30, 2),
        (30, 28, 3),
        (28, 29, 4),
        (29, 14, 5),
        (14, 32, 6),
        (32, 4, 7),
        (4, 3, 8),
        (3, 30, 9),
        (30, 28, 10),
        (28, 29, 11),
        (29, 14, 12),
        (14, 32, 13),
        (32, 4, 14)]],
      5: []}


      {1: [],
        2: [],
        3: [[(8, 3, 0),
          (3, 30, 2),
          (30, 28, 3),
          (28, 29, 4),
          (29, 14, 5),
          (14, 32, 6),
          (32, 4, 7),
          (4, 3, 8),
          (3, 30, 9),
          (30, 28, 10),
          (28, 29, 11),
          (29, 14, 12),
          (14, 32, 13),
          (32, 4, 14)]],
        4: [[(8, 3, 0),
          (3, 29, 2),
          (29, 28, 3),
          (28, 4, 4),
          (4, 32, 5),
          (32, 14, 6),
          (14, 30, 7),
          (30, 3, 8),
          (3, 29, 9),
          (29, 28, 10),
          (28, 4, 11),
          (4, 32, 12),
          (32, 14, 13),
          (14, 30, 14)]],
        5: [[(8, 3, 0),
          (3, 4, 2),
          (4, 28, 3),
          (28, 14, 4),
          (14, 3, 5),
          (3, 4, 6),
          (4, 28, 7),
          (28, 14, 8),
          (14, 3, 9),
          (3, 4, 10),
          (4, 28, 11),
          (28, 14, 12),
          (14, 3, 13),
          (3, 4, 14)]]}


          









          def set_constraints_for_multi_agent(self, V_nodes:List[int], nodes_dict:Dict[int, int], R_points:List[int]): 

          logger.info(f"Setting constraints for multi-agent problem...")
          employed_agents = ["Agent_" + str(agent_id) for agent_id in self.employed_agents]
          list_of_agents = {x:int(x.split('_')[-1]) for x in employed_agents}
          header = list_of_agents[employed_agents[0]]
          reverse_nodes = {v: k for k, v in nodes_dict.items()}
          
          # NOTE: depot_ind is confirmed to be correct. 
          depot_ind = self.get_depot_index(nodes_dict, header)
          valid_arcs = [(i,j) for i in V_nodes for j in V_nodes if i != j and i != depot_ind and j != depot_ind]
  
          in_arcs = defaultdict(list)
          out_arcs = defaultdict(list)
  
          for i, j in valid_arcs:
              out_arcs[i].append(j)
              in_arcs[j].append(i)
  
          self.tr_times = {(i,j):self.get_travel_time(i,j,nodes_dict) for i in V_nodes for j in V_nodes}
          T_max = max(self.tr_times[(i,depot_ind)] for i in V_nodes if i != depot_ind)
  
          deallocate_memory(valid_arcs)
  
          if self.enable_ga: # NOTE: Finalized 
              for a in self.employed_agents: 
                  for i in range(len(self.initial_population[a][0])-1): 
                      node = reverse_nodes[self.initial_population[a][0][i]]
                      next_node = reverse_nodes[self.initial_population[a][0][i+1]] 
                      self.x[node, next_node, header].setInitialValue(1) 
  
          if "const_0" in self.constraints: #NOTE: FINALIZED 
              try: 
                  for k, v in list_of_agents.items():
                      for j in in_arcs: 
                          self.problem += pl.lpSum(
                              self.x[i,j,v]
                              for i in in_arcs[j]
                          ) <= R_points[j], f"Allowed_visits_for_each_agent_{k}_for_node_{j}"
              
                  logger.debug(f"Constraint | const_0 - All nodes visited multiple times in total | set for cluster ")
              except Exception as e:
                  logger.exception(f"Error setting constraint const_0 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_0 for cluster")
  
          if "const_1" in self.constraints:#NOTE: FINALIZED
              try: 
                  for k, v in list_of_agents.items():
                      self.problem += pl.lpSum(
                          self.x[depot_ind, j, v]
                          for j in V_nodes if j != depot_ind
                      ) == 1, f"{k}_enters_single_area_from_depot_{depot_ind}"
  
                      self.problem += pl.lpSum(
                          self.x[i, depot_ind, v]
                          for i in V_nodes if i != depot_ind
                      ) == 1, f"{k}_leaves_single_area_to_depot_{depot_ind}"
              
                  logger.debug(f"Constraint | const_1 - Each agent enters and leaves the depot once | set for cluster ")
              except Exception as e:
                  logger.exception(f"Error setting constraint const_1 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_1 for Cluster") 
              
          if "const_2" in self.constraints:#NOTE:FINALIZED
              try:
                  for k, v in list_of_agents.items():
                      self.problem += self.p[v,self.timeFrame_per_cluster[0]] == depot_ind, f"Positional_variable_at_start_of_journey_for_{k}" 
                      self.problem += self.p[v,self.timeFrame_per_cluster[-1]] == depot_ind, f"Positional_variable_at_end_of_joureny_for_{k}"
  
                  logger.debug(f"Constraint | const_2 - Positional variable at start and end of journey (depot)| set for cluster ")
              except Exception as e: 
                  logger.exception(f"Error setting constraint const_2 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_2 for Cluster")
  
          if "const_3" in self.constraints: 
              try: 
  
      
  
                  # NOTE: Very deterministic as to when to start and end the trip. 
                  for k, v in list_of_agents.items(): 
                      self.problem += pl.lpSum(
                          self.t[depot_ind, j, v, self.timeFrame_per_cluster[0]]
                          for j in V_nodes if depot_ind != j
                      ) == 1, f"{k}_leaves_depot_{depot_ind}_at_specific_interval"
  
                      self.problem += pl.lpSum(
                          self.t[i, depot_ind, v, self.timeFrame_per_cluster[-1]]
                          for i in V_nodes if depot_ind != i
                      ) == 1, f"{k}_enters_depot_{depot_ind}_at_specific_interval"
                  
                  logger.debug(f"Constraint | const_3 - Each agent leaves and enters the depot at a specific interval | set for cluster ")
              
              except Exception as e:
                  logger.exception(f"Error setting constraint const_3 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_3 for Cluster")
  
          if "const_4" in self.constraints:  # NOTE: Use it as a subtour elimination constraint. 
              #NOTE: This is redundant if const_1 is enabled 
              try:
                  for k, v in list_of_agents.items():
                      self.problem += pl.lpSum(self.x[depot_ind, j, v] for j in V_nodes if j != depot_ind) + \
                                      pl.lpSum(self.x[i, depot_ind, v] for i in V_nodes if i != depot_ind) == 2, f"{k}_start_&_finishes_at_depot_{depot_ind}"  
                  logger.debug(f"Constraint | const_4 - Each agent starts and ends at the depot | set for cluster ") 
              except Exception as e:
                  logger.exception(f"Error setting constraint const_4 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_4 for Cluster")
          
          if "const_5" in self.constraints:#NOTE:FINALIZED
              try: 
                  for k, v in list_of_agents.items():
                      self.problem += self.x[depot_ind, depot_ind, v] == 0,  f"No_loop_at depot_{depot_ind}_for_{k}_at_any_timepoint"
  
                  logger.debug(f"Constraint | const_5 - No loop at depot | set for cluster ")
              except Exception as e:
                  logger.exception(f"Error setting constraint const_5 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_5 for Cluster")
               
          if "const_6" in self.constraints: #NOTE:FINALIZED
              try: 
                  for k, v in list_of_agents.items(): 
                      for j in V_nodes: 
                          if j != depot_ind and nodes_dict[j] not in self.bridge_nodes: 
                              self.problem += pl.lpSum(self.x[i, j, v] for i in in_arcs[j]) == 1, f"Enter_{j}_for_{k}_exluding_depots"
                              self.problem += pl.lpSum(self.x[j, i, v] for i in out_arcs[j]) == 1, f"Leave_{j}_for_{k}_exluding_depots"
                          elif nodes_dict[j] in self.bridge_nodes:
                              print("In bridge nodes : ",j)
                              self.problem += pl.lpSum(self.x[i,j,v] for i in in_arcs[j]) == R_points[j] 
                              self.problem += pl.lpSum(self.x[j,i,v] for i in out_arcs[j]) == R_points[j]
                  
                  logger.debug(f"Constraint | const_6 - Each agent enters and leaves each node once | set for cluster ") 
              except Exception as e:
                  logger.exception(f"Error setting constraint const_6 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_6 for Cluster")
              
          if "const_7" in self.constraints: 
              try: 
                  for k, v in list_of_agents.items():
                      for i in out_arcs:
                          for j in out_arcs[i]:
                              travel_time = self.tr_times[(i, j)]
                              for t_start in self.timeFrame_per_cluster[self.tr_times[(depot_ind,i)]: -(self.tr_times[(j,depot_ind)] + travel_time)]:
                                  # For each possible start time
                                  steps_range = range(t_start, t_start + travel_time)
                                  for t in steps_range:
                                      self.problem += self.busy[v, t] >= self.t[i, j, v, t_start], \
                                          f"Busy_if_travel_{i}_{j}_starts_at_{t_start}_for_{k}_covers_{t}"
  
                  logger.debug(f"Constraint | const_7 - Busy if travel starts at a time | set for cluster ")
  
              except Exception as e : 
                  logger.exception(f"Error setting constraint const_7 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_7 for Cluster")
              
  
          if "const_8" in self.constraints: #NOTE: Won't work with Many Visits
              try: 
                  for k, v in list_of_agents.items():
                      for t in self.timeFrame_per_cluster:
                          self.problem += self.busy[v, t] <= 1, f"Busy_limit_one_travel_per_step_{t}_{k}"
                  logger.debug(f"Constraint | const_8 - Lower and upper bounds on u | set for cluster ")
              except Exception as e:
                  logger.exception(f"Error setting constraint const_8 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_8 for Cluster")
              
          if "const_9" in self.constraints: #NOTE : This constraint is the problem where the agents can't finish on their designated depot. 
              try: 
                  for k, v in list_of_agents.items():
                      for i in V_nodes:
                          if i != depot_ind:
                              for step in self.timeFrame_per_cluster[1:self.tr_times[(depot_ind, i)]]: 
                                  self.problem += (
                                      self.t[depot_ind, i, v, step] == self.t[depot_ind, i, v, self.timeFrame_per_cluster[0]], 
                                      f"Dynamic_time_enforcement_{depot_ind}_{i}_for_{k}_time_{step}"
                              )
                                  
                      for j in V_nodes:
                          if j != depot_ind:             
                              for step in self.timeFrame_per_cluster[-self.tr_times[(j,depot_ind)]:-1]:
                                  self.problem += (
                                      self.t[j, depot_ind, v, step] == self.t[j, depot_ind, v, self.timeFrame_per_cluster[-1]], 
                                      f"Dynamic_time_enforcement_{j}_to_{depot_ind}_for_{k}_time_{step}"
                                      )
                  logger.debug(f"Constraint | const_9 - Dynamic time enforcement | set for cluster ")
  
              except Exception as e:
                  logger.exception(f"Error setting constraint const_9 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_9 for Cluster")
              
          if "const_10" in self.constraints: #NOTE: This is essential for the model to work 
              try: 
                  for k, v in list_of_agents.items(): 
                      for i in V_nodes: 
                          if i != depot_ind: 
                              for step in self.timeFrame_per_cluster[:(self.tr_times[(depot_ind, i)])]: 
                                  self.problem += (
                                      self.t[depot_ind, i, v, step] == self.x[depot_ind, i, v], 
                                      f"Enforce synchronization_between_t_and_x_{depot_ind}_{i}_for_{k}_time_{step}"
                                  )
  
                      for j in V_nodes: 
                          if j != depot_ind: 
                              for step in self.timeFrame_per_cluster[-(self.tr_times[(j, depot_ind)]):]: 
                                  self.problem += (
                                      self.t[j, depot_ind, v, step] == self.x[j, depot_ind, v],
                                      f"Enforce synchronization_between_t_and_x_{j}_to_{depot_ind}_for_{k}_time_{step}"
                                  )
                  logger.debug(f"Constraint | const_10 - Enforce synchronization between t and x | set for cluster ")
  
              except Exception as e:
                  logger.exception(f"Error setting constraint const_10 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_10 for Cluster")
              
          if "const_11" in self.constraints: 
              try: 
                  # NOTE: The below is the same constraint but with no time window slicing.
                  for k, v in list_of_agents.items():
                      for i in out_arcs:
                          for j in out_arcs[i]:
                              for step in self.timeFrame_per_cluster:
                                  arrival_time = step + self.tr_times[(i, j)]
  
                                  # ensure time index exists
                                  if arrival_time + 1 in self.timeFrame_per_cluster:
                                      self.problem += self.t[i, j, v, arrival_time] <= self.t[j, j, v, arrival_time + 1], \
                                          f"TimeProgress_{i}_{j}_at_{step}_agent_{k}" 
                  logger.debug(f"Constraint | const_11 - Time progression | set for cluster ")
  
              except Exception as e:
                  logger.exception(f"Error setting constraint const_11 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_11 for Cluster")
              
          if "const_12" in self.constraints: # NOTE: Also necessary for the model to work. 
              try:
                  for k, v in list_of_agents.items(): 
                      for i in out_arcs: 
                          for j in out_arcs[i]:
                              for step in self.timeFrame_per_cluster[self.tr_times[(depot_ind,i)]:-(self.tr_times[(i,j)]+self.tr_times[(j,depot_ind)])]:
                                  self.problem += pl.lpSum(
                                      self.t[i, j, v, t]
                                      for t in range(step, step + self.tr_times[(i,j)])
                                  ) == self.tr_times[(i,j)] * self.x[i,j,v], \
                                  f"Link_x_and_t_{i}_{j}_for_{k}_at_time_{step}"
                     
                  logger.debug(f"Constraint | const_12 - Link x and t | set for cluster ")
  
              except Exception as e:
                  logger.exception(f"Error setting constraint const_12 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_12 for Cluster")
  
          if "const_13" in self.constraints: # NOTE: This only seems to have an effect on the first agent it encounters. 
              try:
                  for k, v in list_of_agents.items():
                      for i in out_arcs:
                          for j in out_arcs[i]:
                              self.problem += self.e[j,v] >= self.e[i,v] - self.normalized_battery[nodes_dict[i]-1][nodes_dict[j]-1] * self.x[i,j,v], f"Update_remaining_energy_{i}_{j}_for_{k}"
                              self.problem += self.e[i,v] >= self.normalized_battery[nodes_dict[i]-1][nodes_dict[j]-1] * self.x[i, j, v],f"No_travel_if_low_energy_{i}_{j}_for_{k}"
                          self.problem += self.e[i,v] >= self.normalized_battery[nodes_dict[i]-1][nodes_dict[depot_ind]-1] * self.x[i, depot_ind, v],f"Enough_energy_to_return_to_depot_from_{i}_for_{k}"
                      
                      for i in out_arcs: 
                          for j in out_arcs[i]: 
                              for step in self.timeFrame_per_cluster[self.tr_times[(depot_ind,i)]:-(self.tr_times[(i,j)]+self.tr_times[(j,depot_ind)])]: 
                                  self.problem += self.e[j,v] >= self.e[i,v] - self.normalized_battery[nodes_dict[i]-1][nodes_dict[j]-1] * \
                                  pl.lpSum(self.t[i, j, v, t]
                                          for t in range(step, step + self.tr_times[(i, j)])
                                  ),f"Energy_update_{i}_{j}_at_time_{step}_for_{k}"
                          self.problem += self.e[i,v] >= self.normalized_battery[nodes_dict[i]-1][nodes_dict[j]-1] * pl.lpSum(
                              self.t[i,depot_ind,v,t] for t in self.timeFrame_per_cluster[-self.tr_times[(i,depot_ind)]:]
                          ), f"Ensure_depot_return_from{i}_for_{k}_for_correct_time_Steps"
                  logger.debug(f"Constraint | const_13 - Update remaining energy | set for cluster ")
  
              except Exception as e:
                  logger.exception(f"Error setting constraint const_13 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_13 for Cluster")
  
          if "const_14" in self.constraints: # NOTE this is unnecessary. 
              try: 
                  for k, v in list_of_agents.items():
                      for i in out_arcs:
                          for t in self.timeFrame_per_cluster: 
                              self.problem += self.e[i,v] >= 0, f"Energy_cannot_be_negative_{i}_for_{k}_at_time_{t}"
  
                      self.problem += self.e[depot_ind, v] == self.max_battery_norm, f"Every_agent_starts_with_full_battery_{k}"
  
                  logger.debug(f"Constraint | const_14 - Energy cannot be negative | set for cluster ")
              except Exception as e:
                  logger.exception(f"Error setting constraint const_14 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_14 for Cluster")
  
          if "const_15" in self.constraints:
              try:
                  for i in out_arcs:
                      for j in out_arcs[i]:
                          # NOTE: Slicing the time window even more here may be brittle: since we are already enforcing positional and journey to depots in specific timesteps. 
                          # for step in self.timeFrame_per_cluster[(tr_times[(depot_ind, i)]): -(tr_times[(i, j)] + tr_times[(j, depot_ind)])]:
                          for step in self.timeFrame_per_cluster:
                              self.problem += pl.lpSum(self.t[i,j,v,step] for _,v in list_of_agents.items()) <= 1, f"Unique_Time_visits_constraint_at_travel_{i}_{j}_at_time_{step}" 
  
                  logger.debug(f"Constraint | const_15 - Unique Time visits constraint | set for cluster ")
              except Exception as e:
                  logger.exception(f"Error setting constraint const_15 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_15 for cluster: {e}")
          
          if "const_16" in self.constraints: 
              try: 
                  for k1, v1 in list_of_agents.items() : 
                      for k2, v2 in list_of_agents.items() : 
                          if k1 != k2 : 
                              for step in self.timeFrame_per_cluster: 
                                  self.problem += pl.lpSum(self.t[i, j, v1, step] - self.t[i,j, v2, step] for i in out_arcs for j in out_arcs[i]) != 0, f"Agent_unique_paths_for_{k1}_and_{k2}_at_time_{step}"
                  logger.debug(f"Constraint | const_16 - Agent unique paths | set for cluster ")
  
              except Exception as e: 
                  logger.exception(f"Error setting constraint const_16 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_16 for cluster: {e}")
          
          if "const_17" in self.constraints: 
              try: 
                  for i in out_arcs: 
                      for j in out_arcs[i]: 
                          if nodes_dict[i] not in self.bridge_nodes and nodes_dict[j] not in self.bridge_nodes:
                              for k, v in list_of_agents.items(): 
                                  self.problem += self.x[i,j,v] + self.x[j,i,v] <= 1, f"No_loops_in_path_for_{k}_at_{i}_{j}"
                  logger.debug(f"Constraint | const_17 - No loops in path | set for cluster ")
  
              except Exception as e:
                  logger.exception(f"Error setting constraint const_17 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_17 for cluster: {e}")
              
          if "const_18" in self.constraints: 
              try: 
                  M = len(V_nodes)
                  for k, v in list_of_agents.items(): 
                      for i in out_arcs: 
                          for j in out_arcs[i]: 
                              for step in self.timeFrame_per_cluster[1:-2]: 
                                  self.problem += self.p[v, step] <= j + (1 - self.t[i, j, v, step]) * M, f"Positional_alignment_with_step_{step}_for_{k}_at_{j}{i}"
                                  self.problem += self.p[v, step] >= j - (1 - self.t[i, j, v, step]) * M, f"Positional_alignment_with_step_{step}_for_{k}_at_-{j}{i}"
                  # M = len(V_nodes)
                  # for k, v in list_of_agents.items(): 
                  #     for i in out_arcs: 
                  #         for j in out_arcs[i]: 
                  #             for step in self.timeFrame_per_cluster: 
                  #                 arrival_step = step + self.tr_times[(i,j)]
                  #                 if arrival_step in self.timeFrame_per_cluster: 
                  #                     self.problem += self.p[v, arrival_step] <= j + (1 - self.t[i, j, v, step]) * M, f"Positional_alignment_with_step_{step}_for_{k}_at_{j}{i}"
                  #                     self.problem += self.p[v, arrival_step] >= j - (1 - self.t[i, j, v, step]) * M, f"Positional_alignment_with_step_{step}_for_{k}_at_-{j}{i}"
                  logger.debug(f"Constraint | const_18 - Positional alignment | set for cluster ")
              except Exception as e:
                  logger.exception(f"Error setting constraint const_18 for cluster: {e}")
                  raise ValueError(f"Error in setting constraint const_18 for cluster: {e}") 
  
          if "const_19" in self.constrains: 
              # Agent can leave the depot at any valid time (before it becomes too late to complete a trip)
              for k, v in list_of_agents.items():
                  for j in V_nodes:
                      if j != depot_ind:
                          valid_departure_window = self.timeFrame_per_cluster[:-(self.tr_times[(depot_ind, j)] + self.tr_times[(j, depot_ind)])]
                          self.problem += pl.lpSum(self.t[depot_ind, j, v, t] for t in valid_departure_window) == 1, \
                              f"{k}_leaves_depot_{depot_ind}_within_valid_time"
  
                          valid_return_window = self.timeFrame_per_cluster[-(self.tr_times[(j, depot_ind)] + 1):]
                          self.problem += pl.lpSum(self.t[j, depot_ind, v, t] for t in valid_return_window) == 1, \
                              f"{k}_returns_to_depot_{depot_ind}_within_valid_time"




















                              for k, v in list_of_agents.items():
    # Agent leaves depot exactly once at some valid time
    self.problem += pl.lpSum(
        self.t[depot_ind, j, v, t]
        for j in V_nodes if j != depot_ind
        for t in self.timeFrame_per_cluster[:-(tr_times[(j, depot_ind)] + 1)]
    ) == 1, f"{k}_departs_from_depot_once"

    # Agent returns to depot exactly once at some valid time
    self.problem += pl.lpSum(
        self.t[i, depot_ind, v, t]
        for i in V_nodes if i != depot_ind
        for t in self.timeFrame_per_cluster[-(tr_times[(i, depot_ind)] + 1):]
    ) == 1, f"{k}_returns_to_depot_once"









    Solving problem...:   0%|                                                                                                                                                                             | 0/5 [00:00<?, ?step/s]     dist_1    dist_2    dist_3    dist_4    dist_5    dist_6    dist_7    dist_8    dist_9   dist_10  ...     tt_32     tt_33     tt_34     tt_35     tt_36  X_coords  Y_coords  Area_id                geometry  cluster
0  0.410536  0.276769  0.000000  0.310597  0.418397  0.520683  0.500775  0.302992  0.369197  0.464034  ...  0.230544  0.372662  0.373793  0.516998  0.548937    748590   4436700        3  POINT (748590 4436700)      1.0
1  0.585116  0.532376  0.347225  0.000000  0.191328  0.366351  0.558614  0.348449  0.516929  0.641015  ...  0.183629  0.602338  0.607897  0.701881  0.694646    748600   4437200        4  POINT (748600 4437200)      1.0
2  0.626893  0.510337  0.266745  0.198839  0.385289  0.555995  0.679860  0.459122  0.589607  0.662551  ...  0.032461  0.532188  0.558335  0.684404  0.719731    748350   4437000       14  POINT (748350 4437000)      1.0
3  0.837213  0.717572  0.501156  0.311094  0.494661  0.696652  0.899107  0.710746  0.829175  0.845961  ...  0.245422  0.672541  0.718519  0.830579  0.877534    748100   4437230       28  POINT (748100 4437230)      1.0
4  0.816343  0.695921  0.476451  0.298114  0.484290  0.684063  0.878365  0.686364  0.805798  0.827428  ...  0.223395  0.656123  0.700737  0.814932  0.861653    748120   4437200       29  POINT (748120 4437200)      1.0
5  0.615804  0.464614  0.226158  0.318143  0.491941  0.649752  0.718122  0.515788  0.602640  0.637705  ...  0.151425  0.457141  0.493330  0.638124  0.698157    748280   4436800       30  POINT (748280 4436800)      1.0
6  0.599862  0.489979  0.246554  0.175665  0.358348  0.524431  0.644311  0.418783  0.555565  0.641015  ...  0.000000  0.525826  0.547218  0.671440  0.700904    748400   4437000       32  POINT (748400 4437000)      1.0
7  0.318910  0.311993  0.256143  0.263498  0.249562  0.285321  0.274535  0.000000  0.203695  0.413554  ...  0.331042  0.472313  0.444106  0.530764  0.496714    748900   4436900        8                    None      1.0

[8 rows x 113 columns]
(np.int64(1), 8)
{(np.int64(1), 8): [2, 3, 1], (np.int64(2), 9): [4, 5], (np.int64(4), 8): [2, 3, 1], (np.int64(0), 9): [4, 5], (np.int64(3), 8): [2, 3, 1]}
Solving problem...:   0%|                                                                                                                                                                             | 0/5 [00:00<?, ?step/s]
Time taken: 0.25814080238342285
{}