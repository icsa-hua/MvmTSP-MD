#Optimization MVMTSP problem formulation 
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpStatus, lpDot
import pulp as pl
import pandas as pd
import pdb
import time
import io
import numpy as np
import sys
# import networkx as nx

def pragma():
    return "Hello pragma"

def assign_agents_to_areas(m,D): 
    agents_per_area = m // len(D) 
    remaining_areas = m % len(D) 
    area_agents = {area: agents_per_area for area in D}

    agent_assignments = {agent : 0 for agent in range(1,m+1)} 
    agent_counter = 1
    # pdb.set_trace()
    for area in D: 
        num_agents = area_agents[area]
        for _ in range(num_agents): 
            agent_assignments[agent_counter] = area
            agent_counter += 1 
        

    for i in range(remaining_areas): 
        area = D[i]
        agent_assignments[agent_counter] += area 
        agent_counter += 1

    return agent_assignments

def mvmTSP_problem_formulation(m:int, depots:list ):
    
    # old_stdout = sys.stdout 
    # new_stdout = io.StringIO()
    # sys.stdout = new_stdout

    

    print("Starting the problem formulation...")

    #Creating the Problem Variables 
    m = 3
    depots = [8,9]
    theta = 2000 #Maximum energy capacity. 
    M = 25 #A sufficiently large positive number for BIG-M method  
    V = pd.read_csv('../areas.csv') #Vertices (points and their ids)
    V = V.to_numpy() 
    V_n = V[:,2].astype(np.int8).tolist() #Number of nodes

    D = depots #Starting points. 
    K = range(1, m + 1,1) #[1,2,...,m,m+1]

    c_ij = pd.read_csv('../distance_cost.csv') #Cost of distance between every pair of nodes. 
    c_ij = c_ij.to_numpy() #36x36xm distances

    e_ij = pd.read_csv('../energy_cost.csv') #Cost of energy between every pair of nodes. 
    e_ij = e_ij.to_numpy() #36x36xm energy

    C = pd.read_csv('../customers.csv')
    C = C.to_numpy() #36x1 cities ids

    edges = pd.read_csv('../edges.csv')
    edges = edges.to_numpy() 

    # TS = LpVariable.matrix(
    #     name = "Time", lowBound=1, upBound=146, cat=pl.LpContinuous,
    #     indices=range(1,146)
    # )
    TS = range(1,146,1)
    # required_visits = np.random.randint(low=1, high=m, size=len(V)) #Maximum number of visits that an area has to accept. 
    # if sum(required_visits) > m*(len(V)-len(D)): 
    #     required_visits = [np.floor(m/2)] * len(V)
    required_visits = m * np.ones(len(V)).astype(int)
    areas_in_path = int(theta / np.average(e_ij))
    charge_points = int(np.floor(len(V_n) / areas_in_path) )  
    # required_visits[[d-1 for d in D]] += charge_points 

    #Optimization Variables. 
    x_ijk = LpVariable.dicts("x_ijk", (V_n,V_n,K), 0,1,LpBinary)
    z_ik = LpVariable.dicts("z_ik", (V_n, K), 0,1,LpBinary) 
    y_ik = LpVariable.dicts("y_ik", (V_n, K), 0,1,LpBinary)
    time_cons = LpVariable.dicts("time_cons", (V_n,K, TS), 0,1,LpBinary)
    path_length = LpVariable.dicts("path_length", K, lowBound=0, cat='Continuous')
    # total_distance = LpVariable.dicts("total_distance", K, lowBound=0, cat='Continuous')
    x = LpVariable.dicts("x", (V_n,V_n, K, TS), 0,1,LpBinary)
    e = LpVariable.dicts("e", (V_n),0,theta, pl.LpContinuous)
    #Optimization Problem 
    mvmtsp = LpProblem("MVM_TSP", LpMinimize)
    
    # pdb.set_trace()

    agent_assignment = assign_agents_to_areas(m,D)
    print(agent_assignment)
        
    # #All Constraints 
    # #Constraint N1 == Start from any depot and end at any depot. FINALIZED 
    for k in K:
            d = agent_assignment[k]
            # mvmtsp += lpSum(x_ijk[d][j][k] for j in V_n if j!=d) == required_visits[d-1]+1, f"Start_from_depot_{k}_{d}"
            # mvmtsp += lpSum(x_ijk[j][d][k] for j in V_n if j!=d) == required_visits[d-1]+1, f"Return_to_depot_{k}_{d}"
            mvmtsp += lpSum(x_ijk[d][j][k] for j in V_n if j!=d) >= 1, f"Start_from_depot_{k}_{d}"
            mvmtsp += lpSum(x_ijk[j][d][k] for j in V_n if j!=d) >= 1, f"Return_to_depot_{k}_{d}"
    
    #Ο περιορισμός αυτός φαίνεται να επηρεάζει τον συνολικό αριθμό των περιοχών 
    #που βγαίνουν στο τέλος. Θέλουμε το j να είναι διάφορο του d ώστε να αποτρέψουμε 
    #τις περιττές επισκέψεις από αρχική περιοχή σε αρχική περιοχή. Επίσης προσαρμόζουμε το 
    #required_visits, διότι ξεκινάει από το μηδέν. Το λογικό είναι ο αριθμός των επισκέψεων από και προς μία 
    #αρχική περιοχή να ισούται με των αριθμό των επαναφορτήσεων + 1 επίσκεψη γιατί ξεκινάμε 
    #και τελειώνουμε στην αρχική περιοχή. 

    #Constraint N1.5 == Number of visits to depots
    for k in K : 
        d = agent_assignment[k]
        mvmtsp += lpSum(y_ik[d][k]) <= charge_points, f"Number_of_visits_to_depot_{d}_{k}"
    # #Με αυτήν την παραδοχή ελέγχουμε ότι ο συνολικός αριθμός των επισκέψεων των πωλητών 
    #στις αρχικές περιοχές και για επαναφόρτιση θα είναι μικρότερος από το άθροισμα 
    #των ελάχιστων φορών που χρειάζεται να φορτίσει ένας πωλητής επί τον αριθμό των πωλητών συν των αριθμό των επισκέψεων 
    #που θα πρέπει να ολοκληρωθεί ούτως ή άλλως λόγω του 1 περιορισμού. 


    #Constraint N2 == Visit each city once. #No self loops allowed
    for k in K:
        for j in V_n:
            if j not in D: 
                # mvmtsp += (lpSum(x_ijk[i][j][k] for i in V_n if i!=j and i not in D) >= 1, f"Sum_Arrival_{j}_{k}")
                # mvmtsp += (lpSum(x_ijk[j][i][k] for i in V_n if i!=j and i not in D) >= 1, f"Sum_Departure_{k}_{j}")
                mvmtsp += (lpSum(x_ijk[i][j][k] for i in V_n if i!=j and i not in D) >= 1, f"Sum_Arrival_{j}_{k}")
                mvmtsp += (lpSum(x_ijk[j][i][k] for i in V_n if i!=j and i not in D) >= 1, f"Sum_Departure_{k}_{j}")
                mvmtsp += (lpSum(x_ijk[j][j][k]) == 0, f"Sum_Self_Loops_{k}_{j}")
    # Με τον περιορισμό αυτό, θέλουμε να επιτρέψουμε στους πωλητές μας να περάσουν από όλες τις πόλεις μία φορά
    #και να μην κάνουν άσκοπους κύκλους. Φυσικά αυτό δεν θα ισχύει για τις αρχικές περιοχές.   

    # for k in K: 
    #     for t in TS[:-1]: 
    #         for j in V_n: 
    #             mvmtsp += lpSum(x[j][i][k][t] + x[i][j][k][t+1] for i in V_n) <= 0 ,f"No_return_to_the_same_point{j}_{t}_{k}"

    #Constraint N3 == Energy approved path.  
    # for k in K: 
    #    mvmtsp += lpSum(e_ij[i-1][j-1] * x_ijk[i][j][k] for i in V_n for j in V_n if i!=j) < theta, f"Energy_UpperBound_{k}" #Constraint 3 For Energy
    #    mvmtsp += lpSum(e_ij[i-1][j-1] * x_ijk[i][j][k] for i in V_n for j in V_n if i!=j) > 0, f"Energy_lowerBound_{k}" #Constraint 3 For Energy

    # Contraint N4 == Balancing enter and exits from a city for agent k.  
    # for k in K:
    #     for j in V_n:
    #         for i in V_n : 
    #             if i != j and i not in D and j not in D:
    #                 mvmtsp += lpSum(x_ijk[j][i][k]) == lpSum(x_ijk[i][j][k]), f"Flow_conservation_{j}_{i}_by_{k}"
    
    # for k in K:
    #     for j in V_n: 
    #         if j!=agent_assignment[k]: 
    #             mvmtsp += lpSum(x_ijk[i][j][k] for i in V_n if i != j) == lpSum(x_ijk[j][i][k] for i in V_n if i!=j), f"Flow_conservation_{j}_by_{k}"
    #Με αυτόν τον περιορισμό σιγουρεύουμε ότι οι επισκέψεις από και προς μία πόλη είναι ίδιες.

    #Constraint N5 == Multi visits constraint for the city. 
    # for j in V_n : 
    #     if j == agent_assignment[k] or  j not in D :
    #         mvmtsp += lpSum(x_ijk[i][j][k] for i in V_n if i!=j and i not in D for k in K) <= required_visits[j-1], f'Min_visits_{j}_{k}' #Constraint 2 For Visits
    #         mvmtsp += lpSum(y_ik[j][k] for k in K) <= required_visits[j-1], f"Min_visits_y_{j}"
    #Με τον περιορισμό αυτό θέλουμε να ορίσουμε πως όλες οι πόλεις θα έχουν έως και  
    #όσες επισκέψεις όσες ορίζουμε εμείς, συνολικά από όλους τους πωλητές. 

    #Constraint N6. 
    # for k in K : 
    #     for d in D: 
    #         if d != agent_assignment[k]: 
    #             mvmtsp += lpSum(x_ijk[d][j][k] + x_ijk[j][d][k] for j in V_n) == 0, f"No_other_depot_visits_{k}_{d}"
    
    #Με αυτόν τον περιορισμό, ελέγχουμε ότι όλες οι αρχικές περιοχές θα έχουν πολλαπλούς πωλητές, 
    #μοιράζοντας τον φόρτο ανάλογα με το πλήθος των περιοχών και των πωλητών. 

    # Constraint N7 == To ensure that only a single agent is above a city each single time step. 
    # for t in TS:
    #     mvmtsp += lpSum(x[i][j][k][t] for k in K for i in V_n if i not in D for j in V_n if i!=j) <= 1,f"Single_Agent_at_at_time_{t}"
    # Με αυτόν τον περιορισμό εξασφαλίζουμε ότι σε κάθε πόλη κάθε χρονική στιγμή θα υπάρχει εώς ένας πωλητής 

    # for t in TS: 
    #     mvmtsp += lpSum(time_cons[i][k][t] for k in K for i in V_n if i not in D ) <= 1, f"Single{t}_Agent_at_at_time"
    # #Θεωρώ πως η δεύτερη λύση είναι καλύτερη μιάς και χρησιμοποιεί λιγότερο αποθηκευτικό χώρο 


    #Constraint N7.5 == Prevent loops and unecessarry visits at depots. 
    for k in K: 
        for t in TS[:-1]: 
                d = agent_assignment[k]
                mvmtsp += x[d][d][k][t] == 0 
                mvmtsp += x[d][d][k][t] + x[d][d][k][t+1] <= 1, f"No_loop_at_depot_{d}_{t}_{k}"
                mvmtsp += lpSum(x[d][j][k][t] for j in D ) == 0, f"No_unnecessary_visits_at_depot_{d}_{t}_{k}"
   
    # #Constraint N8. Prevent unnecessary viits at depots. 
    # for k in K : 
    #     for d in D: 
    #         for j in D: 
    #             mvmtsp += x_ijk[d][j][k] == 0 
 
   
    #Constraint N9 == Ensure that a customer/city is served by an agent. 
    # for i in C : 
    #     mvmtsp += lpSum(x_ijk[j][i[0]][k] for j in V_n for k in K) >= z_ik[i[0]][k], f"ServiceConstraint_{i}_{k}"
 
    #Constraint N10 == Ensure the path length (number of nodes) is balanced out.     
    # for k in K:
    #     mvmtsp += lpSum(x_ijk[i][j][k] for i in V_n for j in V_n if i!=j) == path_length[k], f"Path_Length_{k}"
    
    # for k1 in K:
    #         for k2 in K:
    #             if k1 != k2:
    #                 mvmtsp += path_length[k1] == path_length[k2],f"Equal_path_{k1}_{k2}"

    #Constraint N11 == Ensure that each edge is visited the exact number of visits.  
    # for (i,j) in edges:
    #     if i not in D and j not in D:
    #         mvmtsp += (
    #             (lpSum(x_ijk[i][j][k] for k in K) + lpSum(x_ijk[j][i][k] for k in K )) <= required_visits[i-1], 
    #             f"Each_edge_visited_{i}_{j}"
    #         )

    #Constraint N13 == Ensure that the drones start their journey fully charged. 
    # for k in K: 
    #     for d in D: 
    #         mvmtsp += e[d][k] == theta 
    
    # for i in D:
    #     mvmtsp += e[i] >= theta - M * (lpSum(z_ik[i][k] for k in K)), f"EnergyFull_at_start_{i}"

    #Constraint N14 == Ensure that travel is corresponding to energy. 
    # for k in K: 
    #     for i in V_n: 
    #         for j in V_n: 
    #             if i!=j: 
    #                 mvmtsp += e[j] >= e[i] - (e_ij[i-1][j-1] * x_ijk[i][j][k]),f"Update_remaining_energy_{i}_{j}_{k}"
    #                 # mvmtsp += e[j] == e[i] - (e_ij[i-1][j-1] * x_ijk[i][j][k]) + (1 - x_ijk[i][j][k] * theta)
    #             if j in D: 
    #                 mvmtsp += e[j] >= theta - M * (lpSum(z_ik[i][k])), f"EnergyFull_at_depot_{i}{j}_{k}"
        
    # for k in K: 
    #     for i in V_n: 
    #         for j in V_n: 
    #             if i!=j: 
    # #                 #Constraint N14.1 == Ensuring travel only if enough energy 
                    # mvmtsp += e[j][k] >= e[i][k] - (e_ij[i-1][j-1] * x_ijk[i][j][k])
    # #                 #Constraint N14.2 == Updating remaining energy 
                    # mvmtsp += e[j][k] >= e[i][k] - (e_ij[i-1][j-1] * x_ijk[i][j][k]) + (1 - x_ijk[i][j][k] * theta)
    # #                 #Constraint 14.3 == Adding constraints to ensure no travel if not enough energy
                    # mvmtsp += e[i][k] >= e_ij[i-1][j-1] * x_ijk[i][j][k]
                    # if j in D: 
                        # mvmtsp += e[j][k] == theta * x_ijk[i][j][k]
                    
    # #Constraint N15 == equires each edge visited by the drone to have at least one customer serviced in this drone trip
    # for k in K: 
    #     for (i,j) in edges:
    #         if i != j:
    #             mvmtsp += z_ik[i][k] + z_ik[j][k] >= 1 - M * (1-x_ijk[i][j][k]),f"one_customer_serviced_{i}_{j}_{k}"

    #Constraint N16 == Ensure that unecessary loops and stays in depots are prevented
    # for k in K: 
    #     for i in V_n: 
    #         mvmtsp += lpSum(x_ijk[i][j][k] + x_ijk[j][i][k] for j in V_n if i==j) == 0, f"No_unnecessary_loops_{i}_{k}"

    #Constraint that destroys the program.
    # for k in K: 
    #     for i in V_n: 
    #         mvmtsp += lpSum((x_ijk[i][j][k] for j in V_n))  + lpSum((x_ijk[j][i][k] for j in V_n)) == 2*y_ik[i][k], f"City Visits_{i}_{k}" #Constraint 2 For Visits

     #The above will make an infeasible solution if <= or == 
    
    # for t in TS:  # For each time step
    #     for j in V_n:  # For each area
    #         mvmtsp += lpSum(x[i][j][k][t] for k in K for i in V_n if i != j) <= 1, f"one_agent_in_{j}_at_time_{t}"
    
    #Constraint 12. 
    w_dist = 1
    w_energy = 0.4
    depots_visit_penalty = 5
    # for k in K: 
    #     mvmtsp.setObjective(w_dist*lpSum(lpDot(c_ij[i-1][j-1],x_ijk[i][j][k]) for i in V_n for j in V_n if i!=j)
    #                         + w_energy * (lpSum(lpDot(e_ij[i-1][j-1],x_ijk[i][j][k]) for i in V_n for j in V_n if i!=j)))
                            
    
    mvmtsp.setObjective(lpSum(lpDot(w_dist*c_ij[i-1][j-1],x_ijk[i][j][k]) for i in V_n for j in V_n if i!=j for k in K))
                        # + (w_energy *lpSum(lpDot(e_ij[i-1][j-1],x_ijk[i][j][k]) for i in V_n for j in V_n if i!=j for k in K)))
                        # + lpSum(depots_visit_penalty * x_ijk[d][j][k] for d in D for j in D for k in K))
    
    mvmtsp.writeLP("MvMTSP.lp")


    print(len(mvmtsp.constraints))
    #Solve the optimization problem 
    mvmtsp.solve()
    #Print the status of the solution 
    print('Status:', LpStatus[mvmtsp.status])
    
    if LpStatus[mvmtsp.status] == "Optimal": 
        paths = {k:[] for k in K}

        for k in K : 
            current_node = agent_assignment[k]
            route = [current_node]
            visited = set(route)
            
            while True: 
                next_node = None
                # Find the next node to visit from the current node
                for j in V_n:
                    if x_ijk[current_node][j][k].value() == 1 :#and j not in visited:
                        next_node = j
                        break
                if next_node is None or (next_node in visited and next_node != agent_assignment[k]):
                    break
                route.append(next_node)
                visited.add(next_node)
                current_node = next_node
                # pdb.set_trace()
            paths[k] = route

        # Print paths for each agent
        for k in paths:
            print(f"Agent {k} path: {' -> '.join(map(str, paths[k]))} with length {len(paths[k])}")    
    else:
        print("No solution found")
        paths = None
    # sys.stdout = old_stdout
    
   
    return depots


# df = pd.read_csv("M.csv")
# c_ij = df.to_numpy()
# n = 36 
# m = 5 
# numDepots = 2 
# D = range(1, numDepots +1)
# V = range(1, n + 1)
# A = range(1, m + 1)
# r = [1] * (n) 
# # pdb.set_trace()

# prob = LpProblem("MVM_TSP", LpMinimize)
# x = LpVariable.dicts("x", (V,V,A), 0, 1, LpBinary)
# y = LpVariable.dicts("y", (V,A), 0,1, LpBinary)

# for k in A : 
#     for d in D: 
#         prob += lpSum(x[d][j][k] for j in V if j != d) == 1, f"Depot_outD_{d}_{k}" #Constraint 1 For Depots 

# for k in A: 
#     for i in V: 
#         prob += lpSum((x[i][j][k] for j in V))  + lpSum((x[j][i][k] for j in V)) == 2*y[i][k], f"City Visits_{i}_{k}" #Constraint 2 For Visits

# for i in V: 
#     prob += lpSum(y[i][k] for k in A) >= r[i-1], f"Min_visits_{i}"

# for k in A: 
#     prob += lpSum(y[d][k] for d in D) == 1, f"Single_depot_per_agent_{k}"

# # prob += lpSum(c_ij[i][j] * x[i][j][a] for i in V for j in V for a in A)
# # for i in V: 
# #     prob += lpSum(x[i][j][a] for j in V for a in A) == 1
# #     prob += lpSum(y[i][k] for k in A >= r[i - 1]), f"Min_visits_{i}"

# for k in A:
#     for d in D:
#         prob += lpSum(x[d][j][k] for j in V if j != d) == 1, f"Start_from_depot_{d}_{k}"
#         prob += lpSum(x[j][d][k] for j in V if j != d) == 1, f"Return_to_depot_{d}_{k}"

# #Objective functions 
# prob += lpSum(c_ij[i-1][j-1] * x[i][j][k] for i in V for j in V for k in A)

# prob.solve()

# for v in prob.variables():
#     if v.varValue > 0: 
#         print(v.name, "=", v.varValue)

# print('Status:', LpStatus[prob.status])



mvmTSP_problem_formulation(5,[8,9])