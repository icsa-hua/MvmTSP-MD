import pulp as pl
import numpy as np
import pandas as pd
# Nodes including the depot (depot = node 0)

nodes = list(range(10))  # Nodes 0 to 4, where node 0 is the depot

# Distance matrix (symmetric for simplicity)
# distances = [
#     [0, 10, 15, 20, 10],
#     [10, 0, 35, 25, 15],
#     [15, 35, 0, 30, 20],
#     [20, 25, 30, 0, 18],
#     [10, 15, 20, 18, 0]
# ]

distances = pd.read_csv('../distance_cost.csv')
import pdb 
pdb.set_trace()
distances = distances.to_numpy().astype(int)


# Agents or salesmen
# agents = [1, 2]
agents = [1,2,3]

# Create the problem
model = pl.LpProblem("Multi_TSP", pl.LpMinimize)

# Decision variables: x[i][j][k] == 1 if agent k travels from node i to node j
x = pl.LpVariable.dicts("x", ((i, j, k) for i in nodes for j in nodes if i != j for k in agents), cat='Binary')
u = pl.LpVariable.dicts("u", ((i, k) for i in nodes for k in agents ), lowBound=0, upBound=len(nodes)-1, cat='Integer')
# Objective function: Minimize the total distance traveled by all agents
model += pl.lpSum(x[i, j, k] * distances[i][j] for i in nodes for j in nodes if i != j for k in agents)

# Each node must be entered and left exactly once by each agent
for k in agents:
    for j in nodes:
        model += pl.lpSum(x[i, j, k] for i in nodes if i != j) == 1, f"enter_{j}_by_{k}"
        model += pl.lpSum(x[j, i, k] for i in nodes if i != j) == 1, f"exit_{j}_by_{k}"

# Subtour elimination constraints (SEC) are simplified for small problems
for k in agents:
    for i in nodes[1:]:
        model += x[i, 0, k] + x[0, i, k] <= 1  # Avoid direct loops back to the depot without visiting other nodes

for k in agents:
    for i in nodes:
        for j in nodes:
            if i != j and i != 0 and j != 0:  # Depot does not need these constraints
                model += u[i, k] - u[j, k] + len(nodes) * x[i, j, k] <= len(nodes) - 1

for k in agents:
    for i in nodes:
        if i != 0:  # Non-depot nodes
            model += u[i, k] >= 1
            model += u[i, k] <= len(nodes) - 1

# Solve the problem using the default solver
model.solve()

# Check if a solution was found
if pl.LpStatus[model.status] == "Optimal":
    # Retrieve the paths
    paths = {k: [] for k in agents}
    for k in agents:
        # Start from the depot
        current_node = 0
        start_node = current_node
        route = [current_node]
        while True:
            next_node = None
            for j in nodes:
                if j != current_node and x[current_node, j, k].varValue == 1:
                    next_node = j
                    break
            if next_node is None or next_node == start_node:
                break
            route.append(next_node)
            current_node = next_node
        paths[k] = route

    # Print the paths for each agent
    for k in paths:
        print(f"Agent {k} path: {' -> '.join(map(str, paths[k]))}")

    print("Optimal solution found.")
else:
    print("No optimal solution available.")
