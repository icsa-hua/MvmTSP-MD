from genetic_algorithm import GA
from deap import base, creator, tools, algorithms 
import pandas as pd 
import random 
import numpy
import numpy as np
import networkx as nx
from typing import Tuple
import pdb

class GASOL(GA): 
    def __init__(self, population_size, generations, nodes, agents,D):
        self.population_size = population_size
        self.generations = generations
        self.nodes = nodes
        self.agents = agents
        self.m = len(agents)
        self.depots = D
        creator.create("FitnessMin", base.Fitness,weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)  
        self.toolbox = base.Toolbox()
        


    def create_graph(self,cost):
        graph = nx.Graph()
        for enter in self.nodes: 
            for exit in self.nodes: 
                graph.add_edge(enter, exit, cost=cost[enter][exit])
        return graph 
    
    def create_graph_dict(self,cost):
        graph = nx.Graph()
        for enter in self.nodes.values(): 
            for ex in self.nodes.values():
                try:
                    graph.add_edge(enter, ex, cost=cost[enter][ex-1])
                except:
                    pdb.set_trace()
        return graph 
    
    def initialize_tours(self):
        all_nodes = self.nodes[:]
        random.shuffle(all_nodes)
        # d = self.depots[np.random.randint(2)]
        
        size_per_agent = len(all_nodes) // self.m
        tours = [] 
        index = 0 
        m = len(self.agents)
        for _ in range(self.m): 
            d = self.depots[_ + 1]
            if _ == m - 1: 
                tour = all_nodes[index:]
            else:
                tour = all_nodes[index:index+size_per_agent]
            index += size_per_agent
            tours.extend([d] + tour + [d])

        return tours 

    def initialize_tours_dict(self):
        # all_nodes = [self.nodes[k] for k in self.nodes]
        all_nodes = list(range(len(self.nodes.values())))
        tmp_dict = {value:key for key,value in self.nodes.items()}
        random.shuffle(all_nodes)
        tour = [] 
        d = tmp_dict[self.depots]
        tour.extend([d] + all_nodes + [d])
        return tour 

    def initialize_population(self):
        pass

    def crossover(self, ind1, ind2): 
        size = min(len(ind1), len(ind2))
        p1, p2 = [0] * size, [0] * size

        if len(ind1) < 2 or len(ind2) < 2:
            return ind1 if len(ind1) > len(ind2) else ind2
        
        common_nodes = [node for node in ind1 if node in ind2 and self.graph.has_node(node)]
        connected_common_nodes = [node for i, node in enumerate(common_nodes[:-1]) if self.graph.has_edge(node, common_nodes[i + 1]) and self.graph.has_edge(common_nodes[i + 1], node)]
        if not connected_common_nodes:
            return ind1 if np.random.rand() > 0.7 else ind2

        crossover_point = np.random.choice(connected_common_nodes)
        cp_index_p1 = ind1.index(crossover_point)
        cp_index_p2 = ind2.index(crossover_point)
        
        # Initialize position of each indices in the individuals
        for i in range(size):
            p1[ind1[i]] = i
            p2[ind2[i]] = i

        # Apply crossover between cx points
        for i in range(cp_index_p1, cp_index_p2):
            # Keep positions of the depot fixed
            if i == 0 or i == size-1:
                continue
            temp1, temp2 = ind1[i], ind2[i]
            ind1[i], ind1[p1[temp2]] = temp2, temp1
            ind2[i], ind2[p2[temp1]] = temp1, temp2
            p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
            p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

        return ind1, ind2


    def mutation(self, individual, indpb): 
        if np.random.rand() < indpb and len(individual) > 2: 
            swap_index1 = np.random.randint(1, len(individual) - 1)
            swap_index2 = np.random.randint(1, len(individual) - 1)

            if (self.graph.has_edge(individual[swap_index1-1],individual[swap_index2]) and 
                self.graph.has_edge(individual[swap_index2], individual[swap_index1 + 1]) and
                self.graph.has_edge(individual[swap_index2-1], individual[swap_index1]) and 
                self.graph.has_edge(individual[swap_index1], individual[swap_index2 + 1])):

                individual[swap_index1], individual[swap_index2] = individual[swap_index2], individual[swap_index1]
        return individual,


    def fitness_evaluation(self,individual, cost: np.ndarray) -> Tuple:
        total_distance = 0 
        for i in range(0, len(individual), 2*(len(self.nodes)//self.m+1)): 
            tour = individual[i:i + 2*(len(self.nodes)//self.m+1)]
            # tour = individual
            print(len(tour))
            for j in range(len(tour)): 
                
                
                try: 
                    total_distance += cost[tour[j-1]][tour[j]]
                except IndexError:
                    print(f"Cost = {cost.shape}")
                    print(f"Tour = {tour}")
                    print(f"j = {j}")
                    print(f"Individual = {individual}")
        return (total_distance,)
    
    def fitness_evaluation_dict(self,individual, cost: np.ndarray) -> Tuple:
        total_distance = 0 
        for i in range(0, len(individual)): 
            
            total_distance += cost[self.nodes[individual[i-1]]][self.nodes[individual[i]]]
            
        return (total_distance,)
        

    def toolbox_config(self): 
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.initialize_tours)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate",self.crossover)    
        self.toolbox.register("mutate", self.mutation, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def toolbox_config_dict(self): 
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.initialize_tours_dict)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate",self.crossover)    
        self.toolbox.register("mutate",  self.mutation, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def run(self, cthr, mthr,cost):
        if not isinstance(self.nodes, dict): 
            self.graph = self.create_graph(cost)
            self.toolbox_config()
            self.toolbox.register("evaluate", self.fitness_evaluation, cost=cost)
            HOF   = tools.ParetoFront()
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", numpy.mean, axis=0)
            stats.register("std", numpy.std, axis=0)
            stats.register("min", numpy.min, axis=0)
            stats.register("max", numpy.max, axis=0)
            population = self.toolbox.population(n=self.population_size)
            algorithms.eaSimple(population, self.toolbox, cxpb=cthr, mutpb=mthr, ngen=self.generations, stats=stats, halloffame=HOF, verbose=False)
            best_paths = tools.selBest(population,self.population_size)
            paths = self.set_tours(paths=best_paths, agents=self.m)
            for i, path in enumerate(paths): 
                print(f"Path for agent {i+1}: {path}")
            return paths, HOF
        self.graph = self.create_graph_dict(cost)
        self.toolbox_config_dict()
        self.toolbox.register("evaluate", self.fitness_evaluation_dict, cost=cost)
        HOF = tools.ParetoFront()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)
        population = self.toolbox.population(n=self.population_size)
        algorithms.eaSimple(population, self.toolbox, cxpb=cthr, mutpb=mthr, ngen=self.generations, stats=stats, halloffame=HOF, verbose=False)
        paths =  tools.selBest(population,1)
        return paths, HOF

    def set_tours(self,paths,agents): 
        unique_paths = dict() 
        path_id = 1 
        values = [self.depots[key] for key in self.depots.keys()]
        for path in paths: 
            if path[0] not in values: 
                continue 

            path_tuple = tuple(path)

            if path_tuple not in unique_paths:
                unique_paths[path_tuple] = path_id
                path_id += 1

            if path_id > agents:
                break
        return unique_paths











