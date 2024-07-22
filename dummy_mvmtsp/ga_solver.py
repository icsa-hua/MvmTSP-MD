from interface.genetic_algorithm import GA
from deap import base, creator, tools, algorithms 
import random 
import numpy
import numpy as np
import networkx as nx
from typing import Tuple
import pdb
import warnings

class GASOL(GA): 
    def __init__(self, population_size, generations, nodes, agents,depots):
        """
        Initializes the GASOL class with the specified parameters.
        
        Args:
            population_size (int): The size of the population for the genetic algorithm.
            generations (int): The number of generations to run the genetic algorithm.
            nodes (list): A list of nodes in the problem.
            agents (list): A list of agents in the problem.
            D (list): A list of depot nodes.
        
        Initializes the fitness function and individual representation for the genetic algorithm.
        """
        super().__init__(population_size, generations, nodes, agents, depots)
        self.population_size = population_size
        self.generations = generations
        self.nodes = nodes
        self.agents = agents
        self.m = len(agents)
        self.depots = depots
        creator.create("FitnessMin", base.Fitness,weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)  
        self.toolbox = base.Toolbox()

    def create_graph(self,cost):
        """
        Creates a graph representation of the problem using the provided cost matrix.
        
        Args:
            cost (dict): A dictionary representing the cost matrix, where the keys are the location IDs and the values are lists of costs for each corresponding location.
        
        Returns:
            nx.Graph: A graph representation of the problem.
        """
                
        graph = nx.Graph()
        for enter in self.nodes: 
            for exit in self.nodes: 
                graph.add_edge(enter, exit, cost=cost[enter][exit])
        return graph 
    
    def create_graph_dict(self,cost):
        
        """ 
        Creates a graph representation of the problem using the provided cost matrix.
                
        Args:
            cost (dict): A dictionary representing the cost matrix, where the keys are the location IDs and the values are lists of costs for each corresponding location.
        
        Returns:
            nx.Graph: A graph representation of the problem.
        """
                
        graph = nx.Graph()
        for enter in self.nodes.keys(): 
            for ex in self.nodes.keys():
                try:
                    graph.add_edge(enter, ex, cost=cost[self.nodes[enter]][self.nodes[ex]-1])
                except:
                    warnings.warn(f"No edge between {enter} and {ex} with corresponding cost")
        return graph 
    
    def initialize_tours(self):

        """
        Initializes the tours for each agent in the genetic algorithm.

        This method creates a list of tours,
        where each tour represents the path that an agent will take.
        The tours are initialized by first shuffling the list of all nodes,
        then assigning a subset of the nodes to each agent's tour,
        ensuring that each agent's tour starts and ends at the depot.

        Returns:
            list: A list of tours, where each tour is a list of nodes representing the path that an agent will take.
        """

        all_nodes = list(self.nodes.keys())
        tmp_dict = {v:k for k,v in self.nodes.items()}
        random.shuffle(all_nodes)
        
        size_per_agent = len(all_nodes) // self.m
        tours = [] 
        visited = []
        index = 0 
        
        for k in range(1, self.m + 1): 
            tour = []
            tmp_nodes = all_nodes.copy()
            d = tmp_dict[self.depots[k]]

            if d in visited:
                continue

            visited.append(d)
            tmp_nodes.remove(d)
            tour.extend([d] + tmp_nodes + [d])
            index += size_per_agent
            tours.extend(tour)

        return tours 

    def initialize_tours_dict(self):

        """
        Initializes a tour for each agent in the genetic algorithm.
        
        Returns:
            list: A list representing the tour for each agent.
        """
                
        all_nodes = list(self.nodes.keys())   
        tmp_dict = {v:k for k,v in self.nodes.items()}
        d = [tmp_dict[self.depots]]
        all_nodes.remove(*d)
        random.shuffle(all_nodes)
        # tour = [] 
        # tour.extend(d + all_nodes + d)
        tour = d + all_nodes + d
        return tour 

    def initialize_population(self):
        pass

    def crossover(self, ind1, ind2): 

        """
        Performs a crossover operation on two individuals (tours) in the genetic algorithm.
        
        The crossover operation selects a random crossover point that is a common node between 
        the two individuals, and then swaps the nodes between the two individuals around that
        crossover point, while maintaining the validity of the tours.
        
        Args:
            ind1 (list): The first individual (tour) to be crossed over.
            ind2 (list): The second individual (tour) to be crossed over.
        
        Returns:
            tuple: A tuple containing the two new individuals (tours) after the crossover operation.
        """
        
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
        """
        Performs a mutation operation on an individual (tour) in the genetic algorithm.
        
        The mutation operation randomly selects two nodes in the tour and swaps their positions,
        as long as the resulting tour is still valid (i.e., all edges between the swapped nodes exist in the graph).
        
        Args:
            individual (list): The individual (tour) to be mutated.
            indpb (float): The probability of mutation occurring.
        
        Returns:
            list: The mutated individual (tour).
        """
                
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
        
        """
        Evaluates the fitness of an individual (tour) in the genetic algorithm.
                
        Args:
            individual (list): The individual (tour) to be evaluated.
            cost (np.ndarray): A 2D numpy array representing the cost (distance) between each pair of nodes.
        
        Returns:
            Tuple[float]: A tuple containing the total distance of the tour.
        """
                
        total_distance = 0 
        for i in range(0, len(individual), 2*(len(self.nodes)//self.m+1)): 
            tour = individual[i:i + 2*(len(self.nodes)//self.m+1)]
            for j in range(len(tour)-1): 
                try: 
                    total_distance += cost[tour[j]][tour[j+1]]
                except IndexError:
                    warnings.warn(f"IndexError: {j}")

        return (total_distance,)
    
    def fitness_evaluation_dict(self,individual, cost: dict) -> Tuple:
        """
        Evaluates the fitness of an individual (tour) in the genetic algorithm.
        
        Args:
            individual (list): The individual (tour) to be evaluated.
            cost (np.ndarray): A dictionary representing the cost (distance) between each pair of nodes.
        
        Returns:
            Tuple[float]: A tuple containing the total distance of the tour.
        """
                
        total_distance = 0 
        for i in range(0, len(individual)-1): 
            try: 
                total_distance += cost[self.nodes[individual[i]]][self.nodes[individual[i+1]]-1]
            except: 
                warnings.warn(f"IndexError: {i}")
        
        return (total_distance,)
        
    def toolbox_config(self):

        """
        Configures the toolbox for the genetic algorithm.
        
        This method sets up the necessary components of the DEAP toolbox for the genetic algorithm. It registers the following:
        
        - `individual`: A function that initializes an individual (tour) using the `self.initialize_tours` method.
        - `population`: A function that initializes a population of individuals using the `self.toolbox.individual` function.
        - `mate`: A function that performs crossover on two individuals using the `self.crossover` method.
        - `mutate`: A function that mutates an individual using the `self.mutation` method with a probability of 0.1.
        - `select`: A function that selects individuals for the next generation using tournament selection with a tournament size of 3.
        """
                
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.initialize_tours)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate",self.crossover)    
        self.toolbox.register("mutate", self.mutation, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def toolbox_config_dict(self): 
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.initialize_tours_dict)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate",self.crossover)    
        self.toolbox.register("mutate",  self.mutation, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def run(self, cthr, mthr,cost, enable_individual_fitness=False):
        
        """
        Runs the genetic algorithm to solve the optimization problem.
        
        Args:
            cthr (float): The crossover threshold for the genetic algorithm.
            mthr (float): The mutation threshold for the genetic algorithm.
            cost (dict): A dictionary representing the cost (distance) between each pair of nodes.
            enable_individual_fitness (bool, optional): Whether to enable individual fitness evaluation. Defaults to False.
        
        Returns:
            optimal path solution (list of paths, or a single path), tools.ParetoFront]: A tuple containing the best paths found by the genetic algorithm and the Pareto front.
        """
                
        HOF   = tools.ParetoFront()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)
        
        if not enable_individual_fitness: 
            self.graph = self.create_graph(cost)
            self.toolbox_config()
            self.toolbox.register("evaluate", self.fitness_evaluation, cost=cost)
            
            population = self.toolbox.population(n=self.population_size)
            algorithms.eaSimple(population, self.toolbox, cxpb=cthr, mutpb=mthr, ngen=self.generations, stats=stats, halloffame=HOF, verbose=False)
            best_paths = tools.selBest(population,self.population_size)
            paths = self.set_tours(paths=best_paths, agents=self.m)

            return paths, HOF
        
        else:
            self.graph = self.create_graph_dict(cost)
            self.toolbox_config_dict()
            self.toolbox.register("evaluate", self.fitness_evaluation_dict, cost=cost)
            
            population = self.toolbox.population(n=self.population_size)
            algorithms.eaSimple(population, self.toolbox, cxpb=cthr, mutpb=mthr, ngen=self.generations, stats=stats, halloffame=HOF, verbose=False)
            paths =  tools.selBest(population,1)

            return paths, HOF

    def set_tours(self,paths,agents): 
        """
        Converts a list of paths into a dictionary of unique paths, where each path is assigned a unique identifier.
        
        Args:
            paths (list): A list of paths, where each path is a list of node IDs.
            agents (int): The number of agents (vehicles) to consider.
        
        Returns:
            dict: A dictionary where the keys are the unique paths and the values are the unique path IDs.
        """
                
        unique_paths = dict() 
        path_id = 1 
        tmp_dict = {v:k for k,v in self.nodes.items()}
        values = {tmp_dict[self.depots[k]] for k in self.agents}

        for path in paths: 
            if path[0] in values: 
                path = tuple(path[0:len(self.nodes.keys())+1])
                if path not in unique_paths:
                    unique_paths[path] = path_id
                    path_id += 1
                    if path_id > agents:
                        break
                    
        return unique_paths











