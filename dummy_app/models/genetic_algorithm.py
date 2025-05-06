import numpy as np 
import random
import pandas as pd
import networkx as nx 
from dummy_app.tools.logger import logger
from dummy_app.tools.autonomize import create_model_graph, get_weights
from typing import Dict, List, Any, Tuple 
from deap import base, creator, tools, algorithms



class GASolution:


    def __init__(self, population:int=200, generations:int=100, nodes:pd.DataFrame=None, depot:int=0 )->None:
        self.population_size = population 
        self.generations = generations 
        self.nodes = nodes 
        self.graph = nx.Graph() 
        self.depot = depot 

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox() 


    def create_graph(self, cost:Dict[str,pd.DataFrame])->nx.Graph:
        """ 
        Creates a graph representation using the provided cost matrices.
        
        Args:
            weights (dict): Dictionary containing 'distance', 'energy', and 'travel_time' cost matrices.

        Returns:
            nx.Graph: Graph with nodes and weighted edges.
        """
        weights = get_weights()
        return create_model_graph(cost=cost, nodes=self.nodes, weights=weights)


    def initialize_tour(self):
        """
        Initializes a tour starting and ending at a depot, visiting all nodes.

        Returns:
            list: A complete tour (list of area IDs).
        """
        all_nodes = list(self.nodes.keys()) 
        tmp = {v:k for k,v in self.nodes.items()}
        depot_id = tmp[self.depot]
        logger.debug(f"Depot IDs: {depot_id}")
        # Remove depot(s) from list 
        if depot_id in all_nodes: 
            all_nodes.remove(depot_id)

        random.shuffle(all_nodes)

        tour = [depot_id] + all_nodes + [depot_id]

        return tour 


    def crossover(self, ind1:List[int], ind2:List[int])->Tuple[List[int], List[int]]:
        """
        Perform crossover between two tours based on common nodes.
        The crossover operation selects a random crossover point that is a common node between 
        the two individuals, and then swaps the nodes between the two individuals around that
        crossover point, while maintaining the validity of the tours.
        
        Args:
            ind1 (list): First tour.
            ind2 (list): Second tour.

        Returns:
            tuple: Two new tours after crossover.
        """
        if len(ind1) < 3 or len(ind2) < 3:
            logger.error("Individuals must have at least 3 nodes.")
            return ind1, ind2 #too small tours 
        
        common_nodes = list(set(ind1) & set(ind2))
        common_nodes = [node for node in common_nodes if self.graph.has_node(node)]

        if len(common_nodes) <=2 : # Only depots  
            return ind1, ind2 # Not enough for meaningful crossover 
        
        # Pick random common node not at index 0 or last 
        valid_common_nodes = [node for node in common_nodes if node != ind1[0] and node != ind1[-1]]
        if not valid_common_nodes: 
            return ind1, ind2 # No valid common nodes 
        
        crossover_point = random.choice(valid_common_nodes)

        idx1 = ind1.index(crossover_point)
        idx2 = ind2.index(crossover_point)

        new_ind1 = ind1[:idx1] + ind2[idx2:]
        new_ind2 = ind2[:idx2] + ind1[idx1:]

        return new_ind1, new_ind2
    

    def mutation(self, individual:List[int], indpd:float)-> List[int]:
        """
        Mutate a tour by swapping two nodes with given probability.
        The mutation operation randomly selects two nodes in the tour and swaps their positions,
        as long as the resulting tour is still valid (i.e., all edges between the swapped nodes exist in the graph).
        
        Args:
            individual (list): The tour.
            indpb (float): Mutation probability.

        Returns:
            list: Mutated tour.
        """
        if len(individual)<=3: 
            return individual, # Too small to mutate

        if np.random.rand() < indpd: 
            swap_indices = random.sample(range(1,len(individual)-1), 2)
            idx1, idx2 = swap_indices 

            if (self.graph.has_edge(individual[idx1-1], individual[idx2]) and
            self.graph.has_edge(individual[idx2], individual[idx1+1]) and
            self.graph.has_edge(individual[idx2-1], individual[idx1]) and
            self.graph.has_edge(individual[idx1], individual[idx2+1])):
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

        return individual,


    def fitness_evaluation(self, individual:List[int], cost:Dict[str,float]) ->Tuple[float,]: 
        """
        Evaluates fitness: total travel cost (distance + energy).

        Args:
            individual (list): Tour sequence.
            cost (dict): Dict of cost matrices.

        Returns:
            tuple: Total cost as single value.
        """
        weights = get_weights() 
        total_distance = 0.0 
        total_energy = 0.0 
        travel_time = 0.0 
        for i in range(len(individual) -1): 
            try: 
                node_from = self.nodes[individual[i]]
                node_to = self.nodes[individual[i+1]] -1

                total_distance += cost['distance'][node_from][node_to]
                total_energy += cost['energy'][node_from][node_to]
                travel_time += cost['travel_time'][node_from][node_to]

            except KeyError as ke: 
                logger.exception(f"KeyError during fitness evaluation at index {i}: {ke}")
            except IndexError as ie:
                logger.exception(f"IndexError during fitness evaluation at index {i}: {ie}")

        total_cost = (
            weights['distance'] * total_distance + 
            weights['energy'] * total_energy +
            weights['travel_time'] * travel_time
        )
        return (total_cost,)


    def toolbox_config(self)->None:
        """
        Setup DEAP toolbox for the Genetic Algorithm.
        """
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.initialize_tour) 
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutation, indpd=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        logger.debug("Toolbox configured.") 

    
    def run(self, crossover_rate:float, mutation_rate:float, cost:Dict[str, float], enable_indi_fitness:bool=True, verbose:bool=True)->None:
        """
        Run the GA optimization.

        Args:
            crossover_rate (float): Probability of crossover.
            mutation_rate (float): Probability of mutation.
            cost (dict): Cost matrices.
            enable_individual_fitness (bool): Re-evaluate individuals manually after GA.
            seed (int): Random seed (optional).

        Returns:
            tuple: Best paths and fitness.
        """

        logger.debug("Starting Genetic Algorithm...")
        self.graph = self.create_graph(cost) 
        self.toolbox_config() 
        self.toolbox.register("evaluate", self.fitness_evaluation, cost=cost) 

        # Initialize population
        population = self.toolbox.population(n=self.population_size)

        HOF = tools.ParetoFront() 
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0) 

        algorithms.eaSimple(
            population, self.toolbox, 
            cxpb=crossover_rate, mutpb=mutation_rate,
            ngen=self.generations, stats=stats,
            halloffame=HOF, verbose=verbose
        )

        best_individual = tools.selBest(population, 1)[0] 
        if enable_indi_fitness:
            fitness = self.fitness_evaluation(best_individual, cost)
        else: 
            fitness = best_individual.fitness.values    

        # convert internal node indexes back to Area IDs 
        best_path = [self.nodes[node] for node in best_individual]
        logger.debug(f"Best path: {best_path} with fitness: {fitness[0]}") 
        return best_path, fitness[0] 
    
