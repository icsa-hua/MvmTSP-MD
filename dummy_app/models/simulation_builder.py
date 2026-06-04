from __future__ import annotations
from dataclasses import asdict

from dummy_app.core.exceptions import ValidationOptimalityConfirmed
from dummy_app.designs.mvmtsp_config import MVMTSPConfig 
import dummy_app.tools.common as common
from dummy_app.core.schemas.request import ModelRunRequest
from dummy_app.core.schemas.result import ClusterSolveResult, ModelRunResult
from dummy_app.core.statuses import compute_absolute_gap, compute_relative_gap, infer_termination_reason, normalize_solver_status
from dummy_app.pipeline.artifacts import persist_playback_artifacts, persist_run_artifacts
from dummy_app.pipeline.instance_builder import build_problem_instance
from dummy_app.tools.performance_metrics import Metrics
from dummy_app.designs.cluster import Cluster
from dummy_app.models.heuristics.alns_solver import ALNSOptimizationModel, solve_alns_baseline
from dummy_app.models.heuristics.genetic_algorithm_solver import (
    GeneticAlgorithmOptimizationModel,
    solve_genetic_algorithm_baseline,
)
from dummy_app.models.heuristics.global_greedy_nn import GlobalGreedyNNOptimizationModel
from dummy_app.models.heuristics.static_partition_greedy_nn import StaticPartitionGreedyNNOptimizationModel
from dummy_app.models.milp.model import MILPOptimizationModel
from dummy_app.models.milp.solver_adapter import solve_cluster_problem
from dummy_app.models.milp.subtour.strategies import normalize_subtour_mode
from dummy_app.models.RL.controller import RLController
from dummy_app.tools.logger import logger 
from dummy_app.program_config import CENTROIDS_PATH

import os 
import gc
import pdb
import math
import copy 
import time
import random
import pulp as pl 
import numpy as np 
import pandas as pd
import timeout_decorator 
from pyproj import Transformer

from tqdm import tqdm 
from collections import defaultdict
from typing import Any, Callable, List, Dict, Tuple, Mapping

from dummy_app.models.milp.warm_start import build_warm_start_payload, normalize_warm_start_mode

# Functions to transform coordinates from EPSG to UTM 
transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
transformer_to_latlon = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)

class Builder(MVMTSPConfig):
    """
    This class is the builder for the problem formulation. It utilizes PuLP 
    to create the problem and interfaces the distinct components such as 
    clustering, VNI, central hubs and genetic algorithm. 
    """

    def __init__(self, config:Dict[str,Any], trials:int): 
        
        super().__init__(
            env_type=config["env_type"], 
            max_battery=config["max_battery"],
            max_coverage_time=config["max_coverage_time"],
            enable_ga=config["enable_ga"],
            scenario=config["scenario"],
            objective_function=config["objective_function"],
            stage_solution=config["stage_solution"],
            priority=config["priority"],
            validate=config["validate"]
        )
        
        self.NUMBER_OF_AGENTS = config["NUMBER_OF_AGENTS"]
        self.NUMBER_OF_USERS = config["NUMBER_OF_USERS"]
        self.NUMBER_OF_AREAS = config["NUMBER_OF_AREAS"]
        self.agent_altitude = config["altitude"]
        self.Time = 0 
        self.model_name = str(config.get("model_name", "milp"))
        self.solver_backend = str(config.get("solver_backend", "glpk"))
        self.subtour_mode = normalize_subtour_mode(config.get("subtour_mode", config.get("subtour_strategy", "mtz")))
        self.subtour_strategy = self.subtour_mode
        self.objective_strategy = str(config.get("objective_strategy", "legacy_stage"))
        self.scenario_constraint_set = str(config.get("scenario_constraint_set", "default"))

        self.metrics = Metrics(base_dir=f"{os.getcwd()}/assets/results/metrics", verbose=True) 
        self.recharge_time_window:int = 5 #descrete time steps
        self.num_constraints = 0 
        self.variables_count = 0
        self.num_binary_variables = 0
        self.num_continuous_variables = 0
        self.problem_results = defaultdict()
        self.global_nodes_visited:int = 0
        self.visits_per_nodes:Dict[int,int] = {}
        self.total_number_cluster: int = 0 
        self.coordinated_plan = defaultdict(dict)
        self.plan_with_nodes =  defaultdict(dict)
        self.total_data_rate = 0.0 
        self.makespan = 0.0 
        self.base_enable_ga = bool(config["genetic_algorithm"])
        self.enable_ga = self.base_enable_ga
        self.ga_generations = int(config.get("ga_generations", 100))
        self.solver_time_limit_seconds = config.get("solver_time_limit_seconds")
        self.objective_weights = config.get("objective_weights", self.objective_weights)
        self.clustering_feature_weights = config.get("clustering_feature_weights", self.clustering_feature_weights)
        self.random_seed = int(config.get("random_seed", config.get("seed", getattr(self, "random_seed", 42))))
        self.solver_seed = int(config.get("solver_seed", self.random_seed))
        self.fairness_tolerance = int(config.get("fairness_tolerance", getattr(self, "fairness_tolerance", 2)))
        self.time_step_sec = int(config.get("time_step_sec", getattr(self, "time_step_sec", 600)))
        self.warm_start_mode = normalize_warm_start_mode(
            config.get("warm_start_mode", "ga" if self.enable_ga else "none")
        )
        bridge_visit_override = config.get("bridge_node_required_visits_override")
        self.bridge_node_required_visits_override = (
            None if bridge_visit_override in {None, "", 0} else int(bridge_visit_override)
        )
        self.cluster_status_records: List[Dict[str, Any]] = []
        self.solve_status_history: List[Dict[str, Any]] = []
        self.latest_run_summary: Dict[str, Any] = {}
        self.latest_learning_result: Dict[str, Any] = {}
        self.latest_run_report: Dict[str, Any] = {}
        self.latest_run_report_path: str = ""
        self.latest_artifact_dir: str = ""
        self.latest_playback_rows: List[Dict[str, Any]] = []
        self.latest_playback_metadata: Dict[str, Any] = {}
        self.current_problem_instance = None
        self.run_request: ModelRunRequest | None = None
        self.latest_model_run_result: ModelRunResult | None = None
        self.agent_next_available_time: Dict[int, float] = {}
        self.optimization_model = self._create_optimization_model(self.model_name)
        self.learning_enabled = bool(config.get("learning_enabled", False))
        self.learning_controller = None
        if self.learning_enabled:
            self.learning_controller = RLController(
                base_dir=config.get("learning_output_dir", f"{os.getcwd()}/assets/results/rl"),
                alpha=float(config.get("learning_alpha", 0.75)),
            )
        self._seed_random_generators()


    def _seed_random_generators(self) -> None:
        random.seed(int(self.random_seed))
        np.random.seed(int(self.random_seed))


    def _create_optimization_model(self, model_name: str):
        normalized_model_name = str(model_name or "milp").strip().lower()
        if normalized_model_name in {"milp"}:
            return MILPOptimizationModel(self)
        if normalized_model_name in {"ga", "genetic_algorithm"}:
            return GeneticAlgorithmOptimizationModel(self)
        if normalized_model_name in {"alns"}:
            return ALNSOptimizationModel(self)
        if normalized_model_name in {"greedy_nn", "gnn_ntw"}:
            return GlobalGreedyNNOptimizationModel(self)
        if normalized_model_name in {"greedy_partition_nn", "static_partition_greedy_nn"}:
            return StaticPartitionGreedyNNOptimizationModel(self)
        raise ValueError(f"Unsupported model_name '{model_name}'")


    def call_genetic_algorithm(
        self,
        nodes_dict:Dict[int,int],
        cost:Dict[str,float],
        depot:int,
        verbose:bool=False,
        population_size:int=200,
        generations:int=100,
        seed: int | None = None,
    )->Tuple[List[int],Any]:
        if generations is None:
            generations = self.ga_generations
        return super().call_genetic_algorithm(
            nodes_dict,
            cost,
            depot,
            verbose,
            population_size,
            generations,
            seed=seed,
        ) 


    def build_cluster_initializer(self, cluster: Cluster) -> None:
        cluster.initial_population = {}
        cluster.initializer_timeframe_estimate = None
        cluster.warm_start_solution = {}
        cluster.warm_start_summary = {
            "strategy": self.warm_start_mode,
            "available": False,
            "provided_to_solver": False,
            "accepted_by_solver": None,
            "acceptance_source": "",
            "generation_time_sec": 0.0,
            "objective_value": None,
            "solver_objective_value": None,
            "makespan": None,
            "timeframe_estimate": None,
        }

        if self.warm_start_mode == "none":
            return

        if self.warm_start_mode == "ga":
            generation_started_at = time.perf_counter()
            heuristic_solution = solve_genetic_algorithm_baseline(cluster, self)
            generation_time_sec = time.perf_counter() - generation_started_at
            summary, payload = build_warm_start_payload(
                builder=self,
                cluster=cluster,
                heuristic_solution=heuristic_solution,
                strategy_name="ga",
                generation_time_sec=generation_time_sec,
            )
            cluster.warm_start_solution = payload
            cluster.warm_start_summary = summary
            cluster.initializer_timeframe_estimate = summary.get("timeframe_estimate")
            return

        if self.warm_start_mode == "alns":
            generation_started_at = time.perf_counter()
            heuristic_solution = solve_alns_baseline(cluster, self)
            generation_time_sec = time.perf_counter() - generation_started_at
            summary, payload = build_warm_start_payload(
                builder=self,
                cluster=cluster,
                heuristic_solution=heuristic_solution,
                strategy_name="alns",
                generation_time_sec=generation_time_sec,
            )
            cluster.warm_start_solution = payload
            cluster.warm_start_summary = summary
            cluster.initializer_timeframe_estimate = summary.get("timeframe_estimate")
            return

        raise ValueError(f"Unsupported warm_start_mode '{self.warm_start_mode}'")
    

    def assign_agents_to_areas(self, plethos, depots:Any)->Dict[int,int]:
        return super().assign_agents_to_areas(plethos, depots)
    

    def assign_depot_to_cluster(self, clusters:Any, depots_df:pd.DataFrame, distance_matrix):

        def find_duplicate_values(mapping:Dict)->Dict: 
            reverse = {} 
            for key, value in mapping.items():
                reverse.setdefault(value, []).append(key)
            return {val:keys for val, keys in reverse.items() if len(keys) > 1}

        cluster_depot = {} 

        for cluster_id, cluster_df in clusters:
            # Calculate cluster centroid directly
            centroid = cluster_df['Area_id']

            # Find nearest depot 
            min_dist = float('inf')
            best_depot:int = 0 
            for _, depot_row in depots_df.iterrows(): 

                depot_id = int(depot_row['Area_id'])
                for centre in centroid: 
                    dist = distance_matrix[depot_id,int(centre)]
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_depot = depot_id

            cluster_depot[cluster_id] = int(best_depot)


        dulpicates = find_duplicate_values(self.depots_for_agents) 
        return cluster_depot, dulpicates
    

    def separate_depots_from_clusters(self, data: pd.DataFrame): 
        """
        Separates depot rows from cluster data based on depots assigned to agents.
        """

        unique_depots= list(set(self.depots_for_agents.values()))

        is_depot = data['Area_id'].isin(unique_depots)
        depot_data = data[is_depot].copy() 
    
        cluster_data = data[~is_depot].copy()
        return cluster_data, depot_data 
    

    def add_depot_data_to_cluster(self, cluster_df:Tuple[int,pd.DataFrame], depot_row:pd.DataFrame, depot_id:int):

        depot_row= depot_row.copy()
        depot_row  = depot_row[depot_row['Area_id']==depot_id]
        depot_row.loc[:, 'cluster'] = np.mean(cluster_df[1]['cluster'])        
        updated_cluster = pd.concat([cluster_df[1], depot_row], ignore_index=True)
        
        return updated_cluster
    

    def allocate_agents_to_clusters(self, cluster_with_depots:Dict[int,int], priority:pd.DataFrame, duplicates:Dict[int,List[int]]): 
        
        def iterate_depots_for_assignment(cluster_df, depots_for_agents, assignments): 
            
            for depot in set(depots_for_agents.values()): 
                depot_agents = duplicates.get(depot, [
                    agent for agent in self.agents if self.depots_for_agents[agent] == depot
                ])
                depot_clusters = cluster_df[cluster_df['depot']==depot]

                if depot_clusters.empty: 
                    logger.debug(f"❌ No clusters found for depot {depot}")
                    continue

                top_cluster = depot_clusters.index[0]
                if (top_cluster, depot) not in assignments: 
                    assignments[(top_cluster, depot)] = depot_agents
                cluster_df = cluster_df.drop(index=top_cluster)
                
            return cluster_df, assignments 

        # Convert depot assignments to dataframe 
        cluster_df = pd.DataFrame.from_dict(cluster_with_depots, orient='index', columns=['depot'])

        # Join with priority dataframe (not sorted priority)
        cluster_df = cluster_df.join(priority) 
        cluster_df.sort_values(by='Rank', ascending=True, inplace=True)
        assignments = {} 

        while not cluster_df.empty:
            cluster_df, assignments = iterate_depots_for_assignment(cluster_df, self.depots_for_agents, assignments)

        return assignments 


    @timeout_decorator.timeout(1800)
    def solve_problem(self, cluster:Any):
        default_limit = 500 if self.objective_function == "coverage" else None
        time_limit = self.solver_time_limit_seconds if self.solver_time_limit_seconds is not None else default_limit
        solve_metadata = solve_cluster_problem(
            cluster=cluster,
            builder=self,
            time_limit_seconds=time_limit,
            solver_backend=self.solver_backend,
            solver_seed=self.solver_seed,
        )
        cluster.solve_metadata = solve_metadata
        self.solve_status_history.append({"cluster_id": cluster.id, **solve_metadata})



    def preprocess_generated_data(self, distance_matrix:np.ndarray, centroids:list, depots:np.ndarray, num_of_agents:int,  v_ver:float,  v_hor:float,  altitude:int,  coverage_time:int,  user_points=defaultdict()):
        data = super().preprocess_generated_data(
            distance_matrix=distance_matrix,
            centroids=centroids, 
            user_points=user_points,
            depots=depots, 
            num_of_agents=num_of_agents,
            v_ver=v_ver, 
            v_hor=v_hor, 
            altitude=altitude,
            coverage_time=coverage_time
        )
        self.depots_for_agents = self.assign_agents_to_areas(plethos=len(self.agents), depots=self.depots)
        logger.debug("✅ Preprocessing of generated data completed successfully...")
        return data 


    def set_memory_limit(self, max_memory = 1024):
        return super().set_memory_limit(max_memory)
    

    def createGeoDataset(self, data):
        return super().createGeoDataset(data)
     

    def run_model(self, distance_matrix:np.ndarray, data:pd.DataFrame, cue_groups:Dict[int,List[Any]])->Any:
        self._prepare_run_state()
        self.metrics.start_run(
            run_context={
                "model_name": self.model_name,
                "scenario": self.scenario,
                "objective_function": self.objective_function,
                "environment_type": self.env_type,
                "number_of_areas": self.NUMBER_OF_AREAS,
                "number_of_agents": self.NUMBER_OF_AGENTS,
                "number_of_users": self.NUMBER_OF_USERS,
                "stage_solution": self.stage_solution,
                "subtour_mode": self.subtour_mode,
                "subtour_strategy": self.subtour_strategy,
                "solver_backend": self.solver_backend,
            },
            run_id=f"{self.id}_{int(time.time() * 1000)}",
        )

        if self.learning_enabled and self.learning_controller is not None:
            self.learning_controller.start_episode(
                builder=self,
                distance_matrix=distance_matrix,
                data=data,
                cue_groups=cue_groups,
            )
        self.user_points = cue_groups

        logger.debug("Running modular optimization pipeline...")

        self.run_request = self.build_run_request()
        self.current_problem_instance = build_problem_instance(self, distance_matrix, data, cue_groups)
        self.total_number_cluster = len(self.current_problem_instance.prepared_clusters)

        cluster_results: List[ClusterSolveResult] = []
        with tqdm(total=len(self.current_problem_instance.prepared_clusters), desc="Solving problem ", unit="cluster") as pbar:
            for prepared_cluster in self.current_problem_instance.prepared_clusters:
                cluster_result = self.optimization_model.solve_cluster(
                    self.current_problem_instance,
                    prepared_cluster,
                    self.run_request,
                )
                cluster_results.append(cluster_result)
                pbar.update(1)
                logger.debug(f"✅ Cluster {prepared_cluster.cluster_id} solved successfully...")
        
        self.metrics.end_performance_timer() 
        self.metrics.get_memory_usage()

        logger.info("Total Number of Constraints : {}".format(self.num_constraints))
        logger.info("Total Number of Variables : {}".format(self.variables_count))

        self.plan_with_nodes = copy.deepcopy(self.coordinated_plan)
        for agent, plan in self.coordinated_plan.items():
            self.coordinated_plan[agent] = self.get_coordinates_for_path(plan)
            
        rows = []
        for agent, path in self.coordinated_plan.items():
            for (x0, y0), (x1, y1), t in path:
                x0, y0 = transformer_to_latlon.transform(x0, y0)
                x1, y1 = transformer_to_latlon.transform(x1, y1)
                rows.append({
                    'agent':      agent,
                    'from_x':     x0,
                    'from_y':     y0,
                    'to_x':       x1,
                    'to_y':       y1,
                    'time_step':  t
                })
        df = pd.DataFrame(rows)
        df.to_csv(CENTROIDS_PATH, index=False)
        self.latest_playback_rows, self.latest_playback_metadata = self.build_playback_timeline(rows)

        self.latest_run_summary = self.build_run_summary()
        overall_statuses = [result.normalized_status for result in cluster_results]
        all_clusters_have_incumbent = bool(cluster_results) and all(result.incumbent_value is not None for result in cluster_results)

        # Here the path has been solved for each cluster. 
        if overall_statuses and all(status == "optimal" for status in overall_statuses):
            normalized_status = "optimal"
            raw_status = "Optimal"

        elif overall_statuses and all(status in {"optimal", "feasible"} for status in overall_statuses):
            normalized_status = "feasible"
            raw_status = "Feasible"

        elif overall_statuses and all(status in {"optimal", "feasible", "feasible_time_limit"} for status in overall_statuses):
            normalized_status = "feasible" if all_clusters_have_incumbent else "feasible_time_limit"
            raw_status = "Feasible" if all_clusters_have_incumbent else "Not Solved"

        else:
            normalized_status = "error"
            raw_status = "Error"

        summary_objective_value = self.latest_run_summary.get("objective_value")
        solver_objective_value = (
            float(sum(float(result.incumbent_value) for result in cluster_results))
            if cluster_results and all(result.incumbent_value is not None for result in cluster_results)
            else None
        )
        time_limit = self.run_request.solver_time_limit_seconds if self.run_request is not None else self.solver_time_limit_seconds
        best_bound = (
            float(sum(float(result.best_bound) for result in cluster_results))
            if cluster_results and all(result.best_bound is not None for result in cluster_results)
            else None
        )

        self.latest_model_run_result = ModelRunResult(
            run_id=self.metrics.run_id,
            instance_id=self.current_problem_instance.instance_id,
            model_name=self.model_name,
            raw_status=raw_status,
            normalized_status=normalized_status,
            objective_value=float(solver_objective_value) if solver_objective_value is not None else None,
            summary_objective_value=float(summary_objective_value) if summary_objective_value is not None else None,
            incumbent_value=float(solver_objective_value) if solver_objective_value is not None else None,
            best_bound=best_bound,
            absolute_gap=compute_absolute_gap(
                float(solver_objective_value) if solver_objective_value is not None else None,
                best_bound,
            ),
            relative_gap=compute_relative_gap(
                float(solver_objective_value) if solver_objective_value is not None else None,
                best_bound,
            ),
            elapsed_time_seconds=float(getattr(self.metrics, "elapsed_time", 0.0) or 0.0),
            time_limit_seconds=float(time_limit) if time_limit is not None else None,
            termination_reason=infer_termination_reason(raw_status, time_limit),
            summary=dict(self.latest_run_summary),
            cluster_results=cluster_results,
            diagnostics={
                "solve_status_history": list(self.solve_status_history),
                "cluster_status_records": list(self.cluster_status_records),
                "coordinated_plan_agents": sorted(self.coordinated_plan.keys()),
            },
        )
        self.latest_run_summary["solver_objective_value"] = float(solver_objective_value) if solver_objective_value is not None else None
        self.latest_run_summary["summary_objective_value"] = (
            float(summary_objective_value) if summary_objective_value is not None else None
        )
        self.latest_run_summary["best_bound"] = float(best_bound) if best_bound is not None else None
        self.latest_run_summary["optimality_gap_percent"] = (
            float(self.latest_model_run_result.relative_gap) * 100.0
            if self.latest_model_run_result.relative_gap is not None
            else None
        )

        if self.learning_enabled and self.learning_controller is not None:
            self.learning_controller.finish_episode(self, self.latest_run_summary)

        return self.coordinated_plan     


    def _prepare_run_state(self) -> None:
        self._seed_random_generators()
        self.cluster_status_records = []
        self.solve_status_history = []
        self.latest_run_summary = {}
        self.latest_run_report = {}
        self.latest_run_report_path = ""
        self.latest_artifact_dir = ""
        self.latest_playback_rows = []
        self.latest_playback_metadata = {}
        self.latest_model_run_result = None
        self.current_problem_instance = None
        self.run_request = None
        self.problem_results = defaultdict()
        self.coordinated_plan = defaultdict(dict)
        self.plan_with_nodes = defaultdict(dict)
        self.total_data_rate = 0.0
        self.makespan = 0.0
        self.global_nodes_visited = 0
        self.visits_per_nodes = {}
        self.total_number_cluster = 0
        self.num_constraints = 0
        self.variables_count = 0
        self.num_binary_variables = 0
        self.num_continuous_variables = 0
        self.agent_next_available_time = {}
        self.enable_ga = self.base_enable_ga
        self.metrics.reset()


    def apply_runtime_configuration(self, runtime_config: Dict[str, Any]) -> None:
        self.stage_solution = int(runtime_config.get("stage_solution", self.stage_solution))
        self.ga_generations = int(runtime_config.get("ga_generations", self.ga_generations))
        self.solver_time_limit_seconds = runtime_config.get(
            "solver_time_limit_seconds",
            runtime_config.get("time_limit_seconds", self.solver_time_limit_seconds),
        )
        self.random_seed = int(runtime_config.get("random_seed", runtime_config.get("seed", self.random_seed)))
        self.solver_seed = int(runtime_config.get("solver_seed", runtime_config.get("seed", self.solver_seed)))
        self.fairness_tolerance = int(runtime_config.get("fairness_tolerance", self.fairness_tolerance))
        self.time_step_sec = int(runtime_config.get("time_step_sec", self.time_step_sec))
        self.warm_start_mode = normalize_warm_start_mode(runtime_config.get("warm_start_mode", self.warm_start_mode))
        self.objective_weights = runtime_config.get("objective_weights", self.objective_weights)
        self.clustering_feature_weights = runtime_config.get("clustering_feature_weights", self.clustering_feature_weights)
        self.enable_ga = bool(runtime_config.get("enable_ga", self.base_enable_ga))
        self.model_name = str(runtime_config.get("model_name", self.model_name))
        self.subtour_mode = normalize_subtour_mode(runtime_config.get("subtour_mode", runtime_config.get("subtour_strategy", self.subtour_mode)))
        self.subtour_strategy = self.subtour_mode
        self.solver_backend = str(runtime_config.get("solver_backend", self.solver_backend))
        self.objective_strategy = str(runtime_config.get("objective_strategy", self.objective_strategy))
        self.scenario_constraint_set = str(runtime_config.get("scenario_constraint_set", self.scenario_constraint_set))
        bridge_visit_override = runtime_config.get(
            "bridge_node_required_visits_override",
            self.bridge_node_required_visits_override,
        )
        self.bridge_node_required_visits_override = (
            None if bridge_visit_override in {None, "", 0} else int(bridge_visit_override)
        )
        self._seed_random_generators()
        self.optimization_model = self._create_optimization_model(self.model_name)


    def build_run_request(self) -> ModelRunRequest:
        return ModelRunRequest(
            model_name=self.model_name,
            solver_backend=self.solver_backend,
            subtour_mode=self.subtour_mode,
            subtour_strategy=self.subtour_mode,
            scenario_constraint_set=self.scenario_constraint_set,
            objective_strategy=self.objective_strategy,
            warm_start_strategy=self.warm_start_mode,
            solver_time_limit_seconds=self.solver_time_limit_seconds,
            objective_weights=dict(self.objective_weights),
            clustering_feature_weights=dict(self.clustering_feature_weights),
            options={
                "stage_solution": self.stage_solution,
                "enable_ga": self.enable_ga,
                "ga_generations": self.ga_generations,
                "random_seed": self.random_seed,
                "solver_seed": self.solver_seed,
                "fairness_tolerance": self.fairness_tolerance,
                "time_step_sec": self.time_step_sec,
                "bridge_node_required_visits_override": self.bridge_node_required_visits_override,
            },
        )
    

    def get_coordinates_for_path(self, path): 
        coordinates = []
        for point in path: 
            try:
                current_node = self.V.index[self.V['Area_id'] == point[0]][0] 
                next_node = self.V.index[self.V['Area_id'] == point[1]][0]
                current_coords = (float(self.V['X_coords'].iloc[current_node]), float(self.V['Y_coords'].iloc[current_node]))
                next_coords = (float(self.V['X_coords'].iloc[next_node]), float(self.V['Y_coords'].iloc[next_node]))
                coordinates.append((current_coords, next_coords, point[2]))
           
            except IndexError:
                logger.error(f"❌ Node {point[0]} or {point[1]} not found in the dataframe.")
                continue

        return coordinates


    def build_playback_timeline(self, coordinate_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not coordinate_rows:
            return [], {"max_time_step": 0, "num_agents": 0}

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in coordinate_rows:
            grouped[str(row["agent"])].append(dict(row))

        playback_rows: List[Dict[str, Any]] = []
        for agent_id, agent_rows in grouped.items():
            agent_rows.sort(key=lambda item: int(item["time_step"]))
            index = 0
            while index < len(agent_rows):
                current = agent_rows[index]
                segment_rows = [current]
                next_index = index + 1
                while next_index < len(agent_rows):
                    candidate = agent_rows[next_index]
                    if int(candidate["time_step"]) != int(segment_rows[-1]["time_step"]) + 1:
                        break
                    if (
                        float(candidate["from_x"]) != float(current["from_x"])
                        or float(candidate["from_y"]) != float(current["from_y"])
                        or float(candidate["to_x"]) != float(current["to_x"])
                        or float(candidate["to_y"]) != float(current["to_y"])
                    ):
                        break
                    segment_rows.append(candidate)
                    next_index += 1

                duration = len(segment_rows)
                is_wait_segment = (
                    float(current["from_x"]) == float(current["to_x"])
                    and float(current["from_y"]) == float(current["to_y"])
                )
                for offset, segment_row in enumerate(segment_rows):
                    if is_wait_segment:
                        x_pos = float(current["from_x"])
                        y_pos = float(current["from_y"])
                    else:
                        # Convert transition rows into state samples by placing the
                        # first rendered sample at the segment origin.
                        progress = float(offset) / float(max(duration, 1))
                        x_pos = float(current["from_x"]) + (float(current["to_x"]) - float(current["from_x"])) * progress
                        y_pos = float(current["from_y"]) + (float(current["to_y"]) - float(current["from_y"])) * progress

                    playback_rows.append(
                        {
                            "agent": agent_id,
                            "time_step": int(segment_row["time_step"]),
                            "x": x_pos,
                            "y": y_pos,
                            "segment_type": "wait" if is_wait_segment else "move",
                            "from_x": float(current["from_x"]),
                            "from_y": float(current["from_y"]),
                            "to_x": float(current["to_x"]),
                            "to_y": float(current["to_y"]),
                        }
                    )

                index = next_index

        playback_df = pd.DataFrame(playback_rows)
        if not playback_df.empty:
            playback_df = (
                playback_df.sort_values(["agent", "time_step"])
                .drop_duplicates(subset=["agent", "time_step"], keep="last")
                .reset_index(drop=True)
            )
        metadata = {
            "max_time_step": int(playback_df["time_step"].max()) if not playback_df.empty else 0,
            "min_x": float(playback_df["x"].min()) if not playback_df.empty else 0.0,
            "max_x": float(playback_df["x"].max()) if not playback_df.empty else 0.0,
            "min_y": float(playback_df["y"].min()) if not playback_df.empty else 0.0,
            "max_y": float(playback_df["y"].max()) if not playback_df.empty else 0.0,
            "num_agents": int(playback_df["agent"].nunique()) if not playback_df.empty else 0,
        }
        return playback_df.to_dict(orient="records"), metadata


    def regionalization(self, GDF):
        return super().regionalization(GDF)
    
    
    def cluster_prioritization(self, clusters, cue_groups: Mapping[int, Any], distance_matrix):
        return super().cluster_prioritization(clusters, cue_groups, distance_matrix)
    

    def _prepare_cluster_object(self, cluster_input) -> Cluster:
        cluster_object = Cluster(
            cluster=cluster_input.cluster_frame,
            id=cluster_input.cluster_id,
            assignment=cluster_input.assigned_agents,
            depot_id=cluster_input.depot_id,
            max_battery=self.max_battery,
        )
        cluster_object.agent_start_times = {
            agent_id: float(self.agent_next_available_time.get(agent_id, 0.0))
            for agent_id in cluster_input.assigned_agents
        }

        context = cluster_object.get_cluster_content(
            distance=self.distance_columns,
            energy=self.energy_columns,
            time=self.travel_time_columns,
            column_names=["dists", "ees", "travel_times", "area_ids"],
        )

        logger.debug(f"Clustering with {cluster_input.cluster_id} and agents assigned to it: {cluster_input.assigned_agents}")

        try:
            cluster_object.prepare_context(context=context, builder=self)
            logger.debug(f"✅ Context prepared for cluster {cluster_input.cluster_id} successfully...")
        except Exception as exc:
            logger.exception(f"❌ Error processing cluster {cluster_input.cluster_id}: {exc}")
            raise ValueError(f"Error processing cluster {cluster_input.cluster_id}: {exc}") from exc

        cluster_object.set_up_virtual_nodes_properties()
        cluster_object.get_estimated_time_frame(self)
        self.get_cluster_coverage(cluster_object)

        del context
        gc.collect()
        return cluster_object


    def _build_cluster_comparison_metrics(
        self,
        cluster_object: Cluster,
        cluster_input,
        results: Dict[int, Dict[str, Any]],
        uncovered_nodes: List[int] | None = None,
    ) -> Dict[str, Any]:
        target_nodes = sorted(
            {
                int(node_id)
                for node_id in cluster_input.cluster_frame["Area_id"].tolist()
                if int(node_id) != int(cluster_input.depot_id)
            }
        )
        covered_nodes = set()
        nodes_per_uav: Dict[str, int] = {}
        route_distance_per_uav: Dict[str, float] = {}

        for agent_id in cluster_input.assigned_agents:
            visited_nodes = results.get(agent_id, {}).get("visited_nodes", [])
            served_nodes = {int(node_id) for node_id in visited_nodes if int(node_id) != int(cluster_input.depot_id)}
            covered_nodes.update(served_nodes)
            nodes_per_uav[str(agent_id)] = len(served_nodes)
            route_distance_per_uav[str(agent_id)] = float(results.get(agent_id, {}).get("distance", 0.0))

        if uncovered_nodes:
            uncovered_physical_nodes = sorted({int(node_id) for node_id in uncovered_nodes if int(node_id) != int(cluster_input.depot_id)})
        else:
            uncovered_physical_nodes = sorted(set(target_nodes) - covered_nodes)

        node_counts = list(nodes_per_uav.values())
        return {
            "covered_nodes": len(covered_nodes),
            "coverage_ratio": float(len(covered_nodes)) / float(len(target_nodes)) if target_nodes else 0.0,
            "num_uavs_used": sum(1 for node_count in node_counts if node_count > 0),
            "max_route_distance_per_uav": max(route_distance_per_uav.values(), default=0.0),
            "workload_imbalance": (max(node_counts) - min(node_counts)) if node_counts else 0,
            "nodes_per_uav": nodes_per_uav,
            "route_distance_per_uav": route_distance_per_uav,
            "uncovered_nodes": uncovered_physical_nodes,
            "target_node_count": len(target_nodes),
        }


    def _record_heuristic_path_visits(
        self,
        cluster_object: Cluster,
        paths: Dict[int, List[Tuple[int, int, int]]],
        task_sequences: Dict[int, List[int]] | None = None,
    ) -> None:
        unique_nodes_among_paths = set()
        for agent_path in paths.values():
            if not agent_path:
                continue

            for source, target, _ in agent_path:
                unique_nodes_among_paths.add(int(source))
                unique_nodes_among_paths.add(int(target))

        if task_sequences is not None:
            for route_nodes in task_sequences.values():
                if not route_nodes:
                    continue
                for node_id in route_nodes:
                    node_key = int(cluster_object.virtual_nodes.get(int(node_id), int(node_id)))
                    self.visits_per_nodes[node_key] = self.visits_per_nodes.get(node_key, 0) + 1
                depot_key = int(cluster_object.depot_id)
                self.visits_per_nodes[depot_key] = self.visits_per_nodes.get(depot_key, 0) + 1
        else:
            segments = []
            for agent_path in paths.values():
                if not agent_path:
                    continue
                current_source, current_target, _ = agent_path[0]
                for source, target, _ in agent_path[1:]:
                    if (source, target) == (current_source, current_target):
                        continue
                    segments.append((current_source, current_target))
                    current_source, current_target = source, target
                segments.append((current_source, current_target))

            for source, target in segments:
                if source == target and int(target) != int(cluster_object.depot_id):
                    self.visits_per_nodes[int(target)] = self.visits_per_nodes.get(int(target), 0) + 1
                elif int(target) == int(cluster_object.depot_id):
                    self.visits_per_nodes[int(target)] = self.visits_per_nodes.get(int(target), 0) + 1

        self.global_nodes_visited += len(unique_nodes_among_paths)
        self.validate_paths(paths=paths, nodes_dict=cluster_object.nodes_dict, cluster=cluster_object)


    def _count_problem_variables(self, cluster_object: Cluster) -> Dict[str, int]:
        problem = getattr(cluster_object, "problem", None)
        if problem is None:
            return {
                "num_variables": 0,
                "num_binary_variables": 0,
                "num_continuous_variables": 0,
            }

        total_variables = 0
        binary_variables = 0
        continuous_variables = 0

        for variable in problem.variables():
            total_variables += 1

            is_binary = False
            if hasattr(variable, "isBinary"):
                try:
                    is_binary = bool(variable.isBinary())
                except Exception:
                    is_binary = False
            if not is_binary:
                try:
                    is_binary = (
                        float(getattr(variable, "lowBound", None)) == 0.0
                        and float(getattr(variable, "upBound", None)) == 1.0
                        and str(getattr(variable, "cat", "")).strip().lower() == "integer"
                    )
                except Exception:
                    is_binary = False

            if is_binary:
                binary_variables += 1
                continue

            if str(getattr(variable, "cat", "")).strip().lower() == "continuous":
                continuous_variables += 1

        return {
            "num_variables": int(total_variables),
            "num_binary_variables": int(binary_variables),
            "num_continuous_variables": int(continuous_variables),
        }


    def _finalize_cluster_solution(
        self,
        cluster_object: Cluster,
        cluster_input,
        request: ModelRunRequest,
        paths: Dict[int, List[Tuple[int, int, int]]],
        raw_status: str,
        status_code: int,
        objective_value: float | None = None,
        incumbent_value: float | None = None,
        best_bound: float | None = None,
        absolute_gap: float | None = None,
        relative_gap: float | None = None,
        elapsed_time_seconds: float = 0.0,
        time_limit_seconds: float | None = None,
        termination_reason: str | None = None,
        diagnostics: Dict[str, Any] | None = None,
        agent_finish_times: Dict[int, float] | None = None,
        record_visits: bool = False,
        uncovered_nodes: List[int] | None = None,
        task_sequences: Dict[int, List[int]] | None = None,
    ) -> ClusterSolveResult:
        if record_visits:
            self._record_heuristic_path_visits(cluster_object, paths, task_sequences=task_sequences)

        results = common.extract_per_agent_metrics(
            paths=paths,
            costs=self.problem_cost_data,
            coverage_energy=self.average_coverage_energy,
            virtual_nodes=cluster_object.virtual_nodes,
            area_ids=cluster_object.original_nodes_dict.values(),
            file_id=self.id,
        )

        total_distance, total_energy, total_time = common.calculate_totals_from_paths(results=results)
        if agent_finish_times is None:
            agent_finish_times = {
                int(agent_id): float(agent_path[-1][2] + 1) if agent_path else 0.0
                for agent_id, agent_path in paths.items()
            }

        agent_next_available_times = {}
        for agent_id in cluster_input.assigned_agents:
            energy_spent = float(results.get(agent_id, {}).get("energy", 0.0))
            recharge_steps = float(common.calculate_recharge_steps(self.max_battery, energy_spent=energy_spent))
            finish_time = float(agent_finish_times.get(agent_id, 0.0))
            next_available_time = finish_time + recharge_steps
            agent_next_available_times[agent_id] = next_available_time
            self.agent_next_available_time[agent_id] = next_available_time
            results.setdefault(agent_id, {})
            results[agent_id]["scheduled_start_time"] = float(cluster_object.agent_start_times.get(agent_id, 0.0))
            results[agent_id]["scheduled_finish_time"] = finish_time
            results[agent_id]["next_available_time"] = next_available_time
            results[agent_id]["recharge_steps"] = recharge_steps

        cluster_total_data_transfer = float(getattr(cluster_object, "total_data_achievable", 0.0) or 0.0)
        cluster_makespan = float(getattr(cluster_object, "makespan_value", 0.0) or 0.0)
        cluster_absolute_makespan = float(getattr(cluster_object, "absolute_makespan_value", cluster_makespan) or cluster_makespan)

        self.total_data_rate += cluster_total_data_transfer
        self.makespan += cluster_makespan

        self.problem_results[f"Cluster_{cluster_object.id}"] = {
            "scenario_name": self.scenario,
            "objective_function": self.objective_function,
            "agent_results": results,
            "Total Distance": total_distance,
            "Total Energy": total_energy,
            "Total Time": total_time,
            "Average Throughput": cluster_object.R,
            "Average SINR": cluster_object.sinr,
            "Makespan": cluster_makespan,
            "Total_Data_Transfer": cluster_total_data_transfer,
            "agent_start_times": dict(cluster_object.agent_start_times),
            "agent_finish_times": dict(agent_finish_times),
            "agent_next_available_times": dict(agent_next_available_times),
        }

        cluster_status_record = {
            "cluster_id": cluster_object.id,
            "status_code": int(status_code),
            "status": raw_status,
            "agent_count": len(cluster_input.assigned_agents),
            "node_count": len(cluster_object.original_nodes_dict),
            "subtour_mode": request.subtour_mode,
            "subtour_strategy": request.subtour_mode,
            "solver_backend": request.solver_backend,
        }
        self.cluster_status_records.append(cluster_status_record)
        self.metrics.record_cluster_result(
            {
                "cluster_id": cluster_object.id,
                "status": raw_status,
                "agent_count": len(cluster_input.assigned_agents),
                "node_count": len(cluster_object.original_nodes_dict),
                "total_distance": total_distance,
                "total_energy": total_energy,
                "total_time": total_time,
                "average_throughput": dict(cluster_object.R),
                "average_sinr": dict(cluster_object.sinr),
                "makespan": cluster_makespan,
                "total_data_transfer": cluster_total_data_transfer,
                "subtour_mode": request.subtour_mode,
                "subtour_strategy": request.subtour_mode,
                "solver_backend": request.solver_backend,
            }
        )

        synced_paths = self.synchronize_agent_paths(paths, cluster_object)
        self.flatten_paths_on_time(self.coordinated_plan, synced_paths)

        comparison_metrics = self._build_cluster_comparison_metrics(
            cluster_object=cluster_object,
            cluster_input=cluster_input,
            results=results,
            uncovered_nodes=uncovered_nodes,
        )
        problem_size_metrics = self._count_problem_variables(cluster_object)
        problem_size_metrics["num_constraints"] = int(len(cluster_object.problem.constraints))
        self.num_constraints += int(problem_size_metrics["num_constraints"])
        self.variables_count += int(problem_size_metrics["num_variables"])
        self.num_binary_variables += int(problem_size_metrics["num_binary_variables"])
        self.num_continuous_variables += int(problem_size_metrics["num_continuous_variables"])
        self.problem_results[f"Cluster_{cluster_object.id}"].update(comparison_metrics)
        cluster_metrics = {
            "agent_count": len(cluster_input.assigned_agents),
            "node_count": len(cluster_object.original_nodes_dict),
            "priority_rank": cluster_input.priority_rank,
            "total_distance": total_distance,
            "total_energy": total_energy,
            "total_time": total_time,
            "makespan": cluster_makespan,
            "absolute_makespan": cluster_absolute_makespan,
            "total_data_transfer": cluster_total_data_transfer,
            **problem_size_metrics,
            **comparison_metrics,
        }

        if objective_value is None:
            if self.objective_function == "coverage":
                objective_value = float(-cluster_total_data_transfer)
            else:
                objective_value = (
                    float(self.objective_weights["energy"]) * float(total_energy)
                    + float(self.objective_weights["distance"]) * float(total_distance)
                    + float(self.objective_weights["travel_time"]) * float(total_time)
                )
        if incumbent_value is None:
            incumbent_value = objective_value

        base_diagnostics = {
            "priority_rank": cluster_input.priority_rank,
            "bridge_nodes": list(cluster_object.bridge_nodes),
            "virtual_nodes": dict(cluster_object.virtual_nodes),
            "agent_start_times": dict(cluster_object.agent_start_times),
            "agent_finish_times": dict(agent_finish_times),
            "agent_next_available_times": dict(agent_next_available_times),
            "warm_start_summary": dict(getattr(cluster_object, "warm_start_summary", {})),
            "problem_size_metrics": dict(problem_size_metrics),
            "request": asdict(request),
        }
        if diagnostics:
            base_diagnostics.update(diagnostics)

        return ClusterSolveResult(
            cluster_id=cluster_object.id,
            raw_status=raw_status,
            normalized_status=normalize_solver_status(raw_status),
            status_code=int(status_code),
            objective_value=objective_value,
            incumbent_value=incumbent_value,
            best_bound=best_bound,
            absolute_gap=absolute_gap,
            relative_gap=relative_gap,
            elapsed_time_seconds=float(elapsed_time_seconds),
            time_limit_seconds=time_limit_seconds,
            termination_reason=termination_reason
            or infer_termination_reason(raw_status, time_limit_seconds),
            agent_paths=synced_paths,
            agent_metrics={str(agent_id): metrics for agent_id, metrics in results.items()},
            cluster_metrics=cluster_metrics,
            diagnostics=base_diagnostics,
        )


    def solve_cluster_instance(self, instance, cluster_input, request: ModelRunRequest) -> ClusterSolveResult:
        cluster_object = self._prepare_cluster_object(cluster_input)

        try:
            paths = cluster_object.problem_formulation(
                builder=self,
                scenario=self.scenario,
                objective_function=self.objective_function,
                stage_solution=self.stage_solution,
            )
            logger.debug(f"✅ Problem created for cluster {cluster_input.cluster_id} successfully...")
        except ValidationOptimalityConfirmed:
            raise
        except Exception as exc:
            logger.exception(f"❌ Error creating problem for cluster {cluster_input.cluster_id}: {exc}")
            raise ValueError(f"Error in creating the problem for Cluster {cluster_input.cluster_id}") from exc

        raw_status = pl.LpStatus.get(cluster_object.problem.status, "Unknown")
        solve_metadata = dict(getattr(cluster_object, "solve_metadata", {}))
        agent_finish_times = {
            int(agent_id): float(getattr(cluster_object.return_step[agent_id], "varValue", 0.0) or 0.0)
            for agent_id in cluster_input.assigned_agents
        }
        return self._finalize_cluster_solution(
            cluster_object=cluster_object,
            cluster_input=cluster_input,
            request=request,
            paths=paths,
            raw_status=raw_status,
            status_code=int(cluster_object.problem.status),
            objective_value=solve_metadata.get("objective_value"),
            incumbent_value=solve_metadata.get("incumbent_value"),
            best_bound=solve_metadata.get("best_bound"),
            absolute_gap=solve_metadata.get("absolute_gap"),
            relative_gap=solve_metadata.get("relative_gap"),
            elapsed_time_seconds=float(solve_metadata.get("elapsed_time_seconds", 0.0) or 0.0),
            time_limit_seconds=solve_metadata.get("time_limit_seconds"),
            termination_reason=solve_metadata.get("termination_reason"),
            diagnostics={
                "solver_seed": solve_metadata.get("solver_seed"),
                "first_feasible_time_seconds": solve_metadata.get("first_feasible_time_seconds"),
                "first_optimality_gap_percent": solve_metadata.get("first_optimality_gap_percent"),
                "explored_bnb_nodes": solve_metadata.get("explored_bnb_nodes"),
                "active_bnb_nodes": solve_metadata.get("active_bnb_nodes"),
                "feasible_solution_found": solve_metadata.get("feasible_solution_found"),
                "time_limit_reached": solve_metadata.get("time_limit_reached"),
                "optimality_proven": solve_metadata.get("optimality_proven"),
                "subtour_mode": request.subtour_mode,
                "dfj_rounds": solve_metadata.get("dfj_rounds"),
                "dfj_solve_passes": solve_metadata.get("dfj_solve_passes"),
                "dfj_cuts_added": solve_metadata.get("dfj_cuts_added"),
                "dfj_round_history": solve_metadata.get("dfj_round_history", []),
                "violated_subtours": solve_metadata.get("violated_subtours", []),
                "progress_events": solve_metadata.get("progress_events", []),
                "solver_log_path": solve_metadata.get("solver_log_path", ""),
            },
            agent_finish_times=agent_finish_times,
        )


    def solve_cluster_instance_heuristic(
        self,
        instance,
        cluster_input,
        request: ModelRunRequest,
        heuristic_solver: Callable[[Any, Any], Any],
        heuristic_name: str,
    ) -> ClusterSolveResult:
        cluster_object = self._prepare_cluster_object(cluster_input)
        solve_started_at = time.perf_counter()
        heuristic_solution = heuristic_solver(cluster_object, self)
        elapsed_time_seconds = time.perf_counter() - solve_started_at

        cluster_object.total_data_achievable = float(heuristic_solution.total_data_transfer)
        cluster_object.makespan_value = float(heuristic_solution.makespan)
        cluster_object.absolute_makespan_value = float(heuristic_solution.makespan)

        return self._finalize_cluster_solution(
            cluster_object=cluster_object,
            cluster_input=cluster_input,
            request=request,
            paths=heuristic_solution.agent_paths,
            raw_status=heuristic_solution.raw_status,
            status_code=int(heuristic_solution.status_code),
            objective_value=heuristic_solution.diagnostics.get("objective_value"),
            incumbent_value=heuristic_solution.diagnostics.get("objective_value"),
            elapsed_time_seconds=float(elapsed_time_seconds),
            termination_reason="heuristic_completed",
            diagnostics={"heuristic_name": heuristic_name, **dict(heuristic_solution.diagnostics)},
            agent_finish_times=heuristic_solution.agent_finish_times,
            record_visits=True,
            uncovered_nodes=heuristic_solution.diagnostics.get("uncovered_physical_nodes", []),
            task_sequences=heuristic_solution.task_sequences,
        )


    def clustering(self, cluster, cluster_id, assignment, depot_id)->Dict:
        legacy_request = self.run_request or self.build_run_request()
        cluster_result = self.solve_cluster_instance(
            instance=self.current_problem_instance,
            cluster_input=type(
                "LegacyClusterInput",
                (),
                {
                    "cluster_frame": cluster,
                    "cluster_id": cluster_id,
                    "assigned_agents": assignment,
                    "depot_id": depot_id,
                    "priority_rank": None,
                },
            )(),
            request=legacy_request,
        )
        return cluster_result.agent_paths


    def get_depot_index(self, ordered_nodes, k): 
        axx = [i for i, value in enumerate(ordered_nodes.values()) if value == self.depots_for_agents[k]]
        return axx[0]


    def validate_paths(self, paths, nodes_dict, cluster):
        max_time_steps = cluster.timeframe[-1]
        reverse = {v: k for k, v in nodes_dict.items()}
        depot_ind = reverse[cluster.depot_id]
        all_paths = {} 
        key_points = {} 
        
        for agent_id, path in paths.items(): 
            visit_nodes = set() 
            seen_edges = set()

            # 1. Reject agents that haven't been used at this point. 
            if len(path) == 0: 
                logger.debug(f"Agent {agent_id} has no path")
                continue 

            if path[-1][1] != cluster.depot_id: 
                raise ValueError(f"{agent_id} does not return to depot")
            
            visit_nodes.add(cluster.depot_id)

            # 2. Include the very last arrival node
            for step in path:
                for node in (step[0], step[1]):
                    if node not in (cluster.depot_id,) + tuple(cluster.bridge_nodes):
                        visit_nodes.add(node)

            for i in range(len(path)-1):
                
                step = path[i] 
                source_node = step[0] 
                target_node = step[1]
                time_step = step[2] 

                edge = (source_node, target_node)
                keypoint = (target_node, time_step)
                
                if keypoint in key_points and target_node != nodes_dict[depot_ind] and source_node!=nodes_dict[depot_ind]: 
                    logger.debug(f"Collision: Agent {agent_id} and Agent {key_points[keypoint]} from node {source_node} at node {keypoint[0]} at time {keypoint[1]}")

                key_points[keypoint] = agent_id
                
                if edge in seen_edges: 
                    logger.debug(f"Edge {edge} already seen for agent {agent_id}")

                seen_edges.add(edge)
                 
                if source_node not in visit_nodes and source_node != nodes_dict[depot_ind]: 
                    visit_nodes.add(source_node)

                if i != len(path)-2 and target_node == cluster.depot_id:
                    logger.debug(f"Agent {agent_id} visited node {target_node} at time {time_step} before visiting node {cluster.depot_id} at time {time_step+1}")

                if time_step > max_time_steps: 
                    logger.debug(f"Agent {agent_id} has gone over the time limit")


            edge_sequence = tuple((step[0], step[1]) for step in path)
            all_paths[agent_id] = edge_sequence
            
            if self.scenario == "individual": 
                if len(visit_nodes) != len(nodes_dict)-1 : 
                    logger.debug(f"Agent {agent_id} visited only {len(visit_nodes)} nodes out of {len(nodes_dict)-1}")

            logger.debug(f"Agent {agent_id} | Visited_nodes == > {sorted(visit_nodes)} | Cluster_nodes == > {cluster.nodes_dict.values()} | Bridge Nodes == > {cluster.bridge_nodes}")

             
        agent_ids = list(all_paths.keys()) 
        for i in range(len(agent_ids)): 
            for j in range(i + 1, len(agent_ids)):
                if all_paths[agent_ids[i]] == all_paths[agent_ids[j]]:
                    logger.debug(f"Agents {agent_ids[i]} and {agent_ids[j]} have identical paths!")

        
    def get_travel_time(self, i, j, nodes_dict): 
        return math.ceil(self.travel_cost[nodes_dict[i]-1, nodes_dict[j]-1])
             

    def post_process(self, coordinated_plan): 

        def check_for_duplicates(path): 
            # Build a new list instead of modifying the old one.
            unique_plan = []
            if path:
                unique_plan.append(path[0])
                for i in range(len(path) - 1):
                    if path[i] != path[i+1]:
                        unique_plan.append(path[i+1])
                    
            return unique_plan


        if not coordinated_plan: return [] 
        interpolated_paths = {}
        for agent, plan in coordinated_plan.items(): 

            if not plan:
                logger.debug(f"Agent {agent} has no path in cluster {agent}")
                continue

            path = check_for_duplicates(plan)
            
            detailed_log = [] 

            for i in range(len(path)):

                # Get the current event and its start time 
                from_node_id, to_node_id, start_time = path[i] 

                if i + 1 < len(path):
                    end_time = path[i+1][2]
                else:
                    end_time = start_time + 1

                duration = int(round(end_time - start_time))

                start_pos = (from_node_id[0], from_node_id[1])
                end_pos = (to_node_id[0], to_node_id[1]) 

    
                if from_node_id == to_node_id:
                    for t in range(duration):
                        current_time = int(round(start_time)) + t
                        detailed_log.append((*start_pos, current_time))

                else:
                    if duration == 0: # If duration is zero, just add the start point
                        detailed_log.append((*start_pos, int(round(start_time))))
                        continue

                    x_coords = np.linspace(start_pos[0], end_pos[0], duration)
                    y_coords = np.linspace(start_pos[1], end_pos[1], duration)

                    for t in range(duration):
                        current_time = int(round(start_time)) + t
                        detailed_log.append((x_coords[t], y_coords[t], current_time))
                        
            interpolated_paths[agent] = detailed_log

        return interpolated_paths
    

    def synchronize_agent_paths(self, paths:Dict ,cluster:Any)->Dict: 
       
        # Calculate recharge steps 
        problem_results = self.problem_results[f'Cluster_{cluster.id}']['agent_results']
        
        recharge_times = defaultdict(float)

        synced_paths = {} 
        agent_end_times = {}

        for agent in paths: 
            recharge_times[agent] = float(
                problem_results[agent].get(
                    'recharge_steps',
                    common.calculate_recharge_steps(self.max_battery, energy_spent=problem_results[agent]['energy']),
                )
            )
            agent_end_times[agent] = paths[agent][-1][2] + recharge_times[agent] if paths[agent] else -1 
            self.problem_results[f'Cluster_{cluster.id}']['agent_results'][agent]['recharge_steps'] = recharge_times[agent]

        max_mission_time = max(agent_end_times.values()) if agent_end_times else 0 

        new_sync_time = max_mission_time 

        for agent_id, path in paths.items(): 
            if not path:
                synced_paths[agent_id] = [] 
                continue 

            new_path = list(path) 
            last_step = new_path[-1] 
            last_pos = (last_step[0], last_step[1])
            if last_step[0] != cluster.depot_id: 
                last_pos = (int(cluster.depot_id), last_step[1])
            
            current_time = last_step[2] 

            for i in range(int(recharge_times[agent_id])):
                current_time += 1
                new_path.append((*last_pos, current_time))

            idle_steps_needed = new_sync_time - current_time 
            for i in range(int(idle_steps_needed)):
                current_time += 1
                new_path.append((*last_pos, current_time))

            synced_paths[agent_id] = new_path

        return synced_paths 
    

    def flatten_paths_on_time(self, master_paths, new_paths): 

        for agent_id, new_path in new_paths.items():
            if not new_path: 
                continue 

            if agent_id not in master_paths or not master_paths[agent_id]: 
                master_paths[agent_id] = new_path 

            else: 
                last_global_time = master_paths[agent_id][-1][-1] 

                shifted_path = [] 
                for step in new_path: 
                    if len(step) ==3 :
                        x, y, local_time = step 
                    elif len(step) == 4: 
                        x, y, z, local_time = step 
                    else: 
                        x, local_time = step
                    
                    new_global_time = local_time + last_global_time + 1 
                    
                    if len(step) == 3:
                        shifted_path.append((x, y, new_global_time))
                    elif len(step) == 4:
                        shifted_path.append((x, y, z, new_global_time))

                master_paths[agent_id].extend(shifted_path)

        return master_paths
                        

    def get_cluster_coverage(self, cluster:Any):
        altitude = self.agent_altitude/1e3 
        user_height = 1.25/1e3 
        terrain_type = self.env_type 
        coverage_summary = cluster.get_average_coverage(
            user_points = self.user_points,
            altitude = altitude,
            user_height = user_height,
            terrain_type = terrain_type,
            filename=f"coverage_cluster_{cluster.id}.csv",
        )
        self.metrics.record_coverage_result(coverage_summary)
            

    def gather_results(self): 

        if not self.latest_run_summary:
            self.latest_run_summary = self.build_run_summary()

        coverage_diagnostics = self.latest_run_summary.get("coverage_diagnostics", {})
        if coverage_diagnostics:
            logger.info(f"Average coverage diagnostics: {coverage_diagnostics}")

        self.latest_run_report = self.metrics.build_run_report(
            summary=self.latest_run_summary,
            cluster_results=dict(self.problem_results),
            solve_status_history=self.solve_status_history,
        )
        if self.current_problem_instance is not None and self.run_request is not None and self.latest_model_run_result is not None:
            artifact_dir = persist_run_artifacts(
                base_dir=self.metrics.base_dir,
                instance=self.current_problem_instance,
                request=self.run_request,
                result=self.latest_model_run_result,
            )
            self.latest_artifact_dir = str(artifact_dir)
            if self.latest_playback_rows:
                persist_playback_artifacts(
                    artifact_dir=artifact_dir,
                    playback_rows=self.latest_playback_rows,
                    playback_metadata=self.latest_playback_metadata,
                )
            self.latest_run_report["artifact_dir"] = self.latest_artifact_dir
            self.latest_run_report["model_run_result"] = asdict(self.latest_model_run_result)
        report_path = self.metrics.persist_run_report(self.latest_run_report)
        self.latest_run_report_path = str(report_path)

        self.global_nodes_visited = 0 
        self.visits_per_nodes = {}
        self.problem_results = defaultdict()
        self.num_constraints = 0
        self.variables_count = 0
        self.num_binary_variables = 0
        self.num_continuous_variables = 0
        self.total_number_cluster = 0
        self.coordinated_plan = defaultdict(dict)
        return self.latest_run_report
        

    def build_average_coverage_diagnostics(self) -> Dict[str, Any]:
        from dummy_app.models.coverage import CoverageDiagnostics, average_coverage_diagnostics

        records = []
        for coverage_record in self.metrics.coverage_records:
            diagnostics = coverage_record.get("pathloss_diagnostics", {})
            if diagnostics:
                records.append(CoverageDiagnostics(**diagnostics))

        average_diagnostics = average_coverage_diagnostics(records)
        return average_diagnostics.to_dict() if average_diagnostics else {}
        

    def build_run_summary(self) -> Dict[str, Any]:
        depots_len = len(self.depots) if self.depots is not None else 0
        total_nodes_visited = self.global_nodes_visited - (self.total_number_cluster - depots_len)
        total_energy_consumption = 0.0
        total_mission_time = 0.0
        total_distance = 0.0
        total_service_time = 0.0
        total_travel_time = 0.0
        total_target_nodes = 0
        total_covered_nodes = 0
        nodes_per_uav = {str(agent_id): 0 for agent_id in self.agents}
        route_distance_per_uav = {str(agent_id): 0.0 for agent_id in self.agents}

        for cluster in self.problem_results.values():
            total_energy_consumption += float(cluster["Total Energy"])
            total_mission_time += float(cluster["Total Time"])
            total_distance += float(cluster["Total Distance"])
            total_target_nodes += int(cluster.get("target_node_count", 0))
            total_covered_nodes += int(cluster.get("covered_nodes", 0))
            for agent_result in cluster["agent_results"].values():
                total_service_time += float(agent_result.get("service_time", 0.0))
                total_travel_time += float(agent_result.get("travel_time", 0.0))

            cluster_nodes_per_uav = cluster.get("nodes_per_uav", {})
            for agent_id, node_count in cluster_nodes_per_uav.items():
                nodes_per_uav[str(agent_id)] = nodes_per_uav.get(str(agent_id), 0) + int(node_count)

            cluster_route_distances = cluster.get("route_distance_per_uav", {})
            for agent_id, agent_distance in cluster_route_distances.items():
                route_distance_per_uav[str(agent_id)] = route_distance_per_uav.get(str(agent_id), 0.0) + float(agent_distance)

        statuses = [record["status"] for record in self.cluster_status_records]
        feasible_flag = 1.0 if statuses and all(status in {"Optimal", "Feasible"} for status in statuses) else 0.0
        optimal_flag = 1.0 if statuses and all(status == "Optimal" for status in statuses) else 0.0
        timeout_flag = 1.0 if any(status in {"Not Solved", "Undefined"} for status in statuses) else 0.0
        node_coverage_ratio = total_nodes_visited / self.v if self.v else 0.0
        average_visits_per_node = (
            sum(self.visits_per_nodes.values()) / len(self.visits_per_nodes)
            if self.visits_per_nodes
            else 0.0
        )
        average_data_per_cluster = (
            float(self.total_data_rate) / float(self.total_number_cluster)
            if self.total_number_cluster
            else 0.0
        )
        average_makespan_per_cluster = (
            float(self.makespan) / float(self.total_number_cluster)
            if self.total_number_cluster
            else 0.0
        )
        idle_ratio = total_service_time / max(total_mission_time, 1e-6)
        coverage_diagnostics = self.build_average_coverage_diagnostics()
        node_counts = list(nodes_per_uav.values())
        plan_coverage_ratio = float(total_covered_nodes) / float(total_target_nodes) if total_target_nodes else 0.0

        if self.objective_function == "coverage":
            objective_value = float(-self.total_data_rate)
        else:
            objective_value = (
                float(self.objective_weights["energy"]) * total_energy_consumption
                + float(self.objective_weights["distance"]) * total_distance
                + float(self.objective_weights["travel_time"]) * total_mission_time
            )

        return {
            "model_name": self.model_name,
            "solver_backend": self.solver_backend,
            "subtour_mode": self.subtour_mode,
            "subtour_strategy": self.subtour_strategy,
            "objective_strategy": self.objective_strategy,
            "solve_time_seconds": float(getattr(self.metrics, "elapsed_time", 0.0) or 0.0),
            "timeout_flag": timeout_flag,
            "feasible_flag": feasible_flag,
            "optimal_flag": optimal_flag,
            "memory_usage_mb": float(self.metrics.memory_usage or 0.0),
            "objective_value": objective_value,
            "energy_cost": total_energy_consumption,
            "distance": total_distance,
            "mission_time_cost": total_mission_time,
            "travel_time_cost": total_travel_time,
            "service_time_cost": total_service_time,
            "makespan": float(self.makespan),
            "node_coverage_ratio": node_coverage_ratio,
            "total_nodes_visited": float(total_nodes_visited),
            "visits_per_node": dict(self.visits_per_nodes),
            "average_visits_per_node": average_visits_per_node,
            "idle_ratio": idle_ratio,
            "nodes_per_kwh": total_nodes_visited / max(total_energy_consumption, 1e-6),
            "nodes_per_hour": total_nodes_visited / max(total_mission_time / 60.0, 1e-6),
            "total_data_rate": float(self.total_data_rate),
            "data_rate_per_hour": float(self.total_data_rate) / max(total_mission_time / 60.0, 1e-6),
            "data_rate_per_kwh": float(self.total_data_rate) / max(total_energy_consumption, 1e-6),
            "average_data_rate_per_cluster": average_data_per_cluster,
            "average_makespan_per_cluster": average_makespan_per_cluster,
            "covered_nodes": total_covered_nodes,
            "coverage_ratio": plan_coverage_ratio,
            "num_uavs_used": sum(1 for node_count in node_counts if node_count > 0),
            "max_route_distance_per_uav": max(route_distance_per_uav.values(), default=0.0),
            "workload_imbalance": (max(node_counts) - min(node_counts)) if node_counts else 0,
            "nodes_per_uav": nodes_per_uav,
            "route_distance_per_uav": route_distance_per_uav,
            "coverage_diagnostics": coverage_diagnostics,
            "num_clusters": self.total_number_cluster,
            "largest_cluster_size": max((record["node_count"] for record in self.cluster_status_records), default=0),
            "num_constraints": self.num_constraints,
            "num_variables": self.variables_count,
            "num_binary_variables": self.num_binary_variables,
            "num_continuous_variables": self.num_continuous_variables,
            "time_limit_seconds": float(self.solver_time_limit_seconds or 0.0),
        }


    def build_failed_run_summary(self, exc: Exception) -> Dict[str, Any]:
        statuses = [record.get("raw_status", record.get("status", "Unknown")) for record in self.solve_status_history]
        timeout_flag = 1.0 if "timeout" in str(exc).lower() or any(status in {"Not Solved", "Undefined"} for status in statuses) else 0.0
        return {
            "model_name": self.model_name,
            "solver_backend": self.solver_backend,
            "subtour_mode": self.subtour_mode,
            "subtour_strategy": self.subtour_strategy,
            "solve_time_seconds": float(getattr(self.metrics, "elapsed_time", 0.0) or 0.0),
            "timeout_flag": timeout_flag,
            "feasible_flag": 0.0,
            "optimal_flag": 0.0,
            "memory_usage_mb": float(getattr(self.metrics, "memory_usage", 0.0) or 0.0),
            "objective_value": float("inf"),
            "energy_cost": 0.0,
            "distance": 0.0,
            "mission_time_cost": 0.0,
            "makespan": 0.0,
            "node_coverage_ratio": 0.0,
            "nodes_per_kwh": 0.0,
            "nodes_per_hour": 0.0,
            "total_data_rate": 0.0,
            "data_rate_per_hour": 0.0,
            "data_rate_per_kwh": 0.0,
            "covered_nodes": 0,
            "coverage_ratio": 0.0,
            "num_uavs_used": 0,
            "max_route_distance_per_uav": 0.0,
            "workload_imbalance": 0,
            "nodes_per_uav": {str(agent_id): 0 for agent_id in self.agents},
            "route_distance_per_uav": {str(agent_id): 0.0 for agent_id in self.agents},
            "num_clusters": float(self.total_number_cluster),
            "largest_cluster_size": max((record.get("node_count", 0) for record in self.cluster_status_records), default=0),
            "num_constraints": float(self.num_constraints),
            "num_variables": float(self.variables_count),
            "num_binary_variables": float(self.num_binary_variables),
            "num_continuous_variables": float(self.num_continuous_variables),
            "time_limit_seconds": float(self.solver_time_limit_seconds or 0.0),
            "error_type": exc.__class__.__name__,
        }
        


    def process_path_for_gantt(self,path_log):
        """
        Converts a detailed step-by-step log into a list of activity blocks.
        An activity block is a tuple: (label, start_time, duration).
        """
        if not path_log:
            return []

        activities = []
        # Start with the first step in the log
        current_from, current_to, start_time = path_log[0]
        
        for i in range(1, len(path_log)):
            next_from, next_to, _ = path_log[i]
            
            # If the activity changes, log the previous one and start a new one
            if (current_from, current_to) != (next_from, next_to):
                end_time = path_log[i-1][2]
                duration = (end_time - start_time) + 1
                
                # Create a label for the activity
                if current_from == current_to:
                    label = f'Wait @ N{current_from}'
                else:
                    label = f'Move {current_from}→{current_to}'
                
                activities.append((label, start_time, duration))
                
                # Start the new activity
                current_from, current_to, start_time = path_log[i]
        
        # Add the very last activity in the log
        end_time = path_log[-1][2]
        duration = (end_time - start_time) + 1
        if current_from == current_to:
            label = f'Wait @ N{current_from}'
        else:
            label = f'Move {current_from}→{current_to}'
        activities.append((label, start_time, duration))
        
        return activities
    
    def create_gantt_chart(self):
        import matplotlib.pyplot as plt 
        import matplotlib.patches as mpatches
        fig, ax = plt.subplots(figsize=(18, 12))
        paths = self.plan_with_nodes
        # Define colors for different activities
        colors = {
            'Move': 'blue',
            'Wait': 'green'
        }

        agent_lanes = list(paths.keys())
        y_positions = range(len(agent_lanes))

        min_start = float('inf')
        max_end = float('-inf')

        text_vertical_spacing = 1.85  # Increase this value for more space between text labels

        for i, agent_id in enumerate(agent_lanes):
            path = paths[agent_id]
            activities = self.process_path_for_gantt(path)
            label_y = i - 0.25  # Initial label y position
            for j, (activity_label, start, duration) in enumerate(activities):
                activity_type = activity_label.split(' ')[0] # 'Move' or 'Wait'
                color = colors.get(activity_type, 'grey') # Default to grey
                
                # Draw the horizontal bar for the activity
                ax.barh(
                    y=i,                # The lane for this agent
                    width=duration,     # The length of the bar
                    left=start,         # Where the bar starts on the time axis
                    height=0.6,
                    align='center',
                    color=color,
                    edgecolor='black'
                )
                # Add text label inside the bar, staggered vertically for readability
                ax.text(start + duration / 2, label_y, activity_label, 
                        ha='center', va='center', color='white', weight='bold', fontsize=5, clip_on=True)

                # Track min/max for axis limits
                min_start = min(min_start, start)
                max_end = max(max_end, start + duration)

                # Move label_y for next label
                label_y += text_vertical_spacing / max(1, len(activities))  # Use the spacing variable

                # Reset label_y after a long wait period (reset after, not before)
                if activity_type == 'Wait' and duration > 25:
                    label_y = i - 0.25

        ax.set_yticks(list(y_positions))
        ax.set_yticklabels(agent_lanes)
        ax.set_ylabel('Agent ID', fontsize=12)
        ax.invert_yaxis()  # Puts Agent 1 at the top

        ax.set_xlabel('Mission Time (steps)', fontsize=12)
        ax.set_title('Mission Schedule Gantt Chart', fontsize=16, weight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Set xlim to ensure the first step is fully visible
        ax.set_xlim(left=min_start - 1, right=max_end + 1)

        # Create a custom legend
        legend_patches = [mpatches.Patch(color=color, label=label) for label, color in colors.items()]
        ax.legend(handles=legend_patches, loc='upper right')

        plt.tight_layout()
        plt.show()
        plt.close()
