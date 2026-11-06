from __future__ import annotations

import gc
from typing import Any, Dict, List

import pandas as pd

from dummy_app.core.schemas.instance import PreparedClusterInput, ProblemInstance
from dummy_app.tools.logger import logger


def build_problem_instance(runtime: Any, distance_matrix, data: pd.DataFrame, cue_groups: Dict[int, List[Any]]) -> ProblemInstance:
    data_wo_depots, depots = runtime.separate_depots_from_clusters(data)
    logger.debug("Prepared instance: depots separated from candidate cluster nodes.")

    gdf = runtime.createGeoDataset(data_wo_depots)
    clusters = runtime.regionalization(gdf)
    priority = runtime.cluster_prioritization(clusters, cue_groups, distance_matrix)
    if runtime.priority != "yes":
        priority["Rank"] = priority["Rank"].apply(lambda _: 1)

    cluster_with_depots, same_depot_agents = runtime.assign_depot_to_cluster(
        clusters,
        depots,
        distance_matrix=distance_matrix,
    )
    assignments = runtime.allocate_agents_to_clusters(cluster_with_depots, priority, same_depot_agents)

    cluster_groups = list(clusters)
    priority_rank_by_cluster = priority["Rank"].to_dict() if "Rank" in priority else {}

    prepared_clusters: List[PreparedClusterInput] = []
    for (cluster_id, depot_id), agents in assignments.items():
        for grouped_cluster_id, cluster_frame in cluster_groups:
            if grouped_cluster_id != cluster_id:
                continue
            enriched_cluster = runtime.add_depot_data_to_cluster((grouped_cluster_id, cluster_frame), depots, depot_id)
            prepared_clusters.append(
                PreparedClusterInput(
                    cluster_id=int(cluster_id),
                    cluster_frame=enriched_cluster,
                    assigned_agents=list(agents),
                    depot_id=int(depot_id),
                    priority_rank=float(priority_rank_by_cluster.get(cluster_id, 0.0)),
                    metadata={"cluster_label": int(grouped_cluster_id)},
                )
            )
            break

    del data_wo_depots
    del gdf
    del clusters
    del cluster_groups
    del cluster_with_depots
    del same_depot_agents
    del depots
    gc.collect()

    return ProblemInstance(
        instance_id=str(runtime.id),
        model_name=str(getattr(runtime, "model_name", "milp")),
        scenario=runtime.scenario,
        objective_function=runtime.objective_function,
        stage_solution=int(runtime.stage_solution),
        environment_type=runtime.env_type,
        number_of_agents=int(runtime.NUMBER_OF_AGENTS),
        number_of_areas=int(runtime.NUMBER_OF_AREAS),
        number_of_users=int(runtime.NUMBER_OF_USERS),
        depots=[int(depot) for depot in runtime.depots] if runtime.depots is not None else [],
        user_points=cue_groups,
        prepared_data=data,
        distance_matrix=distance_matrix,
        raw_cost_matrices=dict(runtime.problem_cost_data),
        normalized_cost_columns={
            "distance": list(runtime.distance_columns),
            "energy": list(runtime.energy_columns),
            "travel_time": list(runtime.travel_time_columns),
        },
        objective_weights=dict(runtime.objective_weights),
        clustering_feature_weights=dict(runtime.clustering_feature_weights),
        prepared_clusters=prepared_clusters,
        metadata={
            "estimated_cluster_capacity": int(getattr(runtime, "estimated_cluster_capacity", 1)),
            "priority_enabled": runtime.priority == "yes",
        },
    )
