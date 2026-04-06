import math
from typing import Any, Dict, List, Tuple

import numpy as np


ENVIRONMENT_FLAGS: Dict[str, List[float]] = {
    "urban": [1.0, 0.0, 0.0, 0.0],
    "rural": [0.0, 1.0, 0.0, 0.0],
    "forest": [0.0, 0.0, 1.0, 0.0],
    "mountain": [0.0, 0.0, 0.0, 1.0],
}


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-9:
        return 0.0
    return numerator / denominator


def _normalized_counts(builder: Any) -> Tuple[float, float, float]:
    return (
        builder.NUMBER_OF_AREAS / 100.0,
        builder.NUMBER_OF_USERS / max(builder.NUMBER_OF_AREAS, 1),
        builder.NUMBER_OF_AGENTS / 10.0,
    )


def extract_state_features(
    builder: Any,
    distance_matrix: np.ndarray,
    cue_groups: Dict[int, List[Any]],
    history_summary: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, float]]:
    mask = ~np.eye(distance_matrix.shape[0], dtype=bool)
    pairwise = distance_matrix[mask]
    pairwise = pairwise if pairwise.size else np.array([0.0], dtype=np.float32)
    depots = getattr(builder, "depots", None)
    depot_distances = []
    if depots is not None:
        for depot in depots:
            depot_idx = int(depot)
            if depot_idx < distance_matrix.shape[0]:
                depot_distances.extend(distance_matrix[depot_idx].tolist())
    depot_distances_array = np.asarray(depot_distances if depot_distances else [0.0], dtype=np.float32)
    energy_matrix = getattr(builder, "move_energy", np.zeros_like(distance_matrix, dtype=np.float32))
    energy_mask = ~np.eye(energy_matrix.shape[0], dtype=bool)
    arc_energy = energy_matrix[energy_mask] if energy_matrix.size else np.array([0.0], dtype=np.float32)

    avg_pairwise = float(np.mean(pairwise))
    std_pairwise = float(np.std(pairwise))
    max_pairwise = float(np.max(pairwise))
    avg_depot_distance = float(np.mean(depot_distances_array))
    max_depot_distance = float(np.max(depot_distances_array))
    avg_energy = float(np.mean(arc_energy))
    max_energy = float(np.max(arc_energy))
    num_areas_norm, user_density, num_agents_norm = _normalized_counts(builder)
    max_nodes = max(getattr(builder, "estimated_cluster_capacity", builder.NUMBER_OF_AREAS), 1)
    depot_count = len(depots) if depots is not None else 0
    estimated_clusters = float(math.ceil(max(builder.NUMBER_OF_AREAS - depot_count, 1) / max_nodes))
    largest_cluster_estimate = float(math.ceil(max(builder.NUMBER_OF_AREAS - depot_count, 1) / max(estimated_clusters, 1.0)))
    bridge_node_estimate = float(max(builder.NUMBER_OF_AGENTS - 1, 0))
    virtual_node_estimate = bridge_node_estimate
    estimated_num_variables = float(builder.NUMBER_OF_AGENTS * max(builder.NUMBER_OF_AREAS, 1) ** 2)
    estimated_num_constraints = float(builder.NUMBER_OF_AGENTS * max(builder.NUMBER_OF_AREAS, 1) * 6)
    estimated_solver_difficulty = _safe_ratio(estimated_num_variables + estimated_num_constraints, 100000.0)
    env_vector = ENVIRONMENT_FLAGS.get(builder.env_type, [0.0, 0.0, 0.0, 0.0])
    use_case_flag = 1.0 if builder.scenario == "cooperative" else 0.0
    comm_priority = user_density * (1.0 + env_vector[0] * 0.2 + env_vector[2] * 0.1)
    return_margin = _safe_ratio(builder.max_battery - max_energy, max(builder.max_battery, 1))
    cluster_balance_score = _safe_ratio(largest_cluster_estimate, max(builder.NUMBER_OF_AREAS, 1))
    spatial_dispersion = _safe_ratio(std_pairwise, max(avg_pairwise, 1e-6))
    distance_skewness_proxy = _safe_ratio(max_pairwise - avg_pairwise, max(std_pairwise, 1e-6))

    feature_map = {
        "num_areas": num_areas_norm,
        "num_users": user_density,
        "num_agents": num_agents_norm,
        "use_case_flag": use_case_flag,
        "environment_urban": env_vector[0],
        "environment_rural": env_vector[1],
        "environment_forest": env_vector[2],
        "environment_mountain": env_vector[3],
        "avg_inter_node_distance": _safe_ratio(avg_pairwise, max(max_pairwise, 1e-6)),
        "std_inter_node_distance": _safe_ratio(std_pairwise, max(max_pairwise, 1e-6)),
        "max_depot_to_node_distance": _safe_ratio(max_depot_distance, max(max_pairwise, 1e-6)),
        "avg_depot_to_node_distance": _safe_ratio(avg_depot_distance, max(max_pairwise, 1e-6)),
        "distance_skewness": distance_skewness_proxy,
        "battery_capacity": _safe_ratio(builder.max_battery, 1000.0),
        "avg_arc_energy_cost": _safe_ratio(avg_energy, max(max_energy, 1e-6)),
        "max_arc_energy_cost": _safe_ratio(max_energy, max(builder.max_battery, 1.0)),
        "estimated_return_margin": return_margin,
        "estimated_cluster_count_default": _safe_ratio(estimated_clusters, 20.0),
        "largest_cluster_size_estimate": _safe_ratio(largest_cluster_estimate, max(builder.NUMBER_OF_AREAS, 1)),
        "cluster_balance_score": cluster_balance_score,
        "bridge_node_count_estimate": _safe_ratio(bridge_node_estimate, 10.0),
        "virtual_node_count_estimate": _safe_ratio(virtual_node_estimate, 10.0),
        "spatial_dispersion_score": spatial_dispersion,
        "estimated_num_variables": _safe_ratio(estimated_num_variables, 100000.0),
        "estimated_num_constraints": _safe_ratio(estimated_num_constraints, 100000.0),
        "estimated_solver_difficulty_score": estimated_solver_difficulty,
        "user_density": user_density,
        "estimated_comm_priority_score": comm_priority,
        "previous_runtime_estimate": history_summary.get("previous_runtime_estimate", 0.0),
        "previous_timeout_rate": history_summary.get("previous_timeout_rate", 0.0),
    }

    ordered = [
        feature_map["num_areas"],
        feature_map["num_users"],
        feature_map["num_agents"],
        feature_map["use_case_flag"],
        *env_vector,
        feature_map["avg_inter_node_distance"],
        feature_map["std_inter_node_distance"],
        feature_map["max_depot_to_node_distance"],
        feature_map["avg_depot_to_node_distance"],
        feature_map["distance_skewness"],
        feature_map["battery_capacity"],
        feature_map["avg_arc_energy_cost"],
        feature_map["max_arc_energy_cost"],
        feature_map["estimated_return_margin"],
        feature_map["estimated_cluster_count_default"],
        feature_map["largest_cluster_size_estimate"],
        feature_map["cluster_balance_score"],
        feature_map["bridge_node_count_estimate"],
        feature_map["virtual_node_count_estimate"],
        feature_map["spatial_dispersion_score"],
        feature_map["estimated_num_variables"],
        feature_map["estimated_num_constraints"],
        feature_map["estimated_solver_difficulty_score"],
        feature_map["user_density"],
        feature_map["estimated_comm_priority_score"],
        feature_map["previous_runtime_estimate"],
        feature_map["previous_timeout_rate"],
    ]
    return np.asarray(ordered, dtype=np.float64), feature_map
