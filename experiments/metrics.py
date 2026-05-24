from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def _compress_agent_path(agent_path: Sequence[Sequence[Any]]) -> List[Tuple[int, int, int]]:
    if not agent_path:
        return []

    compressed: List[Tuple[int, int, int]] = []
    current_source, current_target, _ = agent_path[0]
    current_length = 1
    for source, target, _ in agent_path[1:]:
        if int(source) == int(current_source) and int(target) == int(current_target):
            current_length += 1
            continue
        compressed.append((int(current_source), int(current_target), int(current_length)))
        current_source, current_target = source, target
        current_length = 1
    compressed.append((int(current_source), int(current_target), int(current_length)))
    return compressed


def aggregate_agent_routes(cluster_results: Sequence[Any]) -> Dict[int, List[Tuple[int, int, int]]]:
    aggregated: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for cluster_result in cluster_results:
        for agent_id, agent_path in dict(cluster_result.agent_paths).items():
            aggregated[int(agent_id)].extend((int(source), int(target), int(timestep)) for source, target, timestep in agent_path)
    return dict(aggregated)


def compute_total_distance(routes: Mapping[int, Sequence[Sequence[Any]]], distance_matrix: np.ndarray) -> float:
    return float(sum(compute_route_distance_per_uav(routes, distance_matrix).values()))


def compute_total_energy(
    routes: Mapping[int, Sequence[Sequence[Any]]],
    energy_matrix: np.ndarray,
    coverage_energy_per_step: float = 0.0,
) -> float:
    return float(sum(compute_route_energy_per_uav(routes, energy_matrix, coverage_energy_per_step=coverage_energy_per_step).values()))


def compute_total_travel_time(routes: Mapping[int, Sequence[Sequence[Any]]], time_matrix: np.ndarray) -> float:
    total_travel_time = 0.0
    for agent_path in routes.values():
        for source, target, _segment_length in _compress_agent_path(agent_path):
            if int(source) == int(target):
                continue
            total_travel_time += float(time_matrix[int(source)][int(target)])
    return float(total_travel_time)


def compute_route_distance_per_uav(
    routes: Mapping[int, Sequence[Sequence[Any]]],
    distance_matrix: np.ndarray,
) -> Dict[str, float]:
    per_uav: Dict[str, float] = {}
    for agent_id, agent_path in routes.items():
        total_distance = 0.0
        for source, target, _segment_length in _compress_agent_path(agent_path):
            if int(source) == int(target):
                continue
            total_distance += float(distance_matrix[int(source)][int(target)])
        per_uav[str(agent_id)] = float(total_distance)
    return per_uav


def compute_route_energy_per_uav(
    routes: Mapping[int, Sequence[Sequence[Any]]],
    energy_matrix: np.ndarray,
    coverage_energy_per_step: float = 0.0,
) -> Dict[str, float]:
    per_uav: Dict[str, float] = {}
    for agent_id, agent_path in routes.items():
        total_energy = 0.0
        for source, target, segment_length in _compress_agent_path(agent_path):
            if int(source) == int(target):
                total_energy += float(coverage_energy_per_step) * float(segment_length)
                continue
            total_energy += float(energy_matrix[int(source)][int(target)])
        per_uav[str(agent_id)] = float(total_energy)
    return per_uav


def compute_std(values: Iterable[float]) -> float:
    series = [float(value) for value in values]
    if not series:
        return 0.0
    return float(np.std(series))


def compute_jain_fairness(values: Iterable[float]) -> float:
    series = np.array([max(float(value), 0.0) for value in values], dtype=float)
    if series.size == 0:
        return 0.0
    denominator = float(series.size) * float(np.square(series).sum())
    if denominator <= 1e-12:
        return 0.0
    numerator = float(series.sum()) ** 2
    return float(numerator / denominator)


def compute_coverage_ratio(routes: Mapping[int, Sequence[Sequence[Any]]], all_nodes: Iterable[int]) -> float:
    target_nodes = {int(node_id) for node_id in all_nodes}
    if not target_nodes:
        return 0.0

    visited_nodes = set()
    for agent_path in routes.values():
        for _source, target, _ in _compress_agent_path(agent_path):
            target_int = int(target)
            if target_int in target_nodes:
                visited_nodes.add(target_int)
    return float(len(visited_nodes)) / float(len(target_nodes))


def count_unvisited_nodes(routes: Mapping[int, Sequence[Sequence[Any]]], all_nodes: Iterable[int]) -> int:
    target_nodes = {int(node_id) for node_id in all_nodes}
    visited_nodes = set()
    for agent_path in routes.values():
        for _source, target, _ in _compress_agent_path(agent_path):
            target_int = int(target)
            if target_int in target_nodes:
                visited_nodes.add(target_int)
    return int(len(target_nodes - visited_nodes))


def format_json_mapping(mapping: Mapping[str, Any]) -> str:
    return json.dumps(dict(mapping), sort_keys=True)
