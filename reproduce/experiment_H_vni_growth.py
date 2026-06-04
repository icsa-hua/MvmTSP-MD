from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping

from tqdm import tqdm

from dummy_app.experiments.runner_utils import create_scenario, format_result_row, run_method, save_results
from dummy_app.program_config import (
    EXPERIMENT_DEFAULT_ENV,
    EXPERIMENT_DEFAULT_MEMORY_LIMIT,
    EXPERIMENT_DEFAULT_OBJECTIVE,
    EXPERIMENT_DEFAULT_SCENARIO,
    EXPERIMENT_DEFAULT_SEEDS,
    EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
    EXPERIMENT_H_BASELINE,
    EXPERIMENT_H_REQUIRED_VISITS,
    EXPERIMENT_RESULTS_DIR,
)


EMPTY_FIELDS = {
    "solver_backend": "",
    "subtour_mode": "",
    "bridge_node_required_visits": None,
    "expanded_nodes_after_vni": None,
    "original_expanded_node_count": None,
    "bridge_nodes_removed": None,
    "bridge_node_count": None,
    "virtual_node_count": None,
    "num_variables": None,
    "num_binary_variables": None,
    "num_continuous_variables": None,
    "num_constraints": None,
    "runtime_sec": None,
    "model_build_time_sec": None,
    "solver_runtime_sec": None,
    "objective_value": None,
    "best_bound": None,
    "optimality_gap_percent": None,
    "coverage_ratio": None,
    "branch_and_bound_nodes": None,
    "artifact_dir": "",
    "error_message": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment H: VNI growth benchmark.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(EXPERIMENT_RESULTS_DIR) / "experiment_H_vni_growth.csv"),
    )
    parser.add_argument("--append", action="store_true", help="Append to the output CSV instead of replacing it.")
    return parser.parse_args()


def _aggregate_vni_metrics(run_result: Any) -> Dict[str, int | None]:
    if run_result is None:
        return {
            "expanded_nodes_after_vni": None,
            "original_expanded_node_count": None,
            "bridge_nodes_removed": None,
            "bridge_node_count": None,
            "virtual_node_count": None,
        }

    expanded_nodes_after_vni = 0
    original_expanded_node_count = 0
    bridge_nodes_removed = 0
    bridge_node_count = 0
    virtual_node_count = 0

    for cluster_result in getattr(run_result, "cluster_results", []):
        cluster_metrics = dict(getattr(cluster_result, "cluster_metrics", {}))
        diagnostics = dict(getattr(cluster_result, "diagnostics", {}))
        virtual_nodes = {
            int(node_id): int(original_node)
            for node_id, original_node in dict(diagnostics.get("virtual_nodes", {})).items()
        }
        removed_original_bridges = len({int(node_id) for node_id in virtual_nodes.values()})
        cluster_node_count = int(cluster_metrics.get("node_count", 0) or 0)

        original_expanded_node_count += cluster_node_count
        bridge_nodes_removed += removed_original_bridges
        bridge_node_count += len({int(node_id) for node_id in diagnostics.get("bridge_nodes", [])})
        virtual_node_count += len(virtual_nodes)
        expanded_nodes_after_vni += max(cluster_node_count - removed_original_bridges, 0)

    return {
        "expanded_nodes_after_vni": int(expanded_nodes_after_vni),
        "original_expanded_node_count": int(original_expanded_node_count),
        "bridge_nodes_removed": int(bridge_nodes_removed),
        "bridge_node_count": int(bridge_node_count),
        "virtual_node_count": int(virtual_node_count),
    }


def _build_row(
    scenario_payload: Mapping[str, Any],
    required_visits: int,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics = dict(EMPTY_FIELDS)
    metrics.update(result.get("metrics", {}))
    metrics.update(_aggregate_vni_metrics(result.get("run_result")))
    metrics["solver_backend"] = str(result.get("solver_backend", ""))
    metrics["subtour_mode"] = str(result.get("subtour_mode", metrics.get("subtour_mode", "")))
    metrics["bridge_node_required_visits"] = int(required_visits)
    metrics["artifact_dir"] = result.get("artifact_dir", "")
    metrics["error_message"] = result.get("error_message", "")
    return format_result_row(
        scenario_payload,
        "MILP",
        status=str(result.get("status", "failed")),
        extra_fields=metrics,
    )


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.append:
        output_path.unlink()

    required_visit_values = [int(value) for value in EXPERIMENT_H_REQUIRED_VISITS]
    total_runs = len(EXPERIMENT_DEFAULT_SEEDS) * len(required_visit_values)

    with tqdm(total=total_runs, desc="Experiment H", unit="run", dynamic_ncols=True) as progress:
        for seed in EXPERIMENT_DEFAULT_SEEDS:
            scenario_payload = create_scenario(
                node_count=int(EXPERIMENT_H_BASELINE["areas"]),
                users_per_area=int(EXPERIMENT_H_BASELINE["users_per_area"]),
                uav_count=int(EXPERIMENT_H_BASELINE["uavs"]),
                coverage_time_profile=str(EXPERIMENT_H_BASELINE["coverage_time_profile"]),
                seed=int(seed),
                scenario_name=EXPERIMENT_DEFAULT_SCENARIO,
                objective_function=EXPERIMENT_DEFAULT_OBJECTIVE,
                env_type=EXPERIMENT_DEFAULT_ENV,
            )

            rows: List[Dict[str, Any]] = []
            for required_visits in required_visit_values:
                progress.set_postfix_str(
                    f"seed={seed} n={scenario_payload['node_count']} u={scenario_payload['users_per_area']} "
                    f"k={scenario_payload['uav_count']} | required_visits={required_visits}",
                    refresh=False,
                )
                result = run_method(
                    scenario_payload,
                    "MILP",
                    model_name="milp",
                    bridge_node_required_visits_override=int(required_visits),
                    time_limit_seconds=EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
                    memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                    stage_solution=1,
                )
                rows.append(_build_row(scenario_payload, int(required_visits), result))
                progress.update(1)

            save_results(output_path, rows)


if __name__ == "__main__":
    main()
