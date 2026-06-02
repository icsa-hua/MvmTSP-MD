from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from dummy_app.experiments.runner_utils import create_scenario, format_result_row, run_method, save_results
from dummy_app.program_config import (
    EXPERIMENT_C_GRID,
    EXPERIMENT_C_WARM_STARTS,
    EXPERIMENT_DEFAULT_ENV,
    EXPERIMENT_DEFAULT_MEMORY_LIMIT,
    EXPERIMENT_DEFAULT_OBJECTIVE,
    EXPERIMENT_DEFAULT_SCENARIO,
    EXPERIMENT_DEFAULT_SEEDS,
    EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
    EXPERIMENT_RESULTS_DIR,
)
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment C: MILP warm-start comparison.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(EXPERIMENT_RESULTS_DIR) / "experiment_C_warm_start.csv"),
    )
    parser.add_argument("--append", action="store_true", help="Append to the output CSV instead of replacing it.")
    return parser.parse_args()


def _warm_start_objective(result: dict) -> float | None:
    run_result = result.get("run_result")
    if run_result is None:
        return None
    objective_values = []
    for cluster_result in run_result.cluster_results:
        warm_start_summary = dict(cluster_result.diagnostics).get("warm_start_summary", {})
        objective_value = warm_start_summary.get("objective_value")
        if objective_value is not None:
            objective_values.append(float(objective_value))
    if not objective_values:
        return None
    return float(sum(objective_values))


def _warm_start_generation_sec(result: dict) -> float:
    run_result = result.get("run_result")
    if run_result is None:
        return 0.0
    return float(
        sum(
            float(dict(cluster_result.diagnostics).get("warm_start_summary", {}).get("generation_time_sec", 0.0) or 0.0)
            for cluster_result in run_result.cluster_results
        )
    )


def _warm_start_acceptance(result: dict) -> dict:
    run_result = result.get("run_result")
    if run_result is None:
        return {
            "warm_start_attempted_clusters": 0,
            "warm_start_accepted_clusters": 0,
            "warm_start_accepted_by_solver": None,
            "warm_start_acceptance_source": "",
        }

    attempted_clusters = 0
    accepted_clusters = 0
    acceptance_sources = set()
    for cluster_result in run_result.cluster_results:
        warm_start_summary = dict(cluster_result.diagnostics).get("warm_start_summary", {})
        if not warm_start_summary.get("provided_to_solver"):
            continue
        attempted_clusters += 1
        if warm_start_summary.get("accepted_by_solver") is True:
            accepted_clusters += 1
        source = str(warm_start_summary.get("acceptance_source", "") or "").strip()
        if source:
            acceptance_sources.add(source)

    accepted_by_solver = None
    if attempted_clusters > 0:
        accepted_by_solver = accepted_clusters == attempted_clusters

    return {
        "warm_start_attempted_clusters": attempted_clusters,
        "warm_start_accepted_clusters": accepted_clusters,
        "warm_start_accepted_by_solver": accepted_by_solver,
        "warm_start_acceptance_source": ",".join(sorted(acceptance_sources)),
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.append:
        output_path.unlink()

    total_runs = (
        len(EXPERIMENT_DEFAULT_SEEDS)
        * len(EXPERIMENT_C_GRID["areas"])
        * len(EXPERIMENT_C_GRID["uavs"])
        * len(EXPERIMENT_C_WARM_STARTS)
    )

    with tqdm(total=total_runs, desc="Experiment C", unit="run", dynamic_ncols=True) as progress:
        for seed in EXPERIMENT_DEFAULT_SEEDS:
            for node_count, uav_count in product(
                EXPERIMENT_C_GRID["areas"],
                EXPERIMENT_C_GRID["uavs"],
            ):
                scenario_payload = create_scenario(
                    node_count=node_count,
                    uav_count=uav_count,
                    coverage_time_profile="medium",
                    seed=seed,
                    scenario_name=EXPERIMENT_DEFAULT_SCENARIO,
                    objective_function=EXPERIMENT_DEFAULT_OBJECTIVE,
                    env_type=EXPERIMENT_DEFAULT_ENV,
                )
                scenario_results = {}
                for warm_start_method in EXPERIMENT_C_WARM_STARTS:
                    progress.set_postfix_str(
                        f"seed={seed} n={node_count} k={uav_count} | {warm_start_method}",
                        refresh=False,
                    )
                    label = f"MILP + {warm_start_method.upper()}" if warm_start_method != "none" else "MILP"
                    result = run_method(
                        scenario_payload,
                        "MILP",
                        model_name="milp",
                        warm_start_mode=warm_start_method,
                        time_limit_seconds=EXPERIMENT_DEFAULT_TIME_LIMIT_SECONDS,
                        memory_limit_bytes=EXPERIMENT_DEFAULT_MEMORY_LIMIT,
                        stage_solution=1,
                    )
                    scenario_results[warm_start_method] = result
                    baseline_metrics = scenario_results.get("none", {}).get("metrics", {})
                    current_metrics = result.get("metrics", {})
                    warm_start_generation_sec = _warm_start_generation_sec(result)
                    runtime_including_warm_start_sec = float(current_metrics.get("runtime_sec") or 0.0)
                    acceptance = _warm_start_acceptance(result)
                    baseline_gap = baseline_metrics.get("optimality_gap_percent")
                    baseline_objective = baseline_metrics.get("objective_value")
                    current_gap = current_metrics.get("optimality_gap_percent")
                    current_objective = current_metrics.get("objective_value")
                    row = format_result_row(
                        scenario_payload,
                        "MILP",
                        status=result["status"],
                        extra_fields={
                            "warm_start_method": warm_start_method,
                            "warm_start_objective": _warm_start_objective(result),
                            **acceptance,
                            "time_to_first_feasible_sec": current_metrics.get("time_to_first_feasible_sec"),
                            "final_objective_value": current_objective,
                            "best_bound": current_metrics.get("best_bound"),
                            "final_optimality_gap_percent": current_gap,
                            "runtime_sec": max(runtime_including_warm_start_sec - warm_start_generation_sec, 0.0),
                            "runtime_including_warm_start_sec": runtime_including_warm_start_sec,
                            "warm_start_generation_sec": warm_start_generation_sec,
                            "branch_and_bound_nodes": current_metrics.get("branch_and_bound_nodes"),
                            "gap_reduction_vs_no_warm_start": 0.0
                            if warm_start_method == "none" or baseline_gap is None or current_gap is None
                            else float(baseline_gap) - float(current_gap),
                            "objective_improvement_vs_no_warm_start": 0.0
                            if warm_start_method == "none" or baseline_objective is None or current_objective is None
                            else float(baseline_objective) - float(current_objective),
                            "artifact_dir": result.get("artifact_dir", ""),
                            "error_message": result.get("error_message", ""),
                        },
                    )
                    save_results(output_path, [row])
                    progress.update(1)


if __name__ == "__main__":
    main()
