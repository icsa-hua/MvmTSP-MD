from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mvmtsp-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mvmtsp-xdg-cache")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduce.execution_script import ALTITUDE, build_problem_context
from dummy_app.visualization import (
    build_cluster_windows,
    build_gantt_segments,
    gantt_segments_to_frame,
    render_mission_schedule_gantt,
)

DEFAULT_OUTPUT = REPO_ROOT / "assets" / "RESULTS_IMAGES" / "mission_schedule_gantt_50areas_5agents.png"
DEFAULT_SCHEDULE_OUTPUT = REPO_ROOT / "assets" / "results" / "mission_schedule_gantt_50areas_5agents.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight mission instance and render an agent-allocation Gantt chart.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PNG output path.")
    parser.add_argument("--schedule-output", type=Path, default=DEFAULT_SCHEDULE_OUTPUT, help="CSV schedule output path.")
    parser.add_argument("--scenario", type=str, default="cooperative", help="Scenario type.")
    parser.add_argument("--objective", type=str, default="energy", help="Objective function.")
    parser.add_argument("--model-name", type=str, default="milp", help="Runtime/model name to execute.")
    parser.add_argument("--num-areas", type=int, default=50, help="Number of areas to generate.")
    parser.add_argument("--num-agents", type=int, default=5, help="Number of agents to allocate.")
    parser.add_argument("--num-users", type=int, default=3, help="Users per area.")
    parser.add_argument("--max-battery", type=int, default=362, help="Maximum battery capacity.")
    parser.add_argument("--max-coverage-time", type=int, default=3, help="Coverage time per serviced node.")
    parser.add_argument("--env", type=str, default="urban", help="Environment type.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--trials", type=int, default=1, help="Number of simulator trials.")
    parser.add_argument("--solver-backend", type=str, default="cplex", help="MILP backend when applicable.")
    parser.add_argument("--subtour-mode", type=str, default="mtz", help="Subtour elimination mode.")
    return parser.parse_args()


def build_args(namespace: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        show_map=False,
        scenario=namespace.scenario,
        objective=namespace.objective,
        model_name=namespace.model_name,
        mode="solve",
        artifact_dir="",
        save_animation=False,
        animation_output="",
        animation_fps=10,
        animation_interval_ms=100,
        enable_ga="no",
        warm_start_mode="",
        num_agents=namespace.num_agents,
        num_users=namespace.num_users,
        max_battery=namespace.max_battery,
        max_coverage_time=namespace.max_coverage_time,
        num_areas=namespace.num_areas,
        env=namespace.env,
        stage_solution=1,
        priority="yes",
        validate=False,
        trials=namespace.trials,
        solver_backend=namespace.solver_backend,
        subtour_mode=namespace.subtour_mode,
        objective_strategy="legacy_stage",
        scenario_constraint_set="default",
        solver_time_limit_seconds=0,
        seed=namespace.seed,
        enable_learning=False,
        learning_alpha=0.2,
        workflow="simulate",
        dataset_output_dir="",
        dataset_path="",
        dataset_seed_count=1,
        dataset_area_values="",
        dataset_user_values="",
        dataset_agent_values="",
        dataset_env_values="",
        dataset_scenario_values="",
        dataset_spread_values="",
        dataset_action_ids="",
    )


def main() -> None:
    cli_args = parse_args()
    runtime_args = build_args(cli_args)

    problem, mobility_sim, ground_users, data, distance_matrix, _, _ = build_problem_context(runtime_args)
    constructor = mobility_sim.run_headless(
        constructor=problem,
        cues=ground_users,
        distance_matrix=distance_matrix,
        data=data,
        altitude=ALTITUDE,
        trials=runtime_args.trials,
    )

    if not constructor.plan_with_nodes:
        raise RuntimeError("The runtime did not produce any synchronized plan data for the Gantt chart.")
    if constructor.latest_model_run_result is None:
        raise RuntimeError("The runtime did not retain a model result for schedule extraction.")

    cluster_results = constructor.latest_model_run_result.cluster_results
    cluster_windows = build_cluster_windows(cluster_results, plan_with_nodes=constructor.plan_with_nodes)
    segments = build_gantt_segments(constructor.plan_with_nodes, cluster_results)
    output_path = render_mission_schedule_gantt(
        segments,
        cli_args.output,
        title=(
            "Mission Schedule Gantt Chart "
            f"({cli_args.num_areas} areas, {cli_args.num_agents} agents, {cli_args.model_name})"
        ),
        cluster_windows=cluster_windows,
    )

    schedule_frame = gantt_segments_to_frame(segments)
    cli_args.schedule_output.parent.mkdir(parents=True, exist_ok=True)
    schedule_frame.to_csv(cli_args.schedule_output, index=False)

    print(f"chart={output_path}")
    print(f"schedule={cli_args.schedule_output}")
    print(f"artifact_dir={constructor.latest_artifact_dir}")


if __name__ == "__main__":
    main()
