import json
import os
from dataclasses import asdict, dataclass
from itertools import product
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from dummy_app.designs.voronoi_map import MapGenerator
from dummy_app.models.RL.action_catalog import SolverAction, build_action_catalog
from dummy_app.models.RL.features import extract_state_features
from dummy_app.models.RL.policies import DQNStylePolicy, OfflineLinearUCBPolicy, default_action_ids
from dummy_app.models.RL.reward import compute_reward
from dummy_app.models.simulation_builder import Builder
from dummy_app.tools.logger import logger

STATE_FEATURE_COLUMNS = [
    "num_areas",
    "num_users",
    "num_agents",
    "use_case_flag",
    "environment_urban",
    "environment_rural",
    "environment_forest",
    "environment_mountain",
    "avg_inter_node_distance",
    "std_inter_node_distance",
    "max_depot_to_node_distance",
    "avg_depot_to_node_distance",
    "distance_skewness",
    "battery_capacity",
    "avg_arc_energy_cost",
    "max_arc_energy_cost",
    "estimated_return_margin",
    "estimated_cluster_count_default",
    "largest_cluster_size_estimate",
    "cluster_balance_score",
    "bridge_node_count_estimate",
    "virtual_node_count_estimate",
    "spatial_dispersion_score",
    "estimated_num_variables",
    "estimated_num_constraints",
    "estimated_solver_difficulty_score",
    "user_density",
    "estimated_comm_priority_score",
    "previous_runtime_estimate",
    "previous_timeout_rate",
]


@dataclass(frozen=True)
class MissionInstanceSpec:
    instance_id: str
    seed: int
    num_areas: int
    num_users: int
    num_agents: int
    env_type: str
    scenario: str
    objective_function: str
    max_battery: int
    map_spread: int


def _make_user_groups(user_points: Dict[int, List[Any]]) -> Dict[int, List[Any]]:
    cue_groups: Dict[int, List[Any]] = {}
    for area_id, points in user_points.items():
        cue_groups[area_id] = [SimpleNamespace(x=point[0], y=point[1]) for point in points]
    return cue_groups


def build_instance_specs(
    area_values: Sequence[int],
    user_values: Sequence[int],
    agent_values: Sequence[int],
    env_values: Sequence[str],
    scenario_values: Sequence[str],
    spread_values: Sequence[int],
    battery_values: Sequence[int],
    objective_function: str,
    seed_count: int,
) -> List[MissionInstanceSpec]:
    specs: List[MissionInstanceSpec] = []
    spec_id = 0
    for seed, num_areas, num_users, num_agents, env_type, scenario, map_spread, max_battery in product(
        range(seed_count),
        area_values,
        user_values,
        agent_values,
        env_values,
        scenario_values,
        spread_values,
        battery_values,
    ):
        specs.append(
            MissionInstanceSpec(
                instance_id=f"instance_{spec_id}",
                seed=seed,
                num_areas=num_areas,
                num_users=num_users,
                num_agents=num_agents,
                env_type=env_type,
                scenario=scenario,
                objective_function=objective_function,
                max_battery=max_battery,
                map_spread=map_spread,
            )
        )
        spec_id += 1
    return specs


def validate_action_catalog(base_config: Dict[str, Any], trials: int) -> pd.DataFrame:
    builder = Builder(dict(base_config), trials)
    rows = []
    seen_ids = set()
    for action in build_action_catalog():
        runtime = action.to_runtime_config()
        builder.apply_runtime_configuration(runtime)
        weights = runtime["objective_weights"]
        stage_solution_valid = runtime["stage_solution"] in {1, 2}
        ga_generations_valid = runtime["ga_generations"] > 0
        time_limit_valid = runtime["time_limit_seconds"] > 0
        weights_sum_valid = abs(sum(weights.values()) - 1.0) <= 0.05
        rows.append(
            {
                "action_id": action.action_id,
                "is_unique": action.action_id not in seen_ids,
                "stage_solution_valid": stage_solution_valid,
                "ga_generations_valid": ga_generations_valid,
                "time_limit_valid": time_limit_valid,
                "weights_sum_valid": weights_sum_valid,
                "warm_start_mode": runtime["warm_start_mode"],
                "solver_mode": runtime["solver_mode"],
                "is_valid": stage_solution_valid and ga_generations_valid and time_limit_valid and weights_sum_valid,
            }
        )
        seen_ids.add(action.action_id)
    return pd.DataFrame(rows)


def generate_dataset(
    base_config: Dict[str, Any],
    trials: int,
    altitude: int,
    lat: float,
    lon: float,
    vertical_velocity: float,
    horizontal_velocity: float,
    coverage_time: int,
    output_dir: str,
    specs: Sequence[MissionInstanceSpec],
    action_ids: Optional[Sequence[str]] = None,
) -> str:
    actions = [action for action in build_action_catalog() if action_ids is None or action.action_id in action_ids]
    rows: List[Dict[str, Any]] = []

    for spec in specs:
        logger.info(f"Generating dataset rows for {spec.instance_id}")
        builder = Builder(
            {
                **dict(base_config),
                "env_type": spec.env_type,
                "max_battery": spec.max_battery,
                "scenario": spec.scenario,
                "objective_function": spec.objective_function,
                "NUMBER_OF_AGENTS": spec.num_agents,
                "NUMBER_OF_USERS": spec.num_users,
                "NUMBER_OF_AREAS": spec.num_areas,
                "altitude": altitude,
                "learning_enabled": False,
            },
            trials,
        )
        map_generator = MapGenerator(
            num_areas=spec.num_areas,
            users_per_area=spec.num_users,
            lon=lon,
            lat=lat,
            low=spec.map_spread,
            high=spec.map_spread,
            seed=spec.seed,
        )
        regions, centroids, user_points, depots, distance_matrix, _ = map_generator.create_environment(
            show_map=False,
            show_3d_map=False,
        )
        cue_groups = _make_user_groups(user_points)
        data = builder.preprocess_generated_data(
            distance_matrix=distance_matrix,
            centroids=centroids,
            depots=depots if not isinstance(depots, list) else np.array(depots),
            num_of_agents=spec.num_agents,
            v_hor=horizontal_velocity,
            v_ver=vertical_velocity,
            altitude=altitude,
            coverage_time=coverage_time,
            user_points=user_points,
        )
        state_vector, feature_map = extract_state_features(
            builder=builder,
            distance_matrix=distance_matrix,
            cue_groups=cue_groups,
            history_summary={"previous_runtime_estimate": 0.0, "previous_timeout_rate": 0.0},
        )
        instance_rows: List[Dict[str, Any]] = []
        baseline_objective: Optional[float] = None

        for action in actions:
            builder.apply_runtime_configuration(action.to_runtime_config())
            try:
                builder.run_model(distance_matrix=distance_matrix, data=data.copy(deep=True), cue_groups=cue_groups)
                run_metrics = dict(builder.latest_run_summary)
                error_message = ""
            except Exception as exc:
                run_metrics = builder.build_failed_run_summary(exc)
                error_message = str(exc)

            objective_value = float(run_metrics.get("objective_value", np.inf))
            feasible_flag = float(run_metrics.get("feasible_flag", 0.0))
            if feasible_flag > 0.0 and np.isfinite(objective_value):
                if baseline_objective is None or objective_value < baseline_objective:
                    baseline_objective = objective_value

            row = {
                "instance_id": spec.instance_id,
                "seed": spec.seed,
                "num_areas": spec.num_areas,
                "num_users": spec.num_users,
                "num_agents": spec.num_agents,
                "scenario": spec.scenario,
                "objective_function": spec.objective_function,
                "environment_type": spec.env_type,
                "map_spread": spec.map_spread,
                "max_battery": spec.max_battery,
                "state_vector": json.dumps(state_vector.tolist()),
                "error_message": error_message,
                **feature_map,
                **asdict(action),
                **run_metrics,
            }
            instance_rows.append(row)

        for row in instance_rows:
            reward, reward_terms = compute_reward(row, baseline_objective)
            row["reward"] = reward
            row["reward_terms"] = json.dumps(reward_terms, sort_keys=True)
            row["energy_budget_tightness"] = float(row.get("max_arc_energy_cost", 0.0))
            rows.append(row)

    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, "solver_dataset.csv")
    pd.DataFrame(rows).to_csv(dataset_path, index=False)
    return dataset_path


def analyze_dataset(dataset_path: str, output_dir: str) -> Dict[str, str]:
    dataset = pd.read_csv(dataset_path)
    os.makedirs(output_dir, exist_ok=True)
    feature_columns = _feature_columns(dataset)
    action_summary = dataset.groupby("action_id").agg(
        mean_reward=("reward", "mean"),
        mean_runtime=("solve_time_seconds", "mean"),
        timeout_rate=("timeout_flag", "mean"),
        feasible_rate=("feasible_flag", "mean"),
        mean_objective=("objective_value", "mean"),
    ).reset_index()
    best_rows = dataset.groupby("instance_id")["reward"].transform("max") == dataset["reward"]
    best_frequency = best_rows.groupby(dataset["action_id"]).mean().reset_index(name="best_frequency")
    action_summary = action_summary.merge(best_frequency, on="action_id", how="left")
    action_summary_path = os.path.join(output_dir, "action_summary.csv")
    action_summary.to_csv(action_summary_path, index=False)

    dominated_actions: List[str] = []
    for _, row in action_summary.iterrows():
        for _, other in action_summary.iterrows():
            if row["action_id"] == other["action_id"]:
                continue
            dominates = (
                other["mean_reward"] >= row["mean_reward"]
                and other["mean_runtime"] <= row["mean_runtime"]
                and other["timeout_rate"] <= row["timeout_rate"]
                and (
                    other["mean_reward"] > row["mean_reward"]
                    or other["mean_runtime"] < row["mean_runtime"]
                    or other["timeout_rate"] < row["timeout_rate"]
                )
            )
            if dominates:
                dominated_actions.append(str(row["action_id"]))
                break

    correlations = dataset[feature_columns].corr().abs()
    redundant_pairs = []
    for i, column in enumerate(feature_columns):
        for other in feature_columns[i + 1:]:
            corr = correlations.loc[column, other]
            if corr >= 0.95:
                redundant_pairs.append({"feature_a": column, "feature_b": other, "abs_correlation": corr})
    redundant_path = os.path.join(output_dir, "redundant_features.csv")
    pd.DataFrame(redundant_pairs).to_csv(redundant_path, index=False)

    summary_path = os.path.join(output_dir, "analysis_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "dominated_actions": sorted(set(dominated_actions)),
                "rarely_useful_actions": action_summary[action_summary["best_frequency"] <= 0.05]["action_id"].tolist(),
                "feature_columns": feature_columns,
            },
            file,
            indent=2,
            sort_keys=True,
        )

    return {
        "action_summary": action_summary_path,
        "redundant_features": redundant_path,
        "analysis_summary": summary_path,
    }


def compare_baselines(dataset_path: str, output_dir: str, alpha: float = 0.75) -> str:
    dataset = pd.read_csv(dataset_path)
    feature_columns = _feature_columns(dataset)
    instance_ids = sorted(dataset["instance_id"].unique())
    split_index = max(int(len(instance_ids) * 0.8), 1)
    train_ids = instance_ids[:split_index]
    test_ids = instance_ids[split_index:] or instance_ids[-1:]
    train = dataset[dataset["instance_id"].isin(train_ids)].copy()
    test = dataset[dataset["instance_id"].isin(test_ids)].copy()

    actions = default_action_ids()
    bandit_policy = OfflineLinearUCBPolicy(len(feature_columns), actions, alpha=alpha)
    bandit_policy.fit(train, feature_columns, epochs=2)
    dqn_policy = DQNStylePolicy(actions)
    dqn_policy.fit(train, feature_columns)

    comparisons = [
        _evaluate_fixed_policy("random_action", train, test, lambda _: np.random.choice(actions)),
        _evaluate_fixed_policy("always_default_config", train, test, lambda _: "A1"),
        _evaluate_fixed_policy("always_single_stage", train, test, lambda _: _best_fixed_action(train, solver_mode="single-stage")),
        _evaluate_fixed_policy("always_two_stage", train, test, lambda _: _best_fixed_action(train, solver_mode="two-stage")),
        _evaluate_fixed_policy("best_fixed_action", train, test, lambda _: _best_fixed_action(train)),
        _evaluate_policy("contextual_bandit", test, feature_columns, bandit_policy.predict_action),
        _evaluate_policy("dqn_style", test, feature_columns, dqn_policy.predict_action),
    ]

    os.makedirs(output_dir, exist_ok=True)
    comparison_path = os.path.join(output_dir, "baseline_comparison.csv")
    pd.DataFrame(comparisons).to_csv(comparison_path, index=False)
    return comparison_path


def _evaluate_fixed_policy(name: str, train: pd.DataFrame, test: pd.DataFrame, chooser) -> Dict[str, Any]:
    chosen_action = chooser(train)
    rows = []
    for instance_id in sorted(test["instance_id"].unique()):
        group = test[test["instance_id"] == instance_id]
        selected = group[group["action_id"] == chosen_action]
        if selected.empty:
            selected = group.iloc[[0]]
        rows.append(selected.iloc[0])
    return _aggregate_policy_rows(name, pd.DataFrame(rows))


def _evaluate_policy(name: str, dataset: pd.DataFrame, feature_columns: List[str], chooser) -> Dict[str, Any]:
    rows = []
    for instance_id in sorted(dataset["instance_id"].unique()):
        group = dataset[dataset["instance_id"] == instance_id]
        action_id = chooser(group, feature_columns)
        selected = group[group["action_id"] == action_id]
        if selected.empty:
            selected = group.iloc[[0]]
        rows.append(selected.iloc[0])
    return _aggregate_policy_rows(name, pd.DataFrame(rows))


def _aggregate_policy_rows(name: str, rows: pd.DataFrame) -> Dict[str, Any]:
    return {
        "policy_name": name,
        "instances_evaluated": int(len(rows)),
        "mean_reward": float(rows["reward"].mean()),
        "mean_runtime": float(rows["solve_time_seconds"].mean()),
        "timeout_rate": float(rows["timeout_flag"].mean()),
        "feasible_rate": float(rows["feasible_flag"].mean()),
        "mean_objective": float(rows["objective_value"].mean()),
    }


def _best_fixed_action(dataset: pd.DataFrame, solver_mode: Optional[str] = None) -> str:
    subset = dataset if solver_mode is None else dataset[dataset["solver_mode"] == solver_mode]
    if subset.empty:
        return "A1"
    return str(subset.groupby("action_id")["reward"].mean().idxmax())


def _feature_columns(dataset: pd.DataFrame) -> List[str]:
    return [column for column in STATE_FEATURE_COLUMNS if column in dataset.columns]
