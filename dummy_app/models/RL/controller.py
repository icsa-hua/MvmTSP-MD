import csv
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from dummy_app.models.RL.action_catalog import SolverAction, build_action_catalog
from dummy_app.models.RL.agent import LinearUCBBandit, load_json, save_json
from dummy_app.models.RL.features import extract_state_features
from dummy_app.models.RL.reward import compute_reward
from dummy_app.tools.logger import logger


class RLController:
    def __init__(self, base_dir: str, alpha: float = 0.75) -> None:
        self.base_dir = base_dir
        self.actions: List[SolverAction] = build_action_catalog()
        self.action_lookup = {action.action_id: action for action in self.actions}
        self.bandit: Optional[LinearUCBBandit] = None
        self.pending_transition: Optional[Dict[str, Any]] = None
        self.history_path = os.path.join(base_dir, "transitions.csv")
        self.baseline_path = os.path.join(base_dir, "baselines.json")
        self.summary_path = os.path.join(base_dir, "history_summary.json")
        self.model_path = os.path.join(base_dir, "bandit_state.npz")
        self.alpha = alpha

    def _ensure_bandit(self, feature_dim: int) -> None:
        if self.bandit is None:
            self.bandit = LinearUCBBandit(
                action_ids=[action.action_id for action in self.actions],
                feature_dim=feature_dim,
                alpha=self.alpha,
                state_path=self.model_path,
            )

    def _instance_key(self, builder: Any) -> str:
        return "|".join(
            [
                builder.scenario,
                builder.env_type,
                str(builder.NUMBER_OF_AREAS),
                str(builder.NUMBER_OF_AGENTS),
                str(builder.NUMBER_OF_USERS),
                builder.objective_function,
            ]
        )

    def _history_summary(self, builder: Any) -> Dict[str, float]:
        summary = load_json(self.summary_path, {})
        return summary.get(self._instance_key(builder), {"previous_runtime_estimate": 0.0, "previous_timeout_rate": 0.0})

    def start_episode(self, builder: Any, distance_matrix, data, cue_groups) -> Dict[str, Any]:
        feature_vector, feature_map = extract_state_features(
            builder=builder,
            distance_matrix=distance_matrix,
            cue_groups=cue_groups,
            history_summary=self._history_summary(builder),
        )
        self._ensure_bandit(feature_vector.shape[0])
        action_id, scores = self.bandit.select_action(feature_vector)
        action = self.action_lookup[action_id]
        runtime_config = action.to_runtime_config()
        builder.apply_runtime_configuration(runtime_config)
        self.pending_transition = {
            "instance_key": self._instance_key(builder),
            "feature_vector": feature_vector,
            "feature_map": feature_map,
            "action": action,
            "scores": scores,
        }
        logger.info(f"Learning agent selected {action.action_id} with runtime config {runtime_config}")
        return runtime_config

    def finish_episode(self, builder: Any, run_metrics: Dict[str, Any]) -> None:
        if self.pending_transition is None or self.bandit is None:
            return

        baselines = load_json(self.baseline_path, {})
        instance_key = self.pending_transition["instance_key"]
        baseline_objective = baselines.get(instance_key)
        reward, reward_terms = compute_reward(run_metrics, baseline_objective)
        action: SolverAction = self.pending_transition["action"]
        context = self.pending_transition["feature_vector"]
        self.bandit.update(action.action_id, context, reward)
        self.bandit.save()

        current_objective = float(run_metrics.get("objective_value", 0.0))
        if baseline_objective is None or current_objective < float(baseline_objective):
            baselines[instance_key] = current_objective
            save_json(self.baseline_path, baselines)

        self._update_history_summary(instance_key, run_metrics)
        self._append_transition(
            {
                "instance_key": instance_key,
                "action_id": action.action_id,
                "reward": reward,
                "solve_time_seconds": run_metrics.get("solve_time_seconds"),
                "timeout_flag": run_metrics.get("timeout_flag"),
                "feasible_flag": run_metrics.get("feasible_flag"),
                "optimal_flag": run_metrics.get("optimal_flag"),
                "objective_value": run_metrics.get("objective_value"),
                "node_coverage_ratio": run_metrics.get("node_coverage_ratio"),
                "time_limit_seconds": run_metrics.get("time_limit_seconds"),
                "feature_map": self.pending_transition["feature_map"],
                "reward_terms": reward_terms,
                "action_config": asdict(action),
            }
        )
        builder.latest_learning_result = {
            "action_id": action.action_id,
            "reward": reward,
            "reward_terms": reward_terms,
            "run_metrics": run_metrics,
        }
        self.pending_transition = None

    def _append_transition(self, row: Dict[str, Any]) -> None:
        directory = os.path.dirname(self.history_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        file_exists = os.path.exists(self.history_path)
        field_names = list(row.keys())
        with open(self.history_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=field_names)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _update_history_summary(self, instance_key: str, run_metrics: Dict[str, Any]) -> None:
        summary = load_json(self.summary_path, {})
        current = summary.get(instance_key, {"count": 0, "runtime_total": 0.0, "timeout_total": 0.0})
        current["count"] += 1
        current["runtime_total"] += float(run_metrics.get("solve_time_seconds", 0.0))
        current["timeout_total"] += float(run_metrics.get("timeout_flag", 0.0))
        count = max(current["count"], 1)
        summary[instance_key] = {
            "count": current["count"],
            "runtime_total": current["runtime_total"],
            "timeout_total": current["timeout_total"],
            "previous_runtime_estimate": current["runtime_total"] / count,
            "previous_timeout_rate": current["timeout_total"] / count,
        }
        save_json(self.summary_path, summary)
