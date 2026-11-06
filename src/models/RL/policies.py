from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from dummy_app.models.RL.action_catalog import build_action_catalog


class OfflineLinearUCBPolicy:
    def __init__(self, feature_dim: int, action_ids: Sequence[str], alpha: float = 0.75) -> None:
        self.feature_dim = feature_dim
        self.action_ids = list(action_ids)
        self.alpha = alpha
        self.A = {action_id: np.eye(feature_dim, dtype=np.float64) for action_id in self.action_ids}
        self.b = {action_id: np.zeros(feature_dim, dtype=np.float64) for action_id in self.action_ids}

    def fit(self, dataset: pd.DataFrame, feature_columns: List[str], epochs: int = 1) -> None:
        ordered_instances = sorted(dataset["instance_id"].unique())
        for _ in range(max(epochs, 1)):
            for instance_id in ordered_instances:
                group = dataset[dataset["instance_id"] == instance_id]
                if group.empty:
                    continue
                action_id = self.predict_action(group, feature_columns)
                reward = float(group.loc[group["action_id"] == action_id, "reward"].iloc[0])
                context = group.iloc[0][feature_columns].to_numpy(dtype=np.float64)
                self.A[action_id] = self.A[action_id] + np.outer(context, context)
                self.b[action_id] = self.b[action_id] + reward * context

    def predict_action(self, group: pd.DataFrame, feature_columns: List[str]) -> str:
        context = group.iloc[0][feature_columns].to_numpy(dtype=np.float64)
        available_actions = group["action_id"].tolist()
        scores: Dict[str, float] = {}
        for action_id in available_actions:
            A_inv = np.linalg.inv(self.A[action_id])
            theta = A_inv @ self.b[action_id]
            scores[action_id] = float(theta @ context + self.alpha * np.sqrt(context @ A_inv @ context))
        return max(scores, key=scores.get)


class DQNStylePolicy:
    def __init__(self, action_ids: Sequence[str]) -> None:
        self.action_ids = list(action_ids)
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            random_state=42,
            max_iter=500,
        )
        self._is_fit = False

    def fit(self, dataset: pd.DataFrame, feature_columns: List[str]) -> None:
        features = dataset.groupby("instance_id")[feature_columns].first().sort_index()
        rewards = dataset.pivot_table(
            index="instance_id",
            columns="action_id",
            values="reward",
            aggfunc="first",
        ).sort_index()
        rewards = rewards.reindex(columns=self.action_ids)
        rewards = rewards.fillna(rewards.min().min() if not rewards.empty else 0.0)
        if features.empty or rewards.empty:
            return
        self.model.fit(features.to_numpy(dtype=np.float64), rewards.to_numpy(dtype=np.float64))
        self._is_fit = True

    def predict_action(self, group: pd.DataFrame, feature_columns: List[str]) -> str:
        available_actions = group["action_id"].tolist()
        if not self._is_fit:
            return available_actions[0]
        context = group.iloc[0][feature_columns].to_numpy(dtype=np.float64).reshape(1, -1)
        predictions = self.model.predict(context)[0]
        scores = {
            action_id: float(predictions[self.action_ids.index(action_id)])
            for action_id in available_actions
        }
        return max(scores, key=scores.get)


def default_action_ids() -> List[str]:
    return [action.action_id for action in build_action_catalog()]
