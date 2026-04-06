import json
import os
from typing import Dict, List, Tuple

import numpy as np


class LinearUCBBandit:
    def __init__(self, action_ids: List[str], feature_dim: int, alpha: float, state_path: str) -> None:
        self.action_ids = action_ids
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.state_path = state_path
        self.A: Dict[str, np.ndarray] = {}
        self.b: Dict[str, np.ndarray] = {}
        self.counts: Dict[str, int] = {}
        self._load()

    def _initialize_state(self) -> None:
        self.A = {action_id: np.eye(self.feature_dim, dtype=np.float64) for action_id in self.action_ids}
        self.b = {action_id: np.zeros(self.feature_dim, dtype=np.float64) for action_id in self.action_ids}
        self.counts = {action_id: 0 for action_id in self.action_ids}

    def _load(self) -> None:
        if not os.path.exists(self.state_path):
            self._initialize_state()
            return
        with np.load(self.state_path, allow_pickle=False) as data:
            self.A = {}
            self.b = {}
            self.counts = {}
            for action_id in self.action_ids:
                self.A[action_id] = data[f"A_{action_id}"]
                self.b[action_id] = data[f"b_{action_id}"]
                self.counts[action_id] = int(data[f"count_{action_id}"][0])

    def save(self) -> None:
        directory = os.path.dirname(self.state_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        payload: Dict[str, np.ndarray] = {}
        for action_id in self.action_ids:
            payload[f"A_{action_id}"] = self.A[action_id]
            payload[f"b_{action_id}"] = self.b[action_id]
            payload[f"count_{action_id}"] = np.asarray([self.counts[action_id]], dtype=np.int64)
        np.savez(self.state_path, **payload)

    def select_action(self, context: np.ndarray) -> Tuple[str, Dict[str, float]]:
        scores: Dict[str, float] = {}
        for action_id in self.action_ids:
            A_inv = np.linalg.inv(self.A[action_id])
            theta = A_inv @ self.b[action_id]
            exploit = float(theta @ context)
            explore = float(self.alpha * np.sqrt(context @ A_inv @ context))
            scores[action_id] = exploit + explore
        best_action = max(scores, key=scores.get)
        return best_action, scores

    def update(self, action_id: str, context: np.ndarray, reward: float) -> None:
        outer = np.outer(context, context)
        self.A[action_id] = self.A[action_id] + outer
        self.b[action_id] = self.b[action_id] + reward * context
        self.counts[action_id] = self.counts.get(action_id, 0) + 1


def load_json(path: str, default: Dict) -> Dict:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str, payload: Dict) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
