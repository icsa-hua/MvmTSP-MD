from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dummy_app.core.schemas.instance import ProblemInstance
from dummy_app.core.schemas.request import ModelRunRequest
from dummy_app.core.schemas.result import ModelRunResult


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default, sort_keys=True)
    tmp_path.replace(path)


def _compact_instance_payload(instance: ProblemInstance) -> dict[str, Any]:
    payload = asdict(instance)
    payload["distance_matrix"] = []
    payload["user_points"] = []
    payload["prepared_clusters"] = []
    payload["raw_cost_matrices"] = {}
    return payload


def _flatten_path_rows(run_result: ModelRunResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster_result in run_result.cluster_results:
        for agent_id, path in cluster_result.agent_paths.items():
            for source, target, timestep in path:
                rows.append(
                    {
                        "cluster_id": cluster_result.cluster_id,
                        "agent_id": agent_id,
                        "source_node": source,
                        "target_node": target,
                        "time_step": timestep,
                        "model_name": run_result.model_name,
                        "normalized_status": cluster_result.normalized_status,
                    }
                )
    return rows


def _cluster_metric_rows(run_result: ModelRunResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster_result in run_result.cluster_results:
        row = {
            "cluster_id": cluster_result.cluster_id,
            "raw_status": cluster_result.raw_status,
            "normalized_status": cluster_result.normalized_status,
            "objective_value": cluster_result.objective_value,
            "incumbent_value": cluster_result.incumbent_value,
            "best_bound": cluster_result.best_bound,
            "absolute_gap": cluster_result.absolute_gap,
            "relative_gap": cluster_result.relative_gap,
            "elapsed_time_seconds": cluster_result.elapsed_time_seconds,
            "time_limit_seconds": cluster_result.time_limit_seconds,
            "termination_reason": cluster_result.termination_reason,
        }
        row.update(cluster_result.cluster_metrics)
        rows.append(row)
    return rows


def persist_run_artifacts(
    base_dir: str | Path,
    instance: ProblemInstance,
    request: ModelRunRequest,
    result: ModelRunResult,
) -> Path:
    artifact_dir = Path(base_dir) / "runs" / result.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    _write_json(artifact_dir / "instance.json", _compact_instance_payload(instance))
    _write_json(artifact_dir / "run_request.json", asdict(request))
    _write_json(artifact_dir / "run_result.json", asdict(result))

    path_rows = _flatten_path_rows(result)
    pd.DataFrame(path_rows).to_csv(artifact_dir / "paths.csv", index=False)

    cluster_metric_rows = _cluster_metric_rows(result)
    pd.DataFrame(cluster_metric_rows).to_csv(artifact_dir / "cluster_metrics.csv", index=False)

    return artifact_dir
