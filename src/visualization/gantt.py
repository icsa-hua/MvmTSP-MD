from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


_SEGMENT_COLORS = {
    "move": "#2f6df6",
    "service": "#2a9d55",
    "recharge": "#f4a261",
    "standby": "#8d99ae",
}


@dataclass
class GanttSegment:
    agent_id: str
    cluster_id: int | None
    segment_type: str
    label: str
    start_step: int
    end_step: int

    @property
    def duration(self) -> int:
        return int(self.end_step - self.start_step + 1)


def _normalize_mapping(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if is_dataclass(payload):
        payload = asdict(payload)
    if isinstance(payload, Mapping):
        return {str(key): value for key, value in payload.items()}
    return dict(payload)


def _normalize_cluster_results(cluster_results: Sequence[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for cluster_result in cluster_results:
        if is_dataclass(cluster_result):
            normalized.append(asdict(cluster_result))
        elif isinstance(cluster_result, Mapping):
            normalized.append(dict(cluster_result))
        else:
            normalized.append(dict(cluster_result))
    return normalized


def _as_int_time(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)


def _step_time(step: Sequence[Any]) -> int:
    return int(step[-1])


def _step_source_target(step: Sequence[Any]) -> tuple[int, int]:
    return int(step[0]), int(step[1])


def _metric_local_end_exclusive(metrics: Mapping[str, Any]) -> int:
    start_step = _as_int_time(metrics.get("scheduled_start_time", 0.0))
    finish_step = _as_int_time(metrics.get("scheduled_finish_time", 0.0))
    next_available_step = _as_int_time(metrics.get("next_available_time", 0.0))
    recharge_steps = max(_as_int_time(metrics.get("recharge_steps", 0.0)), 0)

    if next_available_step >= start_step:
        return max(next_available_step - start_step, 0)
    if finish_step >= start_step:
        return max(finish_step - start_step + recharge_steps, 0)
    if next_available_step > 0:
        return max(next_available_step, 0)
    return max(finish_step + recharge_steps, 0)


def _cluster_local_duration(
    agent_metrics: Mapping[str, Mapping[str, Any]],
    agent_paths: Mapping[str, Sequence[Sequence[Any]]],
) -> int:
    metric_durations = [
        _metric_local_end_exclusive(metrics)
        for metrics in agent_metrics.values()
        if metrics
    ]
    if metric_durations:
        return max(metric_durations)
    return max(
        (
            _step_time(step) + 1
            for path in agent_paths.values()
            for step in path
        ),
        default=0,
    )


def _trim_path_to_local_duration(
    path: Sequence[Sequence[Any]],
    local_duration: int,
) -> List[tuple[Any, ...]]:
    normalized_path = [tuple(step) for step in path]
    if local_duration <= 0:
        return normalized_path
    return [step for step in normalized_path if _step_time(step) < int(local_duration)]


def _infer_final_depot_wait(
    local_path: Sequence[Sequence[Any]],
) -> tuple[int | None, int | None]:
    last_wait_index: int | None = None
    depot_node: int | None = None
    for index in range(len(local_path) - 1, -1, -1):
        source_node, target_node = _step_source_target(local_path[index])
        if source_node == target_node:
            last_wait_index = index
            depot_node = source_node
            break

    if last_wait_index is None or depot_node is None:
        if local_path:
            return int(local_path[-1][1]), None
        return None, None

    wait_start_index = last_wait_index
    while wait_start_index > 0:
        previous_source, previous_target = _step_source_target(local_path[wait_start_index - 1])
        current_time = _step_time(local_path[wait_start_index])
        previous_time = _step_time(local_path[wait_start_index - 1])
        if (
            previous_source == previous_target == depot_node
            and previous_time + 1 == current_time
        ):
            wait_start_index -= 1
            continue
        break

    return int(depot_node), _step_time(local_path[wait_start_index])


def build_cluster_windows(
    cluster_results: Sequence[Any],
    plan_with_nodes: Mapping[Any, Sequence[Sequence[Any]]] | None = None,
) -> List[Dict[str, int]]:
    if plan_with_nodes is not None:
        assignments = build_agent_assignments(plan_with_nodes, cluster_results)
        windows_by_cluster: Dict[int, Dict[str, int]] = {}
        for agent_windows in assignments.values():
            for agent_window in agent_windows:
                cluster_id = int(agent_window["cluster_id"])
                start_step = int(agent_window["start_step"])
                end_step = int(agent_window["segment_end_step"])
                if cluster_id not in windows_by_cluster:
                    windows_by_cluster[cluster_id] = {
                        "cluster_id": cluster_id,
                        "start_step": start_step,
                        "end_step": end_step,
                    }
                else:
                    windows_by_cluster[cluster_id]["start_step"] = min(
                        windows_by_cluster[cluster_id]["start_step"],
                        start_step,
                    )
                    windows_by_cluster[cluster_id]["end_step"] = max(
                        windows_by_cluster[cluster_id]["end_step"],
                        end_step,
                    )
        return sorted(windows_by_cluster.values(), key=lambda window: (window["start_step"], window["cluster_id"]))

    windows: List[Dict[str, int]] = []
    for cluster_result in _normalize_cluster_results(cluster_results):
        agent_metrics = {
            str(agent_id): dict(metrics)
            for agent_id, metrics in _normalize_mapping(cluster_result.get("agent_metrics")).items()
        }
        agent_paths = _normalize_mapping(cluster_result.get("agent_paths"))
        if not agent_metrics:
            continue
        absolute_starts = [_as_int_time(metrics.get("scheduled_start_time", 0.0)) for metrics in agent_metrics.values()]
        absolute_ends = []
        local_duration = _cluster_local_duration(agent_metrics, agent_paths)
        for metrics in agent_metrics.values():
            start_step = _as_int_time(metrics.get("scheduled_start_time", 0.0))
            next_available_step = _as_int_time(metrics.get("next_available_time", 0.0))
            finish_step = _as_int_time(metrics.get("scheduled_finish_time", 0.0))
            recharge_steps = max(_as_int_time(metrics.get("recharge_steps", 0.0)), 0)
            if next_available_step >= start_step:
                absolute_ends.append(max(next_available_step - 1, start_step))
            elif finish_step >= start_step:
                absolute_ends.append(max(finish_step + recharge_steps - 1, start_step))
            else:
                absolute_ends.append(start_step + max(local_duration - 1, 0))
        if not absolute_starts or not absolute_ends:
            continue
        windows.append(
            {
                "cluster_id": int(cluster_result.get("cluster_id", -1)),
                "start_step": int(min(absolute_starts)),
                "end_step": int(max(absolute_ends)),
            }
        )
    return sorted(windows, key=lambda window: (window["start_step"], window["cluster_id"]))


def build_agent_assignments(
    plan_with_nodes: Mapping[Any, Sequence[Sequence[Any]]],
    cluster_results: Sequence[Any],
) -> Dict[str, List[Dict[str, int]]]:
    global_paths_by_agent = {
        str(agent_id): [tuple(step) for step in raw_path]
        for agent_id, raw_path in plan_with_nodes.items()
    }
    assignments: Dict[str, List[Dict[str, int]]] = {}
    global_index_by_agent: Dict[str, int] = {agent_id: 0 for agent_id in global_paths_by_agent}

    for cluster_result in _normalize_cluster_results(cluster_results):
        cluster_id = int(cluster_result.get("cluster_id", -1))
        agent_metrics = {
            str(agent_id): dict(metrics)
            for agent_id, metrics in _normalize_mapping(cluster_result.get("agent_metrics")).items()
        }
        raw_agent_paths = {
            str(agent_id): [tuple(step) for step in path]
            for agent_id, path in _normalize_mapping(cluster_result.get("agent_paths")).items()
            if path
        }
        local_duration = _cluster_local_duration(agent_metrics, raw_agent_paths)
        agent_paths = {
            agent_id: _trim_path_to_local_duration(path, local_duration)
            for agent_id, path in raw_agent_paths.items()
        }
        for agent_id in sorted(set(agent_metrics) | set(agent_paths), key=int):
            metrics = agent_metrics.get(agent_id, {})
            local_path = agent_paths.get(agent_id, [])
            depot_node, depot_wait_start_local = _infer_final_depot_wait(local_path)
            global_path = global_paths_by_agent.get(agent_id, [])
            global_index = int(global_index_by_agent.get(agent_id, 0))

            if local_path and global_index < len(global_path):
                path_length = len(local_path)
                global_slice = global_path[global_index : global_index + path_length]
                if global_slice:
                    start_step = _step_time(global_slice[0])
                    segment_end_step = _step_time(global_slice[-1])
                    local_start_step = _step_time(local_path[0])
                    global_offset = start_step - local_start_step
                else:
                    start_step = _as_int_time(metrics.get("scheduled_start_time", 0.0))
                    segment_end_step = start_step
                    global_offset = start_step
                global_index_by_agent[agent_id] = global_index + path_length
            else:
                start_step = _as_int_time(metrics.get("scheduled_start_time", 0.0))
                next_available_step = _as_int_time(metrics.get("next_available_time", 0.0))
                recharge_steps = max(_as_int_time(metrics.get("recharge_steps", 0.0)), 0)
                finish_step = _as_int_time(metrics.get("scheduled_finish_time", 0.0))
                if next_available_step >= start_step:
                    segment_end_step = max(next_available_step - 1, start_step)
                else:
                    segment_end_step = start_step + max(finish_step + recharge_steps - 1, 0)
                global_offset = start_step

            recharge_steps = max(_as_int_time(metrics.get("recharge_steps", 0.0)), 0)
            own_local_duration = _metric_local_end_exclusive(metrics) if metrics else local_duration
            if depot_wait_start_local is not None:
                recharge_start_local = int(depot_wait_start_local)
            else:
                recharge_start_local = max(own_local_duration - recharge_steps, 0)
            recharge_start_step = int(recharge_start_local + global_offset)

            recharge_end_step = recharge_start_step + recharge_steps - 1
            assignments.setdefault(agent_id, []).append(
                {
                    "cluster_id": cluster_id,
                    "depot_node": int(depot_node) if depot_node is not None else -1,
                    "start_step": int(start_step),
                    "active_end_step": int(min(recharge_start_step - 1, segment_end_step)),
                    "recharge_start_step": int(recharge_start_step),
                    "recharge_end_step": int(min(recharge_end_step, segment_end_step)),
                    "segment_end_step": int(segment_end_step),
                }
            )

    return assignments


def _lookup_assignment(
    agent_id: str,
    time_step: int,
    assignments: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Mapping[str, Any] | None:
    for window in assignments.get(agent_id, []):
        if int(window["start_step"]) <= int(time_step) <= int(window["segment_end_step"]):
            return window
    return None


def _lookup_cluster_id(
    agent_id: str,
    time_step: int,
    assignments: Mapping[str, Sequence[Mapping[str, Any]]],
) -> int | None:
    for window in assignments.get(agent_id, []):
        if int(window["start_step"]) <= int(time_step) <= int(window["segment_end_step"]):
            return int(window["cluster_id"])
    return None


def build_gantt_segments(
    plan_with_nodes: Mapping[Any, Sequence[Sequence[Any]]],
    cluster_results: Sequence[Any],
    *,
    max_service_steps: int = 3,
) -> List[GanttSegment]:
    assignments = build_agent_assignments(plan_with_nodes, cluster_results)
    segments: List[GanttSegment] = []

    for raw_agent_id, raw_path in sorted(plan_with_nodes.items(), key=lambda item: int(item[0])):
        agent_id = str(raw_agent_id)
        path = [tuple(step) for step in raw_path]
        if not path:
            continue

        current_segment: GanttSegment | None = None
        for source_node, target_node, time_step in path:
            assignment = _lookup_assignment(agent_id, int(time_step), assignments)
            cluster_id = _lookup_cluster_id(agent_id, int(time_step), assignments)
            if cluster_id is None and assignment is not None and assignment.get("cluster_id") is not None:
                cluster_id = int(assignment["cluster_id"])
            depot_node = int(assignment.get("depot_node", -1)) if assignment is not None else None

            if int(source_node) != int(target_node):
                segment_type = "move"
                label = f"Move N{int(source_node)}->N{int(target_node)}"
            else:
                stationary_node = int(source_node)
                if depot_node is not None and stationary_node == depot_node:
                    recharge_start_step = int(assignment.get("recharge_start_step", 1 << 60)) if assignment else 1 << 60
                    recharge_end_step = int(assignment.get("recharge_end_step", -1)) if assignment else -1
                    if recharge_start_step <= int(time_step) <= recharge_end_step:
                        segment_type = "recharge"
                        label = f"Recharge @ N{stationary_node}"
                    else:
                        segment_type = "standby"
                        label = f"Standby @ N{stationary_node}"
                else:
                    segment_type = "service"
                    label = f"Service @ N{stationary_node}"

            service_chunk_exhausted = (
                current_segment is not None
                and current_segment.segment_type == "service"
                and current_segment.duration >= int(max_service_steps)
            )
            merge_recharge_across_cluster_boundary = (
                current_segment is not None
                and current_segment.segment_type == "recharge"
                and segment_type == "recharge"
                and current_segment.label == label
                and current_segment.end_step + 1 == int(time_step)
            )

            if (
                current_segment is not None
                and current_segment.segment_type == segment_type
                and current_segment.label == label
                and current_segment.cluster_id == cluster_id
                and current_segment.end_step + 1 == int(time_step)
                and not service_chunk_exhausted
            ):
                current_segment.end_step = int(time_step)
                continue

            if merge_recharge_across_cluster_boundary:
                current_segment.end_step = int(time_step)
                continue

            if current_segment is not None:
                segments.append(current_segment)
            current_segment = GanttSegment(
                agent_id=agent_id,
                cluster_id=cluster_id,
                segment_type=segment_type,
                label=label,
                start_step=int(time_step),
                end_step=int(time_step),
            )

        if current_segment is not None:
            segments.append(current_segment)

    return segments


def gantt_segments_to_frame(segments: Iterable[GanttSegment]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "agent_id": segment.agent_id,
                "cluster_id": segment.cluster_id,
                "segment_type": segment.segment_type,
                "label": segment.label,
                "start_step": segment.start_step,
                "end_step": segment.end_step,
                "duration": segment.duration,
            }
            for segment in segments
        ]
    )


def render_mission_schedule_gantt(
    segments: Sequence[GanttSegment],
    output_path: str | Path,
    *,
    title: str = "Mission Schedule Gantt Chart",
    cluster_windows: Sequence[Mapping[str, int]] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not segments:
        raise ValueError("No Gantt segments were provided.")

    agent_ids = sorted({segment.agent_id for segment in segments}, key=int)
    max_end_step = max(segment.end_step for segment in segments)
    fig_width = min(max(18.0, max_end_step / 7.5), 28.0)
    fig_height = max(5.0, 1.4 * len(agent_ids) + 2.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    lane_positions = {agent_id: index for index, agent_id in enumerate(agent_ids)}
    visible_cluster_ids = {segment.cluster_id for segment in segments if segment.cluster_id is not None}

    if cluster_windows:
        for index, window in enumerate(cluster_windows):
            if int(window["cluster_id"]) not in visible_cluster_ids:
                continue
            if index % 2 == 0:
                ax.axvspan(
                    float(window["start_step"]) - 0.5,
                    float(window["end_step"]) + 0.5,
                    color="#dbeafe",
                    alpha=0.18,
                    lw=0,
                )
            midpoint = (float(window["start_step"]) + float(window["end_step"])) / 2.0
            ax.text(
                midpoint,
                -0.8,
                f"Cluster {int(window['cluster_id'])}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#334155",
                weight="bold",
            )
            ax.axvline(float(window["end_step"]) + 0.5, color="#cbd5e1", linestyle="--", linewidth=0.8, alpha=0.8)

    for segment in segments:
        y_position = lane_positions[segment.agent_id]
        color = _SEGMENT_COLORS.get(segment.segment_type, "#64748b")
        ax.barh(
            y=y_position,
            width=segment.duration,
            left=segment.start_step,
            height=0.62,
            color=color,
            edgecolor="#0f172a",
            linewidth=0.8,
        )

        if segment.duration >= 4:
            ax.text(
                segment.start_step + segment.duration / 2.0,
                y_position,
                segment.label,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                weight="bold",
                clip_on=True,
            )
        elif segment.segment_type == "move":
            ax.text(
                segment.start_step + 0.15,
                y_position - 0.34,
                segment.label.replace("Move ", ""),
                ha="left",
                va="center",
                fontsize=6,
                color="#1e293b",
                rotation=32,
            )

    ax.set_xlim(-1, max_end_step + 2)
    ax.set_yticks([lane_positions[agent_id] for agent_id in agent_ids])
    ax.set_yticklabels([f"Agent {agent_id}" for agent_id in agent_ids], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Mission Time (steps)", fontsize=12)
    ax.set_ylabel("Allocated Agent", fontsize=12)
    ax.set_title(title, fontsize=18, weight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    legend_handles = [
        mpatches.Patch(color=_SEGMENT_COLORS[segment_type], label=label)
        for segment_type, label in (
            ("move", "Move"),
            ("service", "Service"),
            ("recharge", "Recharge"),
            ("standby", "Standby"),
        )
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output
