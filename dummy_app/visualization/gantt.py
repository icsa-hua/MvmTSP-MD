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


def _build_cluster_depot_lookup(cluster_results: Sequence[Any]) -> Dict[int, int]:
    depot_by_cluster: Dict[int, int] = {}
    for cluster_result in _normalize_cluster_results(cluster_results):
        cluster_id = int(cluster_result.get("cluster_id", -1))
        agent_paths = _normalize_mapping(cluster_result.get("agent_paths"))
        for path in agent_paths.values():
            if not path:
                continue
            first_step = list(path)[0]
            depot_by_cluster[cluster_id] = int(first_step[0])
            break
    return depot_by_cluster


def build_cluster_windows(cluster_results: Sequence[Any]) -> List[Dict[str, int]]:
    windows: List[Dict[str, int]] = []
    for cluster_result in _normalize_cluster_results(cluster_results):
        agent_metrics = {
            str(agent_id): dict(metrics)
            for agent_id, metrics in _normalize_mapping(cluster_result.get("agent_metrics")).items()
        }
        if not agent_metrics:
            continue
        absolute_starts = [
            int(round(float(metrics.get("scheduled_start_time", 0.0) or 0.0)))
            for metrics in agent_metrics.values()
        ]
        absolute_ends = [
            int(round(float(metrics.get("scheduled_start_time", 0.0) or 0.0)))
            + int(round(float(metrics.get("next_available_time", 0.0) or 0.0)))
            - 1
            for metrics in agent_metrics.values()
        ]
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
    path_end_by_agent = {
        str(agent_id): int(max((int(step[2]) for step in raw_path), default=-1))
        for agent_id, raw_path in plan_with_nodes.items()
    }
    depot_by_cluster = _build_cluster_depot_lookup(cluster_results)
    assignments: Dict[str, List[Dict[str, int]]] = {}

    for cluster_result in _normalize_cluster_results(cluster_results):
        cluster_id = int(cluster_result.get("cluster_id", -1))
        agent_metrics = {
            str(agent_id): dict(metrics)
            for agent_id, metrics in _normalize_mapping(cluster_result.get("agent_metrics")).items()
        }
        for agent_id, metrics in agent_metrics.items():
            scheduled_start_step = int(round(float(metrics.get("scheduled_start_time", 0.0) or 0.0)))
            scheduled_finish_step = int(round(float(metrics.get("scheduled_finish_time", 0.0) or 0.0)))
            assignments.setdefault(agent_id, []).append(
                {
                    "cluster_id": cluster_id,
                    "depot_node": int(depot_by_cluster.get(cluster_id, -1)),
                    "start_step": scheduled_start_step,
                    "active_end_step": scheduled_start_step + max(scheduled_finish_step - 1, 0),
                    "recharge_start_step": scheduled_start_step + scheduled_finish_step,
                    "segment_end_step": -1,
                }
            )

    for agent_id, windows in assignments.items():
        windows.sort(key=lambda window: (window["start_step"], window["cluster_id"]))
        path_end_step = int(path_end_by_agent.get(agent_id, -1))
        for index, window in enumerate(windows):
            if index + 1 < len(windows):
                window["segment_end_step"] = int(windows[index + 1]["start_step"] - 1)
            else:
                window["segment_end_step"] = path_end_step

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
    cluster_windows = build_cluster_windows(cluster_results)
    segments: List[GanttSegment] = []

    for raw_agent_id, raw_path in sorted(plan_with_nodes.items(), key=lambda item: int(item[0])):
        agent_id = str(raw_agent_id)
        path = [tuple(step) for step in raw_path]
        if not path:
            continue

        current_segment: GanttSegment | None = None
        recharge_active = False
        recharge_depot_node: int | None = None
        previous_step: tuple[int, int, int] | None = None

        for source_node, target_node, time_step in path:
            assignment = _lookup_assignment(agent_id, int(time_step), assignments)
            cluster_id = _lookup_cluster_id(agent_id, int(time_step), assignments)
            if cluster_id is None and assignment is not None and assignment.get("cluster_id") is not None:
                cluster_id = int(assignment["cluster_id"])
            depot_node = int(assignment.get("depot_node", -1)) if assignment is not None else None

            if int(source_node) != int(target_node):
                segment_type = "move"
                label = f"Move N{int(source_node)}->N{int(target_node)}"
                recharge_active = False
                recharge_depot_node = None
            else:
                stationary_node = int(source_node)
                returned_to_depot = (
                    previous_step is not None
                    and int(previous_step[0]) != int(previous_step[1])
                    and int(previous_step[1]) == stationary_node
                )
                departed_from_depot = (
                    previous_step is not None
                    and int(previous_step[0]) == stationary_node
                    and int(previous_step[1]) != stationary_node
                )
                if departed_from_depot:
                    recharge_active = False
                    recharge_depot_node = None

                if depot_node is not None and stationary_node == depot_node:
                    if returned_to_depot:
                        recharge_active = True
                        recharge_depot_node = stationary_node
                    if recharge_active and recharge_depot_node == stationary_node:
                        segment_type = "recharge"
                        label = f"Recharge @ N{stationary_node}"
                    else:
                        segment_type = "standby"
                        label = f"Standby @ N{stationary_node}"
                else:
                    segment_type = "service"
                    label = f"Service @ N{stationary_node}"
                    recharge_active = False
                    recharge_depot_node = None

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
                previous_step = (int(source_node), int(target_node), int(time_step))
                continue

            if merge_recharge_across_cluster_boundary:
                current_segment.end_step = int(time_step)
                previous_step = (int(source_node), int(target_node), int(time_step))
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
            previous_step = (int(source_node), int(target_node), int(time_step))

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
