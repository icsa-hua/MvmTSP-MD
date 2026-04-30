from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

from dummy_app.designs.agents import generate_agent_colormap


def load_playback_artifacts(artifact_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    artifact_path = Path(artifact_dir)
    frames = pd.read_csv(artifact_path / "playback_frames.csv")
    metadata = json.loads((artifact_path / "playback_metadata.json").read_text())
    return frames, metadata


def find_latest_playback_artifact(base_dir: str | Path) -> Path:
    candidates = sorted(
        Path(base_dir).glob("runs/*/playback_frames.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No playback artifacts found under {base_dir}")
    return candidates[0].parent


class PlaybackAnimator:
    def __init__(self, playback_frames: pd.DataFrame, metadata: dict[str, Any]) -> None:
        self.playback_frames = self._normalize_playback_frames(playback_frames)
        self.metadata = dict(metadata)
        self.agents = sorted(self.playback_frames["agent"].astype(str).unique())
        self.max_time_step = int(self.metadata.get("max_time_step", self.playback_frames["time_step"].max()))
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.agent_colors, self.path_colors = generate_agent_colormap(len(self.agents))
        self.scatter = self.ax.scatter([], [], s=180, marker="^", edgecolors="black", zorder=4)
        self.path_lines = []
        self.agent_index = {agent_id: idx for idx, agent_id in enumerate(self.agents)}
        self.timeline = self._build_dense_timeline()
        self.cue_scatter = None

        for idx, agent_id in enumerate(self.agents):
            line, = self.ax.plot([], [], color=self.path_colors[idx], linestyle="--", linewidth=2.0, zorder=3, label=f"Agent {agent_id}")
            self.path_lines.append(line)

        self._draw_background()
        self._configure_axes()

    def _configure_axes(self) -> None:
        min_x = float(self.metadata.get("min_x", self.playback_frames["x"].min()))
        max_x = float(self.metadata.get("max_x", self.playback_frames["x"].max()))
        min_y = float(self.metadata.get("min_y", self.playback_frames["y"].min()))
        max_y = float(self.metadata.get("max_y", self.playback_frames["y"].max()))
        padding_x = max((max_x - min_x) * 0.05, 1e-6)
        padding_y = max((max_y - min_y) * 0.05, 1e-6)

        self.ax.set_xlim(min_x - padding_x, max_x + padding_x)
        self.ax.set_ylim(min_y - padding_y, max_y + padding_y)
        self.ax.set_title("UAV Path Playback")
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="best")

    def _draw_background(self) -> None:
        voronoi_regions = self.metadata.get("voronoi_regions", [])
        for polygon in voronoi_regions:
            if not polygon:
                continue
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            self.ax.plot(xs, ys, color="black", linewidth=1.0, alpha=0.35, zorder=1)

        cue_points = self.metadata.get("cue_points", [])
        if cue_points:
            cue_x = [point["x"] for point in cue_points]
            cue_y = [point["y"] for point in cue_points]
            self.cue_scatter = self.ax.scatter(
                cue_x,
                cue_y,
                c="green",
                s=35,
                edgecolors="black",
                alpha=0.85,
                zorder=2,
                label="Ground CUEs",
            )

    @staticmethod
    def _normalize_playback_frames(playback_frames: pd.DataFrame) -> pd.DataFrame:
        normalized = playback_frames.copy()
        if normalized.empty:
            return normalized

        normalized["agent"] = normalized["agent"].astype(str)
        normalized["time_step"] = normalized["time_step"].astype(int)
        normalized = (
            normalized.sort_values(["agent", "time_step"])
            .drop_duplicates(subset=["agent", "time_step"], keep="last")
            .reset_index(drop=True)
        )
        return normalized

    def _build_dense_timeline(self) -> dict[str, pd.DataFrame]:
        dense_timeline: dict[str, pd.DataFrame] = {}
        timeline_index = pd.Index(range(self.max_time_step + 1), name="time_step")
        for agent_id in self.agents:
            agent_frames = self.playback_frames[self.playback_frames["agent"].astype(str) == agent_id].copy()
            agent_frames = agent_frames.sort_values("time_step")
            agent_frames = agent_frames.drop_duplicates(subset=["time_step"], keep="last")
            agent_frames = agent_frames.set_index("time_step")
            dense = agent_frames.reindex(timeline_index).ffill().bfill()
            dense_timeline[agent_id] = dense.reset_index()
        return dense_timeline

    def update(self, frame: int):
        coords = []
        for agent_id in self.agents:
            agent_timeline = self.timeline[agent_id]
            current_row = agent_timeline.iloc[frame]
            coords.append((float(current_row["x"]), float(current_row["y"])))

            history = agent_timeline.iloc[: frame + 1]
            self.path_lines[self.agent_index[agent_id]].set_data(history["x"], history["y"])

        if coords:
            self.scatter.set_offsets(np.asarray(coords))
            self.scatter.set_color(self.agent_colors[: len(coords)])

        self.ax.set_title(f"UAV Path Playback | t={frame}")
        return [self.scatter, *self.path_lines]

    def build_animation(self, interval_ms: int = 100) -> FuncAnimation:
        return FuncAnimation(
            self.fig,
            self.update,
            frames=range(self.max_time_step + 1),
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )


def render_playback(
    artifact_dir: str | Path,
    save_path: str | Path | None = None,
    fps: int = 10,
    interval_ms: int = 100,
) -> str | None:
    frames, metadata = load_playback_artifacts(artifact_dir)
    animator = PlaybackAnimator(frames, metadata)
    animation = animator.build_animation(interval_ms=interval_ms)

    if save_path is not None:
        target_path = Path(save_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(target_path, writer="ffmpeg", fps=fps)
        plt.close(animator.fig)
        return str(target_path)

    plt.show()
    plt.close(animator.fig)
    return None
