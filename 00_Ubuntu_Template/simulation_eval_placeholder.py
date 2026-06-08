from __future__ import annotations

from pathlib import Path
from typing import Any


def simulation_evaluation_placeholder(
    *,
    checkpoint_path: str | Path,
    dataset_name: str | None = None,
    algorithm: str | None = None,
    lidar_beams: int | None = None,
) -> dict[str, Any]:
    """Return stable report fields until the simulator rollout evaluator exists."""
    _ = checkpoint_path, dataset_name, algorithm, lidar_beams
    return {
        "simulation_eval_status": "not_implemented",
        "simulation_return_mean": "",
        "simulation_normalized_return": "",
        "simulation_lap_completion_rate": "",
        "simulation_collision_rate": "",
        "simulation_episode_seconds_mean": "",
        "simulation_notes": (
            "Placeholder only. Offline action metrics rank behavior matching; deploy safety "
            "still needs simulator rollouts for lap completion, wall avoidance, and recovery."
        ),
    }
