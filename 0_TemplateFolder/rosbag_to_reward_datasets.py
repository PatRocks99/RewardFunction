# %%
"""
Notebook-style ROS2 bag processing for real RoboRacer data.

This script mirrors the ROS decoding style from OldDataProcessing/full_data_processing.ipynb:

- reads ROS2 bag folders with rosbags
- repairs non-UTF8 metadata.yaml files
- registers ackermann_msgs types when the local Python environment does not know them
- decodes AckermannDriveStamped, LaserScan, Odometry, and optional RGB/depth images
- aligns streams by timestamp using closest-previous data
- filters real-world sensor noise before building state/action/reward transitions

Outputs are compressed NPZ rollout datasets compatible with the controller
loaders in this folder. The physical reward functions are embedded here so the
script can run without importing simulator-only reward code. One primary dataset
is written for each physical reward function tested in the offline RL
experiments, plus one all-rewards dataset containing every reward column.

Run from this folder or from the repository parent:

    python 6_Split_Data_Better_Steering/rosbag_to_reward_datasets.py

For interactive use in VS Code/Jupyter, run the # %% cells top to bottom.
"""

from __future__ import annotations

import bisect
import json
import math
import sqlite3
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    from PIL import Image
except ImportError:  # optional; image export is disabled by default
    Image = None

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore
except ImportError:
    Reader = None
    Stores = None
    get_types_from_msg = None
    get_typestore = None


# %%
# ---------------------------------------------------------------------------
# Paths and user settings
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
SCRIPT_DIR = THIS_FILE.parent if THIS_FILE.name.endswith(".py") else Path.cwd()
KNOWN_PROCESSING_DIRS = {
    "5_new_reward_function",
    "6_split_data_better_steering",
    "newdataprocessing",
}
if SCRIPT_DIR.name.lower() in KNOWN_PROCESSING_DIRS:
    WORKSPACE_ROOT = SCRIPT_DIR.parent
    PROCESSING_ROOT = SCRIPT_DIR
else:
    WORKSPACE_ROOT = SCRIPT_DIR
    PROCESSING_ROOT = WORKSPACE_ROOT / "6_Split_Data_Better_Steering"


# Edit these paths for new real-data dumps.
BAG_ROOT = Path("P:/Car/NewCar/NewTrackData")
BAG_ROOTS = [BAG_ROOT]
OUTPUT_ROOT = PROCESSING_ROOT

# Topic names seen in the original notebooks plus common alternatives.
ACKERMANN_TOPICS = [
    "/ackermann_cmd",
    "/drive",
]
LIDAR_TOPICS = [
    "/picoScan_23460001/scan/all_segments_echo0",
    "/scan",
    "/lidar",
]
ODOM_TOPICS = [
    "/odom",
    "/odometry",
]
RGB_TOPICS = [
    "/zed/zed_node/rgb/image_rect_color",
]
DEPTH_TOPICS = [
    "/zed/zed_node/depth/depth_registered",
]

# Leave empty to process all bags found under BAG_ROOT.
ONLY_BAGS: set[str] = set()
MAX_BAGS: int | None = None
MAX_TRANSITIONS_PER_BAG: int | None = None

# Dataset state settings. 108 is the default lidar beam count used by the
# simulator-trained policies; 54/27 are useful for reduced-state studies.
LIDAR_BEAMS = 108
OUTPUT_LIDAR_BEAMS: tuple[int, ...] | None = None
LIDAR_MAX_RANGE = 10.0
MAX_SPEED_MPS = 4.0
MAX_STEERING_RAD = 0.25
TRAINING_STEERING_CLAMP_RAD = 0.25
CONTROL_HZ = 10.0

# Extra scalar appended after lidar and before previous action.
# "front_ttc" keeps the old state width but replaces raw speed with a safety
# feature. Use "speed" for the previous behavior, or "none" to drop the scalar.
STATE_AUX_FEATURE = "front_ttc"
FRONT_TTC_MAX_SECONDS = 3.0
FRONT_TTC_MIN_SPEED_MPS = 0.05

# Real-data filtering and synchronization.
MIN_LIDAR_RANGE = 0.05
MAX_LIDAR_RANGE = 15.0
LIDAR_FILL_VALUE = LIDAR_MAX_RANGE
MIN_VALID_LIDAR_FRACTION = 0.70
MIN_VALID_LIDAR_BEAMS = 60
MAX_LIDAR_DT = 0.25
MAX_ODOM_DT = 0.25
MAX_ACTION_DT = 0.25
MAX_NEXT_STATE_DT = 0.35
MAX_ABS_SPEED_MPS = 8.0
MAX_ABS_STEERING_RAD = 1.2
MIN_MOVING_SPEED_MPS = 0.02

# Collision/terminal heuristics for real data when no collision counter exists.
TERMINAL_FRONT_ARC_FRACTION = 0.25
TERMINAL_CLEARANCE_THRESHOLD_M = 0.12
TERMINAL_PERCENTILE = 5.0
TERMINAL_MIN_CLOSE_BEAMS = 8

# Rewards tested in the offline RL experiments.
REWARD_NAMES = [
    "physical_reward_v1",
    "physical_reward_v2",
    "physical_reward_v3_candidate",
]

PRIMARY_REWARD_FOR_ALL_DATASET = "physical_reward_v1"
SAVE_PER_REWARD_DATASETS = True
SAVE_ALL_REWARDS_DATASET = False
SAVE_NPZ_DATASETS = True
SAVE_H5_DATASETS = True

# Only the strongest 60% of episodes are exported for training. Inside that
# selected set, "expert" is the higher-return half, "medium" is the lower-return
# half, and "mediumexpert" is the union of both.
SAVE_FULL_EPISODE_DATASETS = False
SAVE_QUALITY_SPLIT_DATASETS = True
EPISODE_SELECTION_FRACTION = 0.60
EPISODE_RANKING_METRIC = "return"
QUALITY_SPLIT_NAMES = ("medium", "expert", "mediumexpert")

REWARD_OUTPUT_DIRS: dict[str, Path] = {}
ALL_REWARDS_OUTPUT_DIR = OUTPUT_ROOT / "processed_reward_all_datasets"
OUTPUT_DIR = OUTPUT_ROOT / "processed_reward_V1_datasets"


def configure_output_dirs(output_root: Path | str | None = None) -> None:
    global OUTPUT_ROOT, REWARD_OUTPUT_DIRS, ALL_REWARDS_OUTPUT_DIR, OUTPUT_DIR
    if output_root is not None:
        OUTPUT_ROOT = Path(output_root)
    REWARD_OUTPUT_DIRS = {
        "physical_reward_v1": OUTPUT_ROOT / "processed_reward_V1_datasets",
        "physical_reward_v2": OUTPUT_ROOT / "processed_reward_V2_datasets",
        "physical_reward_v3_candidate": OUTPUT_ROOT / "processed_reward_V3_datasets",
    }
    ALL_REWARDS_OUTPUT_DIR = OUTPUT_ROOT / "processed_reward_all_datasets"
    OUTPUT_DIR = REWARD_OUTPUT_DIRS[PRIMARY_REWARD_FOR_ALL_DATASET]


configure_output_dirs(OUTPUT_ROOT)

EXPORT_IMAGES = False


# %%
# ---------------------------------------------------------------------------
# Standalone policy/reward compatibility layer
# ---------------------------------------------------------------------------

Position3D = tuple[float, float, float]
OrientationQuat = tuple[float, float, float, float]


@dataclass(frozen=True)
class Observation:
    lidar: np.ndarray
    position: Position3D | None = None
    orientation: OrientationQuat | None = None
    linear_speed: float | None = None
    stamp: float = 0.0


@dataclass(frozen=True)
class ControlAction:
    """Physical real-car command used by reward code: speed in m/s, steering in rad."""

    speed: float
    steering: float

    @property
    def throttle(self) -> float:
        # Kept only as a compatibility alias for older reward wording.
        return self.speed


PHYSICAL_REWARD_V1_CONFIG = {
    "lidar_min_range": 0.05,
    "lidar_max_range": 15.0,
    "crash_front_dist": 0.19,
    "safe_front_dist": 0.40,
    "crash_side_dist": 0.19,
    "safe_side_dist": 0.35,
    "max_speed": 4.0,
    "min_speed": 1.0,
    "w_front": 0.40,
    "w_side": 0.20,
    "w_speed": 0.30,
    "w_smooth": 0.10,
    "w_progress": 0.30,
    "progress_distance_scale": 0.20,
    "control_dt": 0.10,
    "smooth_alpha": 10.0,
    "min_progress_speed": 0.15,
    "full_progress_speed": 1.0,
    "stationary_penalty": -0.20,
    "safety_curve": "linear",
    "safety_beta": 6.0,
    "crash_penalty": -1.0,
}


PHYSICAL_REWARD_V2_CONFIG = {
    "lidar_min_range": 0.05,
    "lidar_max_range": 15.0,
    "lidar_fill_value": 15.0,
    "collision_threshold": 0.19,
    "terminal_collision_threshold": 0.12,
    "terminal_percentile": 5.0,
    "terminal_min_valid_beams": 20,
    "safe_front_dist": 0.75,
    "safe_side_dist": 0.45,
    "target_speed": 3.0,
    "ttc_safe": 1.0,
    "ttc_critical": 0.35,
    "steering_slowdown_angle": 0.65,
    "robust_k": 8,
    "robust_percentile": 5.0,
    "w_front": 0.30,
    "w_side": 0.15,
    "w_center": 0.10,
    "w_speed": 0.25,
    "w_steering": 0.10,
    "w_ttc": 0.10,
    "w_progress": 0.25,
    "progress_distance_scale": 0.20,
    "control_dt": 0.10,
    "min_progress_speed": 0.15,
    "full_progress_speed": 1.0,
    "stationary_penalty": -0.20,
    "crash_penalty": -1.0,
}


def physical_reward_v1(
    obs: Observation | None,
    action: ControlAction | np.ndarray | list[float] | tuple[float, ...],
    next_obs: Observation,
    env_info: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> float:
    """Physical-car reward adapted from OldDataProcessing `LidarRewardProcessor`."""
    cfg = _reward_config(config, "physical_reward_v1", PHYSICAL_REWARD_V1_CONFIG)
    lidar = _clean_lidar(next_obs.lidar, cfg)
    speed, steering = _speed_and_steering(next_obs, action, cfg)
    previous_steering = _previous_steering(env_info)

    right, front_right, front_left, left = np.array_split(lidar, 4)
    side_term = 0.5 * (
        _sector_safety(left, cfg["crash_side_dist"], cfg["safe_side_dist"], cfg)
        + _sector_safety(right, cfg["crash_side_dist"], cfg["safe_side_dist"], cfg)
    )
    front_term = 0.5 * (
        _sector_safety(front_left, cfg["crash_front_dist"], cfg["safe_front_dist"], cfg)
        + _sector_safety(front_right, cfg["crash_front_dist"], cfg["safe_front_dist"], cfg)
    )
    speed_term = _linear_score(speed, cfg["min_speed"], cfg["max_speed"])
    smooth_term = float(np.exp(-float(cfg["smooth_alpha"]) * abs(steering - previous_steering)))
    motion_gate = _motion_gate(speed, cfg)
    progress_term = _progress_score(obs, next_obs, speed, cfg) * min(front_term, side_term)

    safety_and_smoothness = (
        float(cfg["w_front"]) * front_term
        + float(cfg["w_side"]) * side_term
        + float(cfg["w_smooth"]) * smooth_term
    )
    reward = (
        motion_gate * safety_and_smoothness
        + float(cfg["w_speed"]) * speed_term
        + float(cfg["w_progress"]) * progress_term
    )
    if motion_gate <= 0.0:
        reward += float(cfg["stationary_penalty"])
    if _physical_crash(lidar, env_info, cfg):
        reward += float(cfg["crash_penalty"])
    return float(reward)


def physical_reward_v2(
    obs: Observation | None,
    action: ControlAction | np.ndarray | list[float] | tuple[float, ...],
    next_obs: Observation,
    env_info: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> float:
    """Physical-car reward adapted from the OldDataProcessing robust TTC reward."""
    cfg = _reward_config(config, "physical_reward_v2", PHYSICAL_REWARD_V2_CONFIG)
    lidar = _clean_lidar(next_obs.lidar, cfg)
    mask = _valid_lidar_mask(lidar, cfg)
    speed, steering = _speed_and_steering(next_obs, action, cfg)

    terms = physical_reward_v2_terms(lidar, mask, speed, steering, env_info, cfg)
    motion_gate = _motion_gate(speed, cfg)
    progress_term = _progress_score(obs, next_obs, speed, cfg) * min(
        terms["front_score"],
        terms["side_score"],
        terms["ttc_score"],
    )
    safety_and_control = (
        float(cfg["w_front"]) * terms["front_score"]
        + float(cfg["w_side"]) * terms["side_score"]
        + float(cfg["w_center"]) * terms["center_balance"]
        + float(cfg["w_steering"]) * terms["steering_score"]
        + float(cfg["w_ttc"]) * terms["ttc_score"]
    )
    reward = (
        motion_gate * safety_and_control
        + float(cfg["w_speed"]) * terms["safe_speed_score"]
        + float(cfg["w_progress"]) * progress_term
    )
    if motion_gate <= 0.0:
        reward += float(cfg["stationary_penalty"])
    if terms["terminal"]:
        reward += float(cfg["crash_penalty"])
    return float(reward)


def physical_reward_v2_terms(
    lidar: np.ndarray,
    lidar_valid_mask: np.ndarray,
    speed_cmd: float,
    steering_cmd: float,
    env_info: Mapping[str, Any] | None,
    cfg: Mapping[str, Any],
) -> dict[str, float | bool | int]:
    sectors = _split_lidar_sectors(lidar)
    masks = _split_lidar_sectors(lidar_valid_mask.astype(np.uint8))
    clearances = {
        name: _robust_sector_clearance(
            sectors[name],
            masks[name],
            k=int(cfg["robust_k"]),
            percentile=float(cfg["robust_percentile"]),
            min_range=float(cfg["lidar_min_range"]),
            max_range=float(cfg["lidar_max_range"]),
            fill_value=float(cfg["lidar_fill_value"]),
        )
        for name in sectors
    }

    terminal, terminal_clearance, terminal_close_beams = _terminal_from_lidar(
        lidar,
        lidar_valid_mask,
        threshold=float(cfg["terminal_collision_threshold"]),
        percentile=float(cfg["terminal_percentile"]),
        min_valid_beams=int(cfg["terminal_min_valid_beams"]),
        min_range=float(cfg["lidar_min_range"]),
        max_range=float(cfg["lidar_max_range"]),
    )
    terminal = bool(terminal or _env_collision(env_info))

    front_clearance = clearances["front"]
    side_clearance = 0.5 * (clearances["left"] + clearances["right"])
    center_balance = 1.0 - abs(clearances["left"] - clearances["right"]) / max(
        clearances["left"] + clearances["right"],
        1e-6,
    )
    center_balance = float(np.clip(center_balance, 0.0, 1.0))

    front_score = _linear_score(front_clearance, cfg["collision_threshold"], cfg["safe_front_dist"])
    side_score = _linear_score(side_clearance, cfg["collision_threshold"], cfg["safe_side_dist"])
    ttc = front_clearance / max(abs(float(speed_cmd)), 1e-3)
    ttc_score = _linear_score(ttc, cfg["ttc_critical"], cfg["ttc_safe"])

    speed_norm = float(
        np.clip(float(speed_cmd) / max(float(cfg["target_speed"]), 1e-6), 0.0, 1.0)
    )
    steering_gate = 1.0 - np.clip(
        abs(float(steering_cmd)) / max(float(cfg["steering_slowdown_angle"]), 1e-6),
        0.0,
        1.0,
    )
    safe_speed_score = speed_norm * min(front_score, ttc_score) * float(steering_gate)
    steering_score = float(np.exp(-2.0 * abs(float(steering_cmd))))

    return {
        "terminal": terminal,
        "terminal_clearance": float(terminal_clearance),
        "terminal_close_beams": int(terminal_close_beams),
        "front_clearance": float(front_clearance),
        "side_clearance": float(side_clearance),
        "center_balance": float(center_balance),
        "front_score": float(front_score),
        "side_score": float(side_score),
        "ttc": float(ttc),
        "ttc_score": float(ttc_score),
        "safe_speed_score": float(safe_speed_score),
        "steering_score": float(steering_score),
    }


def physical_reward_v3_candidate(
    obs: Observation | None,
    action: ControlAction | np.ndarray | list[float] | tuple[float, ...],
    next_obs: Observation,
    env_info: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> float:
    """Conservative physical-compatible fallback reward for post-failure ablations."""
    cfg = _reward_config(config, "physical_reward_v3_candidate", PHYSICAL_REWARD_V2_CONFIG)
    lidar = _clean_lidar(next_obs.lidar, cfg)
    mask = _valid_lidar_mask(lidar, cfg)
    speed, steering = _speed_and_steering(next_obs, action, cfg)
    terms = physical_reward_v2_terms(lidar, mask, speed, steering, env_info, cfg)
    front = terms["front_score"]
    ttc = terms["ttc_score"]
    side = terms["side_score"]
    smooth = float(np.exp(-2.5 * abs(float(steering))))
    motion_gate = _motion_gate(speed, cfg)
    progress_term = _progress_score(obs, next_obs, speed, cfg) * min(front, side, ttc)
    reward = motion_gate * (0.35 * min(front, ttc) + 0.20 * side + 0.20 * smooth)
    if front > 0.7 and ttc > 0.7:
        reward += 0.25 * np.clip(speed / max(float(cfg["target_speed"]), 1e-6), 0.0, 1.0)
    reward += float(cfg["w_progress"]) * progress_term
    if motion_gate <= 0.0:
        reward += float(cfg["stationary_penalty"])
    if terms["terminal"]:
        reward += float(cfg["crash_penalty"])
    return float(reward)


def _reward_config(
    config: Mapping[str, Any] | None,
    section: str,
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(defaults)
    if config:
        scoped = config.get(section, {}) if isinstance(config.get(section, {}), Mapping) else {}
        merged.update(scoped)
        for key in defaults:
            if key in config:
                merged[key] = config[key]
    return merged


def _clean_lidar(lidar: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(lidar, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    min_range = float(cfg["lidar_min_range"])
    max_range = float(cfg["lidar_max_range"])
    fill = float(cfg.get("lidar_fill_value", max_range))
    valid = np.isfinite(values) & (values > min_range) & (values <= max_range)
    clean = values.copy()
    clean[~valid] = fill
    return clean.astype(np.float32)


def _valid_lidar_mask(lidar: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(lidar, dtype=np.float32).reshape(-1)
    return (
        np.isfinite(values)
        & (values > float(cfg["lidar_min_range"]))
        & (values <= float(cfg["lidar_max_range"]))
    ).astype(np.uint8)


def _speed_and_steering(
    obs: Observation,
    action: ControlAction | np.ndarray | list[float] | tuple[float, ...],
    cfg: Mapping[str, Any],
) -> tuple[float, float]:
    control = _as_control_action(action)
    speed = obs.linear_speed
    if speed is None or not np.isfinite(float(speed)):
        speed = max(float(control.speed), 0.0) * float(
            cfg.get("max_speed", cfg.get("target_speed", 4.0))
        )
    return float(speed), float(control.steering)


def _previous_steering(env_info: Mapping[str, Any] | None) -> float:
    if not env_info:
        return 0.0
    previous = env_info.get("previous_action")
    if previous is None:
        return 0.0
    return float(_as_control_action(previous).steering)


def _as_control_action(
    value: ControlAction | np.ndarray | list[float] | tuple[float, ...],
) -> ControlAction:
    if isinstance(value, ControlAction):
        return value
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size < 2:
        raise ValueError("Physical reward action must contain speed and steering")
    return ControlAction(speed=float(values[0]), steering=float(values[1]))


def _sector_safety(
    distances: np.ndarray,
    crash: float,
    safe: float,
    cfg: Mapping[str, Any],
) -> float:
    values = np.asarray(distances, dtype=float)
    if values.size == 0:
        return 0.0
    minimum = float(np.nanmin(values))
    if not np.isfinite(minimum):
        return 0.0
    if str(cfg.get("safety_curve", "linear")).lower() == "exponential":
        return _exponential_score(minimum, crash, safe, float(cfg["safety_beta"]))
    return _linear_score(minimum, crash, safe)


def _linear_score(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x) or hi <= lo:
        return 0.0
    return float(np.clip((float(x) - float(lo)) / (float(hi) - float(lo)), 0.0, 1.0))


def _exponential_score(x: float, lo: float, hi: float, beta: float) -> float:
    t = _linear_score(x, lo, hi)
    if t <= 0.0 or beta <= 0.0:
        return t
    return float(np.clip((np.exp(beta * t) - 1.0) / (np.exp(beta) - 1.0), 0.0, 1.0))


def _motion_gate(speed: float, cfg: Mapping[str, Any]) -> float:
    return _linear_score(
        speed,
        float(cfg.get("min_progress_speed", 0.15)),
        float(cfg.get("full_progress_speed", 1.0)),
    )


def _progress_score(
    obs: Observation | None,
    next_obs: Observation,
    speed: float,
    cfg: Mapping[str, Any],
) -> float:
    distance = _position_delta(obs, next_obs)
    if distance is None:
        distance = max(float(speed), 0.0) * float(cfg.get("control_dt", 0.10))
    return _linear_score(distance, 0.0, float(cfg.get("progress_distance_scale", 0.20)))


def _position_delta(obs: Observation | None, next_obs: Observation) -> float | None:
    if obs is None or obs.position is None or next_obs.position is None:
        return None
    previous = np.asarray(obs.position[:2], dtype=np.float32)
    current = np.asarray(next_obs.position[:2], dtype=np.float32)
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
        return None
    return float(np.linalg.norm(current - previous))


def _split_lidar_sectors(lidar: np.ndarray) -> dict[str, np.ndarray]:
    right, front_right, front, front_left, left = np.array_split(np.asarray(lidar), 5)
    return {
        "right": right,
        "front_right": front_right,
        "front": front,
        "front_left": front_left,
        "left": left,
    }


def _robust_sector_clearance(
    sector_ranges: np.ndarray,
    sector_mask: np.ndarray | None,
    *,
    k: int,
    percentile: float,
    min_range: float,
    max_range: float,
    fill_value: float,
) -> float:
    clean = np.asarray(sector_ranges, dtype=np.float32)
    if sector_mask is not None and len(sector_mask) == len(clean):
        clean = clean[np.asarray(sector_mask, dtype=np.uint8) > 0]
    clean = clean[np.isfinite(clean)]
    clean = clean[(clean > min_range) & (clean <= max_range)]
    if clean.size == 0:
        return float(fill_value)
    p_value = float(np.percentile(clean, percentile))
    k_eff = min(int(k), int(clean.size))
    closest_k_mean = float(np.mean(np.partition(clean, k_eff - 1)[:k_eff]))
    return 0.5 * p_value + 0.5 * closest_k_mean


def _terminal_from_lidar(
    lidar: np.ndarray,
    lidar_valid_mask: np.ndarray,
    *,
    threshold: float,
    percentile: float,
    min_valid_beams: int,
    min_range: float,
    max_range: float,
) -> tuple[bool, float, int]:
    sectors = _split_lidar_sectors(lidar)
    masks = _split_lidar_sectors(lidar_valid_mask)
    values = sectors["front"]
    mask = masks["front"]
    valid_values = values[
        (mask > 0) & np.isfinite(values) & (values > min_range) & (values <= max_range)
    ]
    if valid_values.size == 0:
        return False, float("inf"), 0
    n_close = int(np.sum(valid_values < float(threshold)))
    clearance = float(np.percentile(valid_values, percentile))
    terminal = bool(
        valid_values.size >= int(min_valid_beams)
        and n_close >= int(min_valid_beams)
        and clearance < float(threshold)
    )
    return terminal, clearance, n_close


def _physical_crash(
    lidar: np.ndarray,
    env_info: Mapping[str, Any] | None,
    cfg: Mapping[str, Any],
) -> bool:
    if _env_collision(env_info):
        return True
    if lidar.size == 0:
        return False
    right, front_right, front_left, left = np.array_split(lidar, 4)
    side = np.concatenate([left, right])
    front = np.concatenate([front_left, front_right])
    side_min = float(np.nanmin(side)) if side.size else float("inf")
    front_min = float(np.nanmin(front)) if front.size else float("inf")
    return bool(
        (np.isfinite(side_min) and side_min <= float(cfg["crash_side_dist"]))
        or (np.isfinite(front_min) and front_min <= float(cfg["crash_front_dist"]))
    )


def _env_collision(env_info: Mapping[str, Any] | None) -> bool:
    if not env_info:
        return False
    return bool(
        env_info.get("terminated_collision", False)
        or env_info.get("collision", False)
        or float(env_info.get("collision_delta", 0) or 0) > 0
    )


RewardFunction = Callable[
    [Observation | None, ControlAction, Observation, Mapping[str, Any], Mapping[str, Any] | None],
    float,
]

_REWARD_REGISTRY: dict[str, RewardFunction] = {
    "physical_reward_v1": physical_reward_v1,
    "physical_reward_v2": physical_reward_v2,
    "physical_reward_v3_candidate": physical_reward_v3_candidate,
}


def compute_registered_reward(
    name: str,
    obs: Observation | None,
    action: ControlAction,
    next_obs: Observation,
    env_info: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> float:
    normalized = str(name).strip()
    if not normalized:
        raise ValueError("Reward name cannot be empty")
    try:
        reward_fn = _REWARD_REGISTRY[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_REWARD_REGISTRY))
        raise KeyError(f"Unknown reward '{normalized}'. Available rewards: {available}") from exc
    return float(reward_fn(obs, action, next_obs, env_info or {}, config))


# %%
# ---------------------------------------------------------------------------
# Ackermann type registration, metadata repair, and bag discovery
# ---------------------------------------------------------------------------

ACKERMANN_DRIVE_MSG = """
float32 steering_angle
float32 steering_angle_velocity
float32 speed
float32 acceleration
float32 jerk
"""

ACKERMANN_DRIVE_STAMPED_MSG = """
std_msgs/msg/Header header
ackermann_msgs/msg/AckermannDrive drive
"""


def require_rosbags() -> None:
    if Reader is None or Stores is None or get_types_from_msg is None or get_typestore is None:
        raise RuntimeError(
            "Missing dependency 'rosbags'. Install it in the processing environment with:\n"
            "    pip install rosbags lz4 zstandard"
        )


def make_typestore() -> Any:
    """Create a ROS2 typestore and register ackermann message types."""
    require_rosbags()
    preferred_stores = ("ROS2_HUMBLE", "ROS2_IRON", "ROS2_FOXY")
    last_error: Exception | None = None
    for store_name in preferred_stores:
        store = getattr(Stores, store_name, None)
        if store is None:
            continue
        try:
            typestore = get_typestore(store)
            types: dict[str, Any] = {}
            types.update(
                get_types_from_msg(
                    ACKERMANN_DRIVE_MSG,
                    "ackermann_msgs/msg/AckermannDrive",
                )
            )
            types.update(
                get_types_from_msg(
                    ACKERMANN_DRIVE_STAMPED_MSG,
                    "ackermann_msgs/msg/AckermannDriveStamped",
                )
            )
            typestore.register(types)
            return typestore
        except Exception as exc:  # keep trying older stores
            last_error = exc
    raise RuntimeError("Could not create a ROS2 typestore") from last_error


def normalize_metadata_yaml(bag_dir: Path) -> None:
    """Rewrite metadata.yaml so rosbags can read it on Windows.

    rosbags currently reads metadata.yaml with Path.read_text() and no explicit
    encoding. On Windows that can mean cp1252, so UTF-8 bytes for non-ASCII text
    may still fail. Metadata should be structural YAML, so replacing non-ASCII
    characters keeps the bag readable without touching the bag data.
    """
    yaml_path = bag_dir / "metadata.yaml"
    if not yaml_path.exists():
        return
    raw = yaml_path.read_bytes()

    def looks_like_rosbag_metadata(text: str) -> bool:
        return "rosbag2_bagfile_information" in text and "topics_with_message_count" in text

    def write_safe_yaml(text: str, source_encoding: str) -> None:
        text = text.replace("\x00", "")
        text = "".join(
            ch if ch in "\n\r\t" or 32 <= ord(ch) < 127 else "?"
            for ch in text
        )
        safe_text = text.encode("ascii", errors="replace").decode("ascii")
        yaml_path.write_text(safe_text, encoding="utf-8", newline="\n")
        print(f"[normalize] {bag_dir.name}: metadata.yaml {source_encoding} -> ascii-safe utf-8")

    try:
        text = raw.decode("utf-8")
        if not looks_like_rosbag_metadata(text):
            if rebuild_metadata_yaml_from_db3(bag_dir):
                return
        if any(ord(ch) >= 127 for ch in text) or "\x00" in text or any(
            ch not in "\n\r\t" and ord(ch) < 32 for ch in text
        ):
            write_safe_yaml(text, "utf-8")
        return
    except UnicodeDecodeError:
        pass

    for encoding in ("utf-16", "utf-16-le", "utf-16-be", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            if not looks_like_rosbag_metadata(text):
                if rebuild_metadata_yaml_from_db3(bag_dir):
                    return
            write_safe_yaml(text, encoding)
            return
        except Exception:
            continue
    text = raw.decode("utf-8", errors="replace")
    if not looks_like_rosbag_metadata(text):
        if rebuild_metadata_yaml_from_db3(bag_dir):
            return
    write_safe_yaml(text, "utf-8-with-replacement")


def rebuild_metadata_yaml_from_db3(bag_dir: Path) -> bool:
    """Rebuild a minimal metadata.yaml from sqlite topics when metadata is corrupt."""
    db_paths = sorted(bag_dir.glob("*.db3"))
    if not db_paths:
        return False

    topics: list[tuple[int, str, str, str, str]] = []
    per_topic_counts: dict[int, int] = {}
    total_count = 0
    start_ns = 0
    duration_ns = 0

    try:
        with sqlite3.connect(f"file:{db_paths[0].as_posix()}?mode=ro", uri=True, timeout=2) as con:
            cur = con.cursor()
            topics = [
                (
                    int(row[0]),
                    str(row[1]),
                    str(row[2]),
                    str(row[3]),
                    str(row[4] or ""),
                )
                for row in cur.execute(
                    "select id, name, type, serialization_format, offered_qos_profiles from topics order by id"
                )
            ]
            try:
                per_topic_counts = {
                    int(topic_id): int(count)
                    for topic_id, count in cur.execute(
                        "select topic_id, count(*) from messages group by topic_id"
                    )
                }
            except Exception:
                per_topic_counts = {}
            try:
                total_count, min_ts, max_ts = cur.execute(
                    "select count(*), min(timestamp), max(timestamp) from messages"
                ).fetchone()
                total_count = int(total_count or 0)
                start_ns = int(min_ts or 0)
                duration_ns = int(max(0, int(max_ts or 0) - start_ns))
            except Exception:
                total_count = int(sum(per_topic_counts.values()))
    except Exception as exc:
        print(f"[normalize] {bag_dir.name}: could not rebuild metadata from db3: {type(exc).__name__}: {exc}")
        return False

    if not topics:
        return False

    lines = [
        "rosbag2_bagfile_information:",
        "  version: 5",
        "  storage_identifier: sqlite3",
        "  duration:",
        f"    nanoseconds: {duration_ns}",
        "  starting_time:",
        f"    nanoseconds_since_epoch: {start_ns}",
        f"  message_count: {total_count}",
        "  topics_with_message_count:",
    ]
    for topic_id, name, msg_type, serialization_format, qos in topics:
        lines.extend(
            [
                "    - topic_metadata:",
                f"        name: {json.dumps(name)}",
                f"        type: {json.dumps(msg_type)}",
                f"        serialization_format: {json.dumps(serialization_format)}",
                f"        offered_qos_profiles: {json.dumps(qos)}",
                f"      message_count: {int(per_topic_counts.get(topic_id, 0))}",
            ]
        )
    lines.extend(
        [
            '  compression_format: ""',
            '  compression_mode: ""',
            "  relative_file_paths:",
        ]
    )
    for db_path in db_paths:
        lines.append(f"    - {json.dumps(db_path.name)}")
    lines.append("  files:")
    for db_path in db_paths:
        lines.extend(
            [
                f"    - path: {json.dumps(db_path.name)}",
                "      starting_time:",
                f"        nanoseconds_since_epoch: {start_ns}",
                "      duration:",
                f"        nanoseconds: {duration_ns}",
                f"      message_count: {total_count}",
            ]
        )
    lines.extend(
        [
            "  custom_data: ~",
            "",
        ]
    )
    (bag_dir / "metadata.yaml").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"[normalize] {bag_dir.name}: rebuilt metadata.yaml from sqlite topics")
    return True


def find_bag_dirs(root: Path) -> list[Path]:
    """Return folders that look like ROS2 bag directories."""
    bag_dirs: list[Path] = []
    for metadata_path in root.rglob("metadata.yaml"):
        bag_dir = metadata_path.parent
        if any(bag_dir.glob("*.db3")):
            if ONLY_BAGS and bag_dir.name not in ONLY_BAGS:
                continue
            bag_dirs.append(bag_dir)
    bag_dirs = sorted(set(bag_dirs))
    if MAX_BAGS is not None:
        bag_dirs = bag_dirs[: int(MAX_BAGS)]
    return bag_dirs


def bag_time_sec(t_ns: int) -> float:
    return float(t_ns) * 1e-9


def ros_time_to_sec(stamp: Any) -> float | None:
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    return float(sec) + float(nanosec) * 1e-9


def choose_topic(connections: Iterable[Any], candidates: list[str]) -> str | None:
    available = {conn.topic for conn in connections}
    for topic in candidates:
        if topic in available:
            return topic
    return None


# %%
# ---------------------------------------------------------------------------
# ROS message decoding
# ---------------------------------------------------------------------------


def decode_ackermann(msg: Any) -> dict[str, float] | None:
    drive = getattr(msg, "drive", None)
    if drive is None:
        # Some real logs store AckermannDrive directly.
        drive = msg
    speed = float(getattr(drive, "speed", np.nan))
    steering = float(getattr(drive, "steering_angle", np.nan))
    acceleration = float(getattr(drive, "acceleration", np.nan))
    jerk = float(getattr(drive, "jerk", np.nan))
    steering_velocity = float(getattr(drive, "steering_angle_velocity", np.nan))
    if not np.isfinite(speed) or not np.isfinite(steering):
        return None
    if abs(speed) > MAX_ABS_SPEED_MPS or abs(steering) > MAX_ABS_STEERING_RAD:
        return None
    return {
        "speed_mps": speed,
        "steering_rad": steering,
        "acceleration": acceleration if np.isfinite(acceleration) else 0.0,
        "jerk": jerk if np.isfinite(jerk) else 0.0,
        "steering_velocity": steering_velocity if np.isfinite(steering_velocity) else 0.0,
    }


def decode_laserscan(msg: Any) -> tuple[np.ndarray, dict[str, float]]:
    ranges = np.asarray(getattr(msg, "ranges", []), dtype=np.float32).reshape(-1)
    metadata = {
        "angle_min": float(getattr(msg, "angle_min", np.nan)),
        "angle_max": float(getattr(msg, "angle_max", np.nan)),
        "angle_increment": float(getattr(msg, "angle_increment", np.nan)),
        "time_increment": float(getattr(msg, "time_increment", np.nan)),
        "scan_time": float(getattr(msg, "scan_time", np.nan)),
        "range_min": float(getattr(msg, "range_min", MIN_LIDAR_RANGE)),
        "range_max": float(getattr(msg, "range_max", MAX_LIDAR_RANGE)),
    }
    return ranges, metadata


def decode_odometry(msg: Any) -> dict[str, Any] | None:
    pose = getattr(msg, "pose", None)
    twist = getattr(msg, "twist", None)
    pose_pose = getattr(pose, "pose", None)
    twist_twist = getattr(twist, "twist", None)
    position_msg = getattr(pose_pose, "position", None)
    orientation_msg = getattr(pose_pose, "orientation", None)
    linear_msg = getattr(twist_twist, "linear", None)
    angular_msg = getattr(twist_twist, "angular", None)

    position = None
    if position_msg is not None:
        position = (
            float(getattr(position_msg, "x", np.nan)),
            float(getattr(position_msg, "y", np.nan)),
            float(getattr(position_msg, "z", np.nan)),
        )
        if not np.all(np.isfinite(position)):
            position = None

    orientation = None
    if orientation_msg is not None:
        orientation = (
            float(getattr(orientation_msg, "x", np.nan)),
            float(getattr(orientation_msg, "y", np.nan)),
            float(getattr(orientation_msg, "z", np.nan)),
            float(getattr(orientation_msg, "w", np.nan)),
        )
        if not np.all(np.isfinite(orientation)):
            orientation = None

    linear_speed = None
    if linear_msg is not None:
        vx = float(getattr(linear_msg, "x", 0.0))
        vy = float(getattr(linear_msg, "y", 0.0))
        vz = float(getattr(linear_msg, "z", 0.0))
        if np.all(np.isfinite([vx, vy, vz])):
            linear_speed = float(math.sqrt(vx * vx + vy * vy + vz * vz))

    angular_z = None
    if angular_msg is not None:
        value = float(getattr(angular_msg, "z", np.nan))
        angular_z = value if np.isfinite(value) else None

    if position is None and linear_speed is None:
        return None
    return {
        "position": position,
        "orientation": orientation,
        "linear_speed": linear_speed,
        "angular_z": angular_z,
    }


def decode_image_to_numpy(msg: Any) -> tuple[np.ndarray, str]:
    """Robust image decoder copied/adapted from FullDataProcessing."""
    encoding = str(getattr(msg, "encoding", ""))
    height = int(getattr(msg, "height", 0))
    width = int(getattr(msg, "width", 0))
    step = int(getattr(msg, "step", 0))
    data = bytes(getattr(msg, "data", b""))
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image dimensions: width={width}, height={height}")

    enc = encoding.lower()

    def reshape_with_step(dtype: Any, channels: int) -> np.ndarray:
        itemsize = np.dtype(dtype).itemsize
        row_bytes = width * channels * itemsize
        if step <= 0:
            expected = height * row_bytes
            if len(data) < expected:
                raise ValueError(f"Image data too short: got {len(data)}, expected {expected}")
            return np.frombuffer(data[:expected], dtype=dtype).reshape(height, width, channels)
        needed = height * step
        if len(data) < needed:
            raise ValueError(f"Image data too short for step: got {len(data)}, need {needed}")
        buf = np.frombuffer(data[:needed], dtype=np.uint8).reshape(height, step)
        cropped = buf[:, :row_bytes].reshape(height, width, channels * itemsize)
        return cropped.view(dtype).reshape(height, width, channels)

    if enc in ("rgb8", "bgr8"):
        return reshape_with_step(np.uint8, 3), enc
    if enc in ("rgba8", "bgra8"):
        return reshape_with_step(np.uint8, 4), enc
    if enc == "mono8":
        return reshape_with_step(np.uint8, 1).reshape(height, width), enc
    if enc == "16uc1":
        return _decode_single_channel_image(data, height, width, step, np.uint16), enc
    if enc == "32fc1":
        return _decode_single_channel_image(data, height, width, step, np.float32), enc
    if step == width * 3:
        return np.frombuffer(data[: height * step], dtype=np.uint8).reshape(height, width, 3), "unknown_3ch_u8"
    if step == width * 4:
        return np.frombuffer(data[: height * step], dtype=np.uint8).reshape(height, width, 4), "unknown_4ch_u8"
    if step == width * 2:
        return np.frombuffer(data[: height * step], dtype=np.uint16).reshape(height, width), "unknown_1ch_u16"
    raise NotImplementedError(f"Unsupported image encoding: {encoding!r}")


def _decode_single_channel_image(
    data: bytes,
    height: int,
    width: int,
    step: int,
    dtype: Any,
) -> np.ndarray:
    itemsize = np.dtype(dtype).itemsize
    row_bytes = width * itemsize
    if step <= 0:
        expected = height * row_bytes
        if len(data) < expected:
            raise ValueError(f"Image data too short: got {len(data)}, expected {expected}")
        return np.frombuffer(data[:expected], dtype=dtype).reshape(height, width)
    needed = height * step
    if len(data) < needed:
        raise ValueError(f"Image data too short for step: got {len(data)}, need {needed}")
    buf = np.frombuffer(data[:needed], dtype=np.uint8).reshape(height, step)
    cropped = buf[:, :row_bytes].reshape(height, width, itemsize)
    return cropped.view(dtype).reshape(height, width)


def save_image(arr: np.ndarray, out_path: Path, encoding: str) -> None:
    if Image is None:
        raise RuntimeError("Pillow is not installed")
    enc = encoding.lower()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        if enc in ("bgr8", "bgra8"):
            arr = arr[..., ::-1] if arr.shape[2] == 3 else arr[..., [2, 1, 0, 3]]
        Image.fromarray(arr).save(out_path.as_posix())
    elif arr.ndim == 2 and arr.dtype == np.uint16:
        Image.fromarray(arr, mode="I;16").save(out_path.as_posix())
    elif arr.ndim == 2 and arr.dtype == np.float32:
        np.save(out_path.with_suffix(".npy"), arr)
        finite = np.isfinite(arr)
        if np.any(finite):
            lo = float(np.percentile(arr[finite], 1.0))
            hi = float(np.percentile(arr[finite], 99.0))
            if hi <= lo:
                hi = lo + 1e-6
            vis = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
            vis = np.nan_to_num(vis, nan=0.0, posinf=1.0, neginf=0.0)
            vis_u8 = (vis * 255.0).astype(np.uint8)
        else:
            vis_u8 = np.zeros_like(arr, dtype=np.uint8)
        Image.fromarray(vis_u8).save(out_path.with_name(out_path.stem + "_vis.png"))
    else:
        Image.fromarray(arr.astype(np.uint8)).save(out_path.as_posix())


# %%
# ---------------------------------------------------------------------------
# Filtering, normalization, and flattening
# ---------------------------------------------------------------------------


def clean_lidar_with_mask(ranges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clean real lidar ranges and keep a validity mask for filtering/metadata."""
    values = np.asarray(ranges, dtype=np.float32).reshape(-1)
    valid = np.isfinite(values) & (values > MIN_LIDAR_RANGE) & (values <= MAX_LIDAR_RANGE)
    cleaned = values.copy()
    cleaned[~valid] = np.float32(LIDAR_FILL_VALUE)
    cleaned = np.clip(cleaned, MIN_LIDAR_RANGE, LIDAR_FILL_VALUE).astype(np.float32)
    return cleaned, valid.astype(np.uint8)


def lidar_quality_ok(mask: np.ndarray) -> bool:
    if mask.size == 0:
        return False
    min_beams = min(int(MIN_VALID_LIDAR_BEAMS), int(mask.size))
    return bool(np.sum(mask > 0) >= min_beams and np.mean(mask > 0) >= MIN_VALID_LIDAR_FRACTION)


def downsample_ranges(ranges: np.ndarray, beams: int) -> np.ndarray:
    values = np.asarray(ranges, dtype=np.float32).reshape(-1)
    beams = int(beams)
    if beams <= 0:
        raise ValueError("beams must be positive")
    if values.size == 0:
        return np.full(beams, LIDAR_FILL_VALUE, dtype=np.float32)
    if values.size == beams:
        return values.astype(np.float32)
    source_x = np.linspace(0.0, 1.0, values.size)
    target_x = np.linspace(0.0, 1.0, beams)
    return np.interp(target_x, source_x, values).astype(np.float32)


def normalize_lidar(ranges: np.ndarray, beams: int = LIDAR_BEAMS) -> np.ndarray:
    reduced = downsample_ranges(ranges, beams)
    return np.clip(reduced / np.float32(LIDAR_MAX_RANGE), 0.0, 1.0).astype(np.float32)


def normalize_speed(speed_mps: float | None) -> float:
    if speed_mps is None or not np.isfinite(float(speed_mps)):
        return 0.0
    return float(np.clip(float(speed_mps) / max(float(MAX_SPEED_MPS), 1e-6), 0.0, 1.0))


def normalize_steering(steering_rad: float | None) -> float:
    if steering_rad is None or not np.isfinite(float(steering_rad)):
        return 0.0
    clipped = float(
        np.clip(
            float(steering_rad),
            -float(TRAINING_STEERING_CLAMP_RAD),
            float(TRAINING_STEERING_CLAMP_RAD),
        )
    )
    return float(
        np.clip(
            clipped / max(float(MAX_STEERING_RAD), 1e-6),
            -1.0,
            1.0,
        )
    )


def normalize_action(action: dict[str, float] | None) -> np.ndarray:
    if not action:
        return np.zeros(2, dtype=np.float32)
    return np.asarray(
        [
            normalize_speed(action.get("speed_mps")),
            normalize_steering(action.get("steering_rad")),
        ],
        dtype=np.float32,
    )


def front_arc_values(lidar_clean_m: np.ndarray, fraction: float = TERMINAL_FRONT_ARC_FRACTION) -> np.ndarray:
    values = np.asarray(lidar_clean_m, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    width = max(1, int(round(values.size * float(fraction))))
    center = values.size // 2
    start = max(0, center - width // 2)
    stop = min(values.size, start + width)
    front = values[start:stop]
    return front[
        np.isfinite(front)
        & (front > MIN_LIDAR_RANGE)
        & (front <= MAX_LIDAR_RANGE)
    ]


def front_clearance_m(lidar_clean_m: np.ndarray) -> float:
    front = front_arc_values(lidar_clean_m)
    if front.size == 0:
        return float(LIDAR_FILL_VALUE)
    return float(np.percentile(front, TERMINAL_PERCENTILE))


def front_ttc_seconds(lidar_clean_m: np.ndarray, speed_mps: float | None) -> float:
    if speed_mps is None or not np.isfinite(float(speed_mps)):
        return float("inf")
    speed = abs(float(speed_mps))
    if speed < FRONT_TTC_MIN_SPEED_MPS:
        return float("inf")
    return front_clearance_m(lidar_clean_m) / speed


def normalize_front_ttc(lidar_clean_m: np.ndarray, speed_mps: float | None) -> float:
    ttc = front_ttc_seconds(lidar_clean_m, speed_mps)
    if not np.isfinite(ttc):
        return 1.0
    return float(np.clip(ttc / max(float(FRONT_TTC_MAX_SECONDS), 1e-6), 0.0, 1.0))


def state_aux_feature(lidar_clean_m: np.ndarray, speed_mps: float | None) -> np.ndarray:
    mode = str(STATE_AUX_FEATURE).strip().lower()
    if mode in {"", "none"}:
        return np.empty(0, dtype=np.float32)
    if mode == "speed":
        return np.asarray([normalize_speed(speed_mps)], dtype=np.float32)
    if mode == "front_ttc":
        return np.asarray([normalize_front_ttc(lidar_clean_m, speed_mps)], dtype=np.float32)
    raise ValueError("STATE_AUX_FEATURE must be one of: 'front_ttc', 'speed', 'none'")


def flatten_state(
    lidar_clean_m: np.ndarray,
    speed_mps: float | None,
    previous_action_norm: np.ndarray,
    *,
    beams: int = LIDAR_BEAMS,
) -> np.ndarray:
    return np.concatenate(
        [
            normalize_lidar(lidar_clean_m, beams=beams),
            state_aux_feature(lidar_clean_m, speed_mps),
            np.asarray(previous_action_norm, dtype=np.float32).reshape(2),
        ]
    ).astype(np.float32)


def front_collision_from_lidar(lidar_clean_m: np.ndarray, mask: np.ndarray) -> tuple[bool, float, int]:
    values = np.asarray(lidar_clean_m, dtype=np.float32).reshape(-1)
    valid = np.asarray(mask, dtype=np.uint8).reshape(-1) > 0
    if values.size == 0 or valid.size != values.size:
        return False, float("inf"), 0
    width = max(1, int(round(values.size * TERMINAL_FRONT_ARC_FRACTION)))
    center = values.size // 2
    start = max(0, center - width // 2)
    stop = min(values.size, start + width)
    sector = values[start:stop]
    sector_valid = valid[start:stop]
    sector = sector[sector_valid & np.isfinite(sector)]
    sector = sector[(sector > MIN_LIDAR_RANGE) & (sector <= MAX_LIDAR_RANGE)]
    if sector.size == 0:
        return False, float("inf"), 0
    clearance = float(np.percentile(sector, TERMINAL_PERCENTILE))
    n_close = int(np.sum(sector < TERMINAL_CLEARANCE_THRESHOLD_M))
    terminal = bool(n_close >= TERMINAL_MIN_CLOSE_BEAMS and clearance < TERMINAL_CLEARANCE_THRESHOLD_M)
    return terminal, clearance, n_close


# %%
# ---------------------------------------------------------------------------
# Stream containers and alignment helpers
# ---------------------------------------------------------------------------


@dataclass
class TimedAction:
    t: float
    raw: dict[str, float]
    normalized: np.ndarray


@dataclass
class TimedOdom:
    t: float
    position: tuple[float, float, float] | None
    orientation: tuple[float, float, float, float] | None
    linear_speed: float | None
    angular_z: float | None


@dataclass
class TimedLidar:
    t: float
    raw: np.ndarray
    clean: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, float]


@dataclass
class StateFrame:
    bag: str
    t: float
    lidar: TimedLidar
    odom: TimedOdom | None
    action: TimedAction
    previous_action: TimedAction | None
    episode_id: int
    timestep: int


def previous_item(items: list[Any], t: float, max_dt: float | None) -> tuple[Any, float] | None:
    if not items:
        return None
    times = [item.t for item in items]
    idx = bisect.bisect_right(times, t) - 1
    if idx < 0:
        return None
    dt = float(t - items[idx].t)
    if dt < 0 or (max_dt is not None and dt > float(max_dt)):
        return None
    return items[idx], dt


def observation_from_frame(frame: StateFrame) -> Observation:
    odom = frame.odom
    return Observation(
        lidar=frame.lidar.clean.astype(np.float32),
        position=odom.position if odom else None,
        orientation=odom.orientation if odom else None,
        linear_speed=odom.linear_speed if odom else frame.action.raw.get("speed_mps"),
        stamp=frame.t,
    )


def env_info_for_transition(
    frame: StateFrame,
    next_frame: StateFrame,
    terminal_collision: bool,
    terminal_clearance: float,
    terminal_close_beams: int,
) -> dict[str, Any]:
    previous_action = (
        ControlAction(
            speed=float(frame.previous_action.raw.get("speed_mps", 0.0)),
            steering=float(frame.previous_action.raw.get("steering_rad", 0.0)),
        )
        if frame.previous_action is not None
        else ControlAction(speed=0.0, steering=0.0)
    )
    return {
        "previous_action": previous_action,
        "collision": bool(terminal_collision),
        "terminated_collision": bool(terminal_collision),
        "collision_delta": int(terminal_collision),
        "lap_target_reached": False,
        "front_min_range": float(np.min(next_frame.lidar.clean))
        if next_frame.lidar.clean.size
        else None,
        "terminal_clearance": terminal_clearance,
        "terminal_close_beams": terminal_close_beams,
    }


# %%
# ---------------------------------------------------------------------------
# Bag decoding and transition assembly
# ---------------------------------------------------------------------------


def read_bag_streams(bag_dir: Path, typestore: Any) -> tuple[list[TimedLidar], list[TimedAction], list[TimedOdom], dict[str, Any]]:
    require_rosbags()
    normalize_metadata_yaml(bag_dir)
    stats = {
        "bag": bag_dir.name,
        "lidar_seen": 0,
        "lidar_kept": 0,
        "lidar_dropped_quality": 0,
        "actions_seen": 0,
        "actions_kept": 0,
        "odom_seen": 0,
        "odom_kept": 0,
        "decode_errors": 0,
        "topics": {},
    }
    lidars: list[TimedLidar] = []
    actions: list[TimedAction] = []
    odoms: list[TimedOdom] = []

    with Reader(bag_dir.as_posix()) as reader:
        connections = list(reader.connections)
        ack_topic = choose_topic(connections, ACKERMANN_TOPICS)
        lidar_topic = choose_topic(connections, LIDAR_TOPICS)
        odom_topic = choose_topic(connections, ODOM_TOPICS)
        rgb_topic = choose_topic(connections, RGB_TOPICS)
        depth_topic = choose_topic(connections, DEPTH_TOPICS)
        selected_topics = {topic for topic in [ack_topic, lidar_topic, odom_topic, rgb_topic, depth_topic] if topic}
        if ack_topic is None:
            raise RuntimeError(f"Missing Ackermann topic in {bag_dir.name}. Tried: {ACKERMANN_TOPICS}")
        if lidar_topic is None:
            raise RuntimeError(f"Missing lidar topic in {bag_dir.name}. Tried: {LIDAR_TOPICS}")
        stats["topics"] = {
            "ackermann": ack_topic,
            "lidar": lidar_topic,
            "odom": odom_topic,
            "rgb": rgb_topic,
            "depth": depth_topic,
        }
        selected_connections = [conn for conn in connections if conn.topic in selected_topics]

        image_index = 0
        for conn, t_ns, rawdata in reader.messages(connections=selected_connections):
            t = bag_time_sec(t_ns)
            try:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            except Exception as exc:
                stats["decode_errors"] += 1
                if stats["decode_errors"] <= 5:
                    print(f"[{bag_dir.name}] deserialize error {conn.topic}: {type(exc).__name__}: {exc}")
                continue

            try:
                if conn.topic == ack_topic:
                    stats["actions_seen"] += 1
                    decoded = decode_ackermann(msg)
                    if decoded is None:
                        continue
                    actions.append(TimedAction(t=t, raw=decoded, normalized=normalize_action(decoded)))
                    stats["actions_kept"] += 1
                elif conn.topic == lidar_topic:
                    stats["lidar_seen"] += 1
                    raw_ranges, metadata = decode_laserscan(msg)
                    clean, mask = clean_lidar_with_mask(raw_ranges)
                    if not lidar_quality_ok(mask):
                        stats["lidar_dropped_quality"] += 1
                        continue
                    lidars.append(TimedLidar(t=t, raw=raw_ranges, clean=clean, valid_mask=mask, metadata=metadata))
                    stats["lidar_kept"] += 1
                elif odom_topic and conn.topic == odom_topic:
                    stats["odom_seen"] += 1
                    decoded_odom = decode_odometry(msg)
                    if decoded_odom is None:
                        continue
                    odoms.append(TimedOdom(t=t, **decoded_odom))
                    stats["odom_kept"] += 1
                elif EXPORT_IMAGES and conn.topic in {rgb_topic, depth_topic}:
                    arr, enc = decode_image_to_numpy(msg)
                    image_dir = PROCESSING_ROOT / "processed_reward_images" / bag_dir.name / conn.topic.strip("/").replace("/", "__")
                    save_image(arr, image_dir / f"{image_index:06d}.png", enc)
                    image_index += 1
            except Exception as exc:
                stats["decode_errors"] += 1
                if stats["decode_errors"] <= 5:
                    print(f"[{bag_dir.name}] decode error {conn.topic}: {type(exc).__name__}: {exc}")

    actions.sort(key=lambda item: item.t)
    lidars.sort(key=lambda item: item.t)
    odoms.sort(key=lambda item: item.t)
    return lidars, actions, odoms, stats


def build_state_frames(
    bag_name: str,
    lidars: list[TimedLidar],
    actions: list[TimedAction],
    odoms: list[TimedOdom],
    *,
    episode_id: int,
) -> tuple[list[StateFrame], dict[str, int]]:
    frames: list[StateFrame] = []
    stats = {
        "frames_considered": len(lidars),
        "frames_kept": 0,
        "dropped_no_action": 0,
        "dropped_no_odom": 0,
        "dropped_stationary": 0,
    }
    for timestep, lidar in enumerate(lidars):
        action_match = previous_item(actions, lidar.t, MAX_ACTION_DT)
        if action_match is None:
            stats["dropped_no_action"] += 1
            continue
        action, _ = action_match
        odom = None
        if odoms:
            odom_match = previous_item(odoms, lidar.t, MAX_ODOM_DT)
            if odom_match is None:
                stats["dropped_no_odom"] += 1
                continue
            odom, _ = odom_match
        speed = odom.linear_speed if odom and odom.linear_speed is not None else action.raw.get("speed_mps")
        if speed is not None and abs(float(speed)) < MIN_MOVING_SPEED_MPS:
            stats["dropped_stationary"] += 1
            continue
        prev_action_match = previous_item(actions, action.t - 1e-9, MAX_ACTION_DT)
        previous_action = prev_action_match[0] if prev_action_match else None
        frames.append(
            StateFrame(
                bag=bag_name,
                t=lidar.t,
                lidar=lidar,
                odom=odom,
                action=action,
                previous_action=previous_action,
                episode_id=episode_id,
                timestep=len(frames),
            )
        )
        stats["frames_kept"] += 1
    return frames, stats


def build_transitions_from_frames(frames: list[StateFrame]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    transitions: list[dict[str, Any]] = []
    stats = {
        "candidate_pairs": max(0, len(frames) - 1),
        "transitions_kept": 0,
        "dropped_next_gap": 0,
        "dropped_after_terminal": 0,
        "terminals": 0,
        "episode_splits_from_terminal": 0,
        "episode_splits_from_gap": 0,
        "episodes": 0,
    }
    if not frames:
        return transitions, stats

    current_episode_id = int(frames[0].episode_id)
    episode_timestep = 0
    skip_terminal_state_pair = False
    candidate_pairs = list(zip(frames[:-1], frames[1:]))
    for pair_index, (current, nxt) in enumerate(candidate_pairs):
        if skip_terminal_state_pair:
            stats["dropped_after_terminal"] += 1
            skip_terminal_state_pair = False
            continue

        dt = float(nxt.t - current.t)
        if dt <= 0 or dt > MAX_NEXT_STATE_DT:
            stats["dropped_next_gap"] += 1
            if episode_timestep > 0:
                current_episode_id += 1
                episode_timestep = 0
                stats["episode_splits_from_gap"] += 1
            continue

        terminal_collision, clearance, close_beams = front_collision_from_lidar(
            nxt.lidar.clean,
            nxt.lidar.valid_mask,
        )
        terminal = bool(terminal_collision)
        action_norm = current.action.normalized.astype(np.float32)
        previous_action_norm = (
            current.previous_action.normalized.astype(np.float32)
            if current.previous_action is not None
            else np.zeros(2, dtype=np.float32)
        )
        next_previous_action_norm = action_norm
        current_speed = (
            current.odom.linear_speed
            if current.odom and current.odom.linear_speed is not None
            else current.action.raw.get("speed_mps")
        )
        next_speed = (
            nxt.odom.linear_speed
            if nxt.odom and nxt.odom.linear_speed is not None
            else nxt.action.raw.get("speed_mps")
        )
        obs_vec = flatten_state(
            current.lidar.clean,
            current_speed,
            previous_action_norm,
        )
        next_obs_vec = flatten_state(
            nxt.lidar.clean,
            next_speed,
            next_previous_action_norm,
        )
        obs = observation_from_frame(current)
        next_obs = observation_from_frame(nxt)
        env_info = env_info_for_transition(current, nxt, terminal_collision, clearance, close_beams)
        action_control = ControlAction(
            speed=float(current.action.raw.get("speed_mps", 0.0)),
            steering=float(current.action.raw.get("steering_rad", 0.0)),
        )

        rewards = {
            f"reward_{name}": float(
                compute_registered_reward(name, obs, action_control, next_obs, env_info, None)
            )
            for name in REWARD_NAMES
        }
        transitions.append(
            {
                "observation": obs_vec,
                "next_observation": next_obs_vec,
                "action": action_norm,
                "reward_columns": rewards,
                "done": terminal,
                "terminal": terminal,
                "truncated": False,
                "episode_id": current_episode_id,
                "timestep": episode_timestep,
                "reward_name": "all_physical_rewards",
                "raw_lidar": current.lidar.clean.astype(np.float32),
                "next_raw_lidar": nxt.lidar.clean.astype(np.float32),
                "raw_lidar_length": int(current.lidar.clean.size),
                "next_raw_lidar_length": int(nxt.lidar.clean.size),
                "linear_speed": obs.linear_speed if obs.linear_speed is not None else np.nan,
                "next_linear_speed": next_obs.linear_speed if next_obs.linear_speed is not None else np.nan,
                "front_ttc": front_ttc_seconds(current.lidar.clean, current_speed),
                "front_ttc_norm": normalize_front_ttc(current.lidar.clean, current_speed),
                "next_front_ttc": front_ttc_seconds(nxt.lidar.clean, next_speed),
                "next_front_ttc_norm": normalize_front_ttc(nxt.lidar.clean, next_speed),
                "control_action": action_norm,
                "previous_control_action": previous_action_norm,
                "info": {
                    "bag": current.bag,
                    "t": current.t,
                    "next_t": nxt.t,
                    "dt": dt,
                    "bag_frame_timestep": int(current.timestep),
                    "next_bag_frame_timestep": int(nxt.timestep),
                    "episode_id": int(current_episode_id),
                    "episode_timestep": int(episode_timestep),
                    "terminal_clearance": clearance,
                    "terminal_close_beams": close_beams,
                    "terminal_collision": terminal_collision,
                    "lidar_valid_fraction": float(np.mean(current.lidar.valid_mask > 0)),
                    "next_lidar_valid_fraction": float(np.mean(nxt.lidar.valid_mask > 0)),
                    "raw_speed_mps": current.action.raw.get("speed_mps"),
                    "raw_steering_rad": current.action.raw.get("steering_rad"),
                    "training_steering_rad": float(action_norm[1]) * float(MAX_STEERING_RAD),
                },
            }
        )
        stats["transitions_kept"] += 1
        stats["terminals"] += int(terminal)
        episode_timestep += 1
        if terminal:
            current_episode_id += 1
            episode_timestep = 0
            skip_terminal_state_pair = True
            stats["episode_splits_from_terminal"] += 1
        if MAX_TRANSITIONS_PER_BAG is not None and stats["transitions_kept"] >= int(MAX_TRANSITIONS_PER_BAG):
            break
    stats["episodes"] = len({int(item["episode_id"]) for item in transitions})
    return transitions, stats


def process_bag_dir(bag_dir: Path, typestore: Any, episode_id: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lidars, actions, odoms, read_stats = read_bag_streams(bag_dir, typestore)
    frames, frame_stats = build_state_frames(
        bag_dir.name,
        lidars,
        actions,
        odoms,
        episode_id=episode_id,
    )
    transitions, transition_stats = build_transitions_from_frames(frames)
    stats = {
        **read_stats,
        **{f"frame_{k}": v for k, v in frame_stats.items()},
        **{f"transition_{k}": v for k, v in transition_stats.items()},
    }
    return transitions, stats


# %%
# ---------------------------------------------------------------------------
# Dataset export
# ---------------------------------------------------------------------------


def stack_vectors(transitions: list[dict[str, Any]], key: str, dtype: Any) -> np.ndarray:
    if not transitions:
        return np.empty((0, 0), dtype=dtype)
    return np.stack([np.asarray(item[key], dtype=dtype).reshape(-1) for item in transitions]).astype(dtype)


def stack_lidar(transitions: list[dict[str, Any]], key: str) -> np.ndarray:
    if not transitions:
        return np.empty((0, 0), dtype=np.float32)
    max_len = max(int(np.asarray(item[key]).size) for item in transitions)
    out = np.full((len(transitions), max_len), LIDAR_FILL_VALUE, dtype=np.float32)
    for i, item in enumerate(transitions):
        values = np.asarray(item[key], dtype=np.float32).reshape(-1)
        out[i, : values.size] = values
    return out


def reward_statistics(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def output_dir_for_reward(reward_name: str, split_name: str | None = None) -> Path:
    try:
        base = REWARD_OUTPUT_DIRS[reward_name]
    except KeyError as exc:
        raise KeyError(f"No output directory configured for reward: {reward_name}") from exc
    if split_name:
        return base / split_name
    return base


def group_transitions_by_episode(transitions: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    episodes: dict[int, list[dict[str, Any]]] = {}
    for item in transitions:
        episode_id = int(item["episode_id"])
        episodes.setdefault(episode_id, []).append(item)
    return episodes


def episode_summary(
    episode_id: int,
    items: list[dict[str, Any]],
    *,
    reward_name: str,
) -> dict[str, Any]:
    reward_key = f"reward_{reward_name}"
    rewards = np.asarray([item["reward_columns"][reward_key] for item in items], dtype=np.float32)
    bags = sorted({str(item["info"].get("bag", "")) for item in items})
    start_times = [float(item["info"].get("t", np.nan)) for item in items]
    end_times = [float(item["info"].get("next_t", np.nan)) for item in items]
    finite_start_times = [t for t in start_times if np.isfinite(t)]
    finite_end_times = [t for t in end_times if np.isfinite(t)]
    terminals = [bool(item.get("terminal", False)) for item in items]
    terminal_collisions = [
        bool(item.get("info", {}).get("terminal_collision", False))
        for item in items
    ]
    return {
        "episode_id": int(episode_id),
        "bag": bags[0] if len(bags) == 1 else ",".join(bags),
        "num_transitions": int(len(items)),
        "start_t": float(min(finite_start_times)) if finite_start_times else None,
        "end_t": float(max(finite_end_times)) if finite_end_times else None,
        "duration_s": (
            float(max(finite_end_times) - min(finite_start_times))
            if finite_start_times and finite_end_times
            else None
        ),
        "return": float(np.sum(rewards)) if rewards.size else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards.size else 0.0,
        "terminal": bool(any(terminals)),
        "terminal_collision": bool(any(terminal_collisions)),
    }


def rank_episode_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric = str(EPISODE_RANKING_METRIC).strip().lower()
    if metric not in {"return", "mean_reward", "num_transitions"}:
        raise ValueError("EPISODE_RANKING_METRIC must be one of: return, mean_reward, num_transitions")
    return sorted(
        summaries,
        key=lambda item: (float(item[metric]), int(item["num_transitions"])),
        reverse=True,
    )


def split_top_episode_transitions(
    transitions: list[dict[str, Any]],
    *,
    reward_name: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    episodes = group_transitions_by_episode(transitions)
    summaries = [
        episode_summary(episode_id, items, reward_name=reward_name)
        for episode_id, items in episodes.items()
        if items
    ]
    ranked = rank_episode_summaries(summaries)
    if not ranked:
        empty = {name: [] for name in QUALITY_SPLIT_NAMES}
        return empty, {
            "reward_name": reward_name,
            "ranking_metric": EPISODE_RANKING_METRIC,
            "selection_fraction": EPISODE_SELECTION_FRACTION,
            "total_episodes": 0,
            "selected_episode_count": 0,
            "splits": {name: [] for name in QUALITY_SPLIT_NAMES},
        }

    selection_count = max(1, int(math.ceil(len(ranked) * float(EPISODE_SELECTION_FRACTION))))
    selection_count = min(selection_count, len(ranked))
    selected = ranked[:selection_count]
    selected_ids = [int(item["episode_id"]) for item in selected]

    if len(selected_ids) == 1:
        expert_ids = selected_ids
        medium_ids = selected_ids
    else:
        expert_count = max(1, int(math.ceil(len(selected_ids) / 2.0)))
        expert_count = min(expert_count, len(selected_ids) - 1)
        expert_ids = selected_ids[:expert_count]
        medium_ids = selected_ids[expert_count:]

    split_episode_ids = {
        "medium": medium_ids,
        "expert": expert_ids,
        "mediumexpert": selected_ids,
    }
    split_sets = {
        split_name: set(ids)
        for split_name, ids in split_episode_ids.items()
    }
    split_transitions = {
        split_name: [
            item for item in transitions
            if int(item["episode_id"]) in episode_ids
        ]
        for split_name, episode_ids in split_sets.items()
    }
    metadata = {
        "reward_name": reward_name,
        "ranking_metric": EPISODE_RANKING_METRIC,
        "selection_fraction": float(EPISODE_SELECTION_FRACTION),
        "total_episodes": int(len(ranked)),
        "selected_episode_count": int(len(selected_ids)),
        "selected_episode_ids": selected_ids,
        "splits": {
            split_name: {
                "episode_ids": ids,
                "episode_count": int(len(ids)),
                "transition_count": int(len(split_transitions[split_name])),
            }
            for split_name, ids in split_episode_ids.items()
        },
        "ranked_episodes": ranked,
    }
    return split_transitions, metadata


def output_lidar_beam_counts() -> list[int]:
    values = OUTPUT_LIDAR_BEAMS if OUTPUT_LIDAR_BEAMS else (LIDAR_BEAMS,)
    beam_counts: list[int] = []
    for value in values:
        beams = int(value)
        if beams <= 0:
            raise ValueError("Lidar beam counts must be positive")
        if beams not in beam_counts:
            beam_counts.append(beams)
    return beam_counts


def refresh_transition_observations(transitions: list[dict[str, Any]], beams: int) -> None:
    for item in transitions:
        item["observation"] = flatten_state(
            item["raw_lidar"],
            item["linear_speed"],
            item["previous_control_action"],
            beams=beams,
        )
        item["next_observation"] = flatten_state(
            item["next_raw_lidar"],
            item["next_linear_speed"],
            item["control_action"],
            beams=beams,
        )


def arrays_from_transitions(
    transitions: list[dict[str, Any]],
    *,
    primary_reward: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    primary_column = f"reward_{primary_reward}"
    reward_columns = {
        f"reward_{name}": np.asarray(
            [item["reward_columns"][f"reward_{name}"] for item in transitions],
            dtype=np.float32,
        )
        for name in REWARD_NAMES
    }
    arrays: dict[str, np.ndarray] = {
        "observations": stack_vectors(transitions, "observation", np.float32),
        "actions": stack_vectors(transitions, "action", np.float32),
        "rewards": reward_columns[primary_column].astype(np.float32),
        "next_observations": stack_vectors(transitions, "next_observation", np.float32),
        "dones": np.asarray([item["done"] for item in transitions], dtype=np.bool_),
        "terminals": np.asarray([item["terminal"] for item in transitions], dtype=np.bool_),
        "truncations": np.asarray([item["truncated"] for item in transitions], dtype=np.bool_),
        "episode_ids": np.asarray([item["episode_id"] for item in transitions], dtype=np.int64),
        "timesteps": np.asarray([item["timestep"] for item in transitions], dtype=np.int64),
        "reward_names": np.asarray([primary_reward for _ in transitions], dtype=np.str_),
        "raw_lidar": stack_lidar(transitions, "raw_lidar"),
        "raw_lidar_lengths": np.asarray([item["raw_lidar_length"] for item in transitions], dtype=np.int64),
        "next_raw_lidar": stack_lidar(transitions, "next_raw_lidar"),
        "next_raw_lidar_lengths": np.asarray(
            [item["next_raw_lidar_length"] for item in transitions],
            dtype=np.int64,
        ),
        "linear_speed": np.asarray([item["linear_speed"] for item in transitions], dtype=np.float32),
        "next_linear_speed": np.asarray(
            [item["next_linear_speed"] for item in transitions],
            dtype=np.float32,
        ),
        "front_ttc": np.asarray([item["front_ttc"] for item in transitions], dtype=np.float32),
        "front_ttc_norm": np.asarray([item["front_ttc_norm"] for item in transitions], dtype=np.float32),
        "next_front_ttc": np.asarray([item["next_front_ttc"] for item in transitions], dtype=np.float32),
        "next_front_ttc_norm": np.asarray(
            [item["next_front_ttc_norm"] for item in transitions],
            dtype=np.float32,
        ),
        "lap_count": np.full(len(transitions), -1, dtype=np.int64),
        "next_lap_count": np.full(len(transitions), -1, dtype=np.int64),
        "collision_count": np.zeros(len(transitions), dtype=np.int64),
        "next_collision_count": np.asarray([int(item["terminal"]) for item in transitions], dtype=np.int64),
        "control_actions": stack_vectors(transitions, "control_action", np.float32),
        "previous_control_actions": stack_vectors(transitions, "previous_control_action", np.float32),
        **reward_columns,
    }
    metadata = {
        "dataset_format": "autodrive_real_rosbag_reward_npz_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary_reward": primary_reward,
        "reward_columns": {name: f"reward_{name}" for name in REWARD_NAMES},
        "num_transitions": int(len(transitions)),
        "num_episodes": int(len(set(int(item["episode_id"]) for item in transitions))) if transitions else 0,
        "episode_split_unit": "terminal-separated segments within each bag; large next-state gaps also start a new episode",
        "observation_shape": list(arrays["observations"].shape[1:]),
        "observation_components": {
            "lidar": LIDAR_BEAMS,
            "auxiliary": STATE_AUX_FEATURE,
            "previous_action": ["speed", "steering"],
        },
        "action_shape": list(arrays["actions"].shape[1:]),
        "action_components": ["speed", "steering"],
        "normalization": {
            "lidar": f"clip valid ranges to [{MIN_LIDAR_RANGE}, {LIDAR_FILL_VALUE}] m, downsample to {LIDAR_BEAMS}, divide by {LIDAR_MAX_RANGE}",
            "speed": f"clip speed / {MAX_SPEED_MPS} to [0, 1]",
            "steering": f"clip logged steering to +/-{TRAINING_STEERING_CLAMP_RAD} rad, then divide by {MAX_STEERING_RAD} and clip to [-1, 1]",
            "front_ttc": f"clip TTC / {FRONT_TTC_MAX_SECONDS} seconds to [0, 1], where 1 means no immediate front TTC risk",
            "actions": "first action is normalized speed [0, 1], second action is normalized steering [-1, 1]",
            "state_aux_feature": STATE_AUX_FEATURE,
        },
        "filtering": {
            "min_lidar_range": MIN_LIDAR_RANGE,
            "max_lidar_range": MAX_LIDAR_RANGE,
            "min_valid_lidar_fraction": MIN_VALID_LIDAR_FRACTION,
            "min_valid_lidar_beams": MIN_VALID_LIDAR_BEAMS,
            "max_lidar_dt": MAX_LIDAR_DT,
            "max_odom_dt": MAX_ODOM_DT,
            "max_action_dt": MAX_ACTION_DT,
            "max_next_state_dt": MAX_NEXT_STATE_DT,
            "terminal_front_arc_fraction": TERMINAL_FRONT_ARC_FRACTION,
            "terminal_clearance_threshold_m": TERMINAL_CLEARANCE_THRESHOLD_M,
            "crash_terminal_ends_episode": True,
            "episode_split_unit": "terminal-separated segments within bag",
        },
        "reward_stats": {name: reward_statistics(values) for name, values in reward_columns.items()},
    }
    return arrays, metadata


def save_h5_dataset(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    infos_json: np.ndarray,
    output_path: Path,
) -> Path:
    if h5py is None:
        raise RuntimeError(
            "SAVE_H5_DATASETS=True but h5py is not installed. "
            "Install h5py in the processing environment before running this script."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as h5:
        for key, values in arrays.items():
            arr = np.asarray(values)
            if arr.dtype.kind in {"U", "S", "O"}:
                h5.create_dataset(key, data=arr.astype(string_dtype), dtype=string_dtype)
            else:
                h5.create_dataset(key, data=arr, compression="gzip")
        h5.create_dataset("infos_json", data=infos_json.astype(string_dtype), dtype=string_dtype)
        h5.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
    return output_path


def save_dataset(
    transitions: list[dict[str, Any]],
    output_path: Path,
    *,
    primary_reward: str,
    run_stats: list[dict[str, Any]],
    split_name: str,
    selection_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays, metadata = arrays_from_transitions(transitions, primary_reward=primary_reward)
    metadata["path"] = str(output_path)
    metadata["h5_path"] = str(output_path.with_suffix(".h5")) if SAVE_H5_DATASETS else None
    metadata["run_stats"] = run_stats
    metadata["split_name"] = split_name
    metadata["episode_selection"] = selection_metadata
    infos_json = np.asarray(
        [json.dumps(item["info"], sort_keys=True) for item in transitions],
        dtype=np.str_,
    )
    if SAVE_NPZ_DATASETS:
        np.savez_compressed(
            output_path,
            **arrays,
            infos_json=infos_json,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
    if SAVE_H5_DATASETS:
        save_h5_dataset(arrays, metadata, infos_json, output_path.with_suffix(".h5"))
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def save_all_outputs(transitions: list[dict[str, Any]], run_stats: list[dict[str, Any]]) -> list[Path]:
    saved: list[Path] = []
    if SAVE_ALL_REWARDS_DATASET:
        out = ALL_REWARDS_OUTPUT_DIR / f"real_ros_all_rewards_lidar_{LIDAR_BEAMS}.npz"
        save_dataset(
            transitions,
            out,
            primary_reward=PRIMARY_REWARD_FOR_ALL_DATASET,
            run_stats=run_stats,
            split_name="all_rewards",
            selection_metadata=None,
        )
        if SAVE_NPZ_DATASETS:
            saved.append(out)
        if SAVE_H5_DATASETS:
            saved.append(out.with_suffix(".h5"))
    if SAVE_PER_REWARD_DATASETS:
        for reward_name in REWARD_NAMES:
            if SAVE_FULL_EPISODE_DATASETS:
                out = output_dir_for_reward(reward_name) / f"real_ros_{reward_name}_all_episodes_lidar_{LIDAR_BEAMS}.npz"
                save_dataset(
                    transitions,
                    out,
                    primary_reward=reward_name,
                    run_stats=run_stats,
                    split_name="all_episodes",
                    selection_metadata=None,
                )
                if SAVE_NPZ_DATASETS:
                    saved.append(out)
                if SAVE_H5_DATASETS:
                    saved.append(out.with_suffix(".h5"))

            if SAVE_QUALITY_SPLIT_DATASETS:
                split_transitions, selection_metadata = split_top_episode_transitions(
                    transitions,
                    reward_name=reward_name,
                )
                for split_name in QUALITY_SPLIT_NAMES:
                    selected_transitions = split_transitions.get(split_name, [])
                    if not selected_transitions:
                        print(f"Skipping empty split {reward_name}/{split_name}")
                        continue
                    out = (
                        output_dir_for_reward(reward_name, split_name)
                        / f"real_ros_{reward_name}_{split_name}_top60_lidar_{LIDAR_BEAMS}.npz"
                    )
                    save_dataset(
                        selected_transitions,
                        out,
                        primary_reward=reward_name,
                        run_stats=run_stats,
                        split_name=split_name,
                        selection_metadata=selection_metadata,
                    )
                    if SAVE_NPZ_DATASETS:
                        saved.append(out)
                    if SAVE_H5_DATASETS:
                        saved.append(out.with_suffix(".h5"))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dirs": {
            reward_name: {
                "base": str(output_dir_for_reward(reward_name)),
                **{
                    split_name: str(output_dir_for_reward(reward_name, split_name))
                    for split_name in QUALITY_SPLIT_NAMES
                },
            }
            for reward_name in REWARD_NAMES
        },
        "saved_files": [str(path) for path in saved],
        "num_transitions": len(transitions),
        "reward_names": REWARD_NAMES,
        "lidar_beams": LIDAR_BEAMS,
        "state_aux_feature": STATE_AUX_FEATURE,
        "action_components": ["speed", "steering"],
        "training_steering_clamp_rad": TRAINING_STEERING_CLAMP_RAD,
        "crash_terminal_ends_episode": True,
        "episode_split_unit": "terminal-separated segments within bag",
        "episode_selection_fraction": EPISODE_SELECTION_FRACTION,
        "quality_split_names": QUALITY_SPLIT_NAMES,
    }
    manifest_path = OUTPUT_ROOT / f"processed_reward_manifest_lidar_{LIDAR_BEAMS}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    saved.append(manifest_path)
    return saved


# %%
# ---------------------------------------------------------------------------
# Run all bags
# ---------------------------------------------------------------------------


def run_processing() -> list[Path]:
    global LIDAR_BEAMS

    if SAVE_H5_DATASETS and h5py is None:
        raise RuntimeError(
            "SAVE_H5_DATASETS=True but h5py is not installed. "
            "Use the f110_data_processing environment or install h5py."
        )
    bag_roots = [Path(root) for root in (BAG_ROOTS or [BAG_ROOT])]
    missing_roots = [root for root in bag_roots if not root.exists()]
    if missing_roots:
        raise FileNotFoundError(
            "BAG_ROOTS contain missing paths: "
            + ", ".join(str(root) for root in missing_roots)
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if SAVE_ALL_REWARDS_DATASET:
        ALL_REWARDS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for reward_name in REWARD_NAMES:
        output_dir_for_reward(reward_name).mkdir(parents=True, exist_ok=True)
        if SAVE_QUALITY_SPLIT_DATASETS:
            for split_name in QUALITY_SPLIT_NAMES:
                output_dir_for_reward(reward_name, split_name).mkdir(parents=True, exist_ok=True)
    bag_dirs_by_path: dict[Path, Path] = {}
    for root in bag_roots:
        for bag_dir in find_bag_dirs(root):
            bag_dirs_by_path[bag_dir.resolve()] = bag_dir
    bag_dirs = sorted(bag_dirs_by_path.values())
    if MAX_BAGS is not None:
        bag_dirs = bag_dirs[: int(MAX_BAGS)]
    print(f"Found {len(bag_dirs)} bag folder(s).")
    print("Inputs:")
    for root in bag_roots:
        print("  ", root)
    print("Output root:", OUTPUT_ROOT)
    print("Outputs:")
    for reward_name in REWARD_NAMES:
        if SAVE_QUALITY_SPLIT_DATASETS:
            for split_name in QUALITY_SPLIT_NAMES:
                print(f"  {reward_name}/{split_name}: {output_dir_for_reward(reward_name, split_name)}")
        if SAVE_FULL_EPISODE_DATASETS:
            print(f"  {reward_name}/all_episodes: {output_dir_for_reward(reward_name)}")
    if not bag_dirs:
        raise RuntimeError("No bag folders found. Check BAG_ROOT and ONLY_BAGS.")

    typestore = make_typestore()
    all_transitions: list[dict[str, Any]] = []
    run_stats: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    next_episode_id = 0
    for bag_index, bag_dir in enumerate(bag_dirs):
        print(f"\n=== Processing bag {bag_index + 1}/{len(bag_dirs)}: {bag_dir.name} ===")
        try:
            transitions, stats = process_bag_dir(bag_dir, typestore, episode_id=next_episode_id)
            all_transitions.extend(transitions)
            run_stats.append(stats)
            if transitions:
                next_episode_id = max(int(item["episode_id"]) for item in transitions) + 1
            print(
                f"kept={len(transitions)} "
                f"lidar={stats.get('lidar_kept', 0)}/{stats.get('lidar_seen', 0)} "
                f"actions={stats.get('actions_kept', 0)}/{stats.get('actions_seen', 0)} "
                f"terminals={stats.get('transition_terminals', 0)} "
                f"episodes={stats.get('transition_episodes', 0)}"
            )
        except Exception as exc:
            print(f"FAILED {bag_dir.name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failures.append(
                {
                    "bag": bag_dir.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    if failures:
        failure_path = OUTPUT_ROOT / f"processed_reward_failures_lidar_{LIDAR_BEAMS}.json"
        failure_path.write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Failures written to {failure_path}")
    if not all_transitions:
        raise RuntimeError("No transitions were produced after filtering.")

    saved: list[Path] = []
    original_lidar_beams = int(LIDAR_BEAMS)
    for beams in output_lidar_beam_counts():
        LIDAR_BEAMS = beams
        refresh_transition_observations(all_transitions, beams)
        print(f"\nSaving datasets with lidar downsampled to {beams} beams")
        saved.extend(save_all_outputs(all_transitions, run_stats))
    LIDAR_BEAMS = original_lidar_beams

    print("\nSaved:")
    for path in saved:
        print("  ", path)
    print(f"Total transitions: {len(all_transitions)}")
    return saved


if __name__ == "__main__":
    run_processing()


# %%
# ---------------------------------------------------------------------------
# Sanity check helpers for interactive use
# ---------------------------------------------------------------------------


def load_npz_summary(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        rewards = data["rewards"]
        return {
            "path": str(path),
            "observations": tuple(data["observations"].shape),
            "actions": tuple(data["actions"].shape),
            "rewards_mean": float(np.mean(rewards)) if rewards.size else 0.0,
            "rewards_min": float(np.min(rewards)) if rewards.size else 0.0,
            "rewards_max": float(np.max(rewards)) if rewards.size else 0.0,
            "dones": int(np.sum(data["dones"])),
            "finite_observations": bool(np.all(np.isfinite(data["observations"]))),
            "finite_actions": bool(np.all(np.isfinite(data["actions"]))),
        }


def print_output_summaries() -> None:
    for output_dir in sorted(set(REWARD_OUTPUT_DIRS.values()) | {ALL_REWARDS_OUTPUT_DIR}):
        for path in sorted(output_dir.rglob("*.npz")):
            print(json.dumps(load_npz_summary(path), indent=2, sort_keys=True))
