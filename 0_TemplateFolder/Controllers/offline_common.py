from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import h5py
except ImportError:
    h5py = None


OBS_KEYS = ["observations", "obs", "states", "state", "o", "x", "lidar"]
NEXT_OBS_KEYS = ["next_observations", "next_obs", "next_states"]
ACTION_KEYS = ["actions", "action", "act", "a", "u", "controls", "t_action", "steering_angle"]
REWARD_KEYS = ["rewards", "reward", "r"]
DONE_KEYS = ["terminals", "dones", "done", "timeouts"]


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_h5_key(h5: h5py.File, candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in h5:
            return key

    def walk(group: h5py.Group, prefix: str = "") -> Optional[str]:
        for name, item in group.items():
            path = f"{prefix}/{name}" if prefix else name
            if isinstance(item, h5py.Dataset) and name in candidates:
                return path
            if isinstance(item, h5py.Group):
                found = walk(item, path)
                if found is not None:
                    return found
        return None

    return walk(h5)


def read_h5_array(h5: h5py.File, path: str) -> np.ndarray:
    item = h5
    for part in path.split("/"):
        item = item[part]
    return np.asarray(item)


def find_npz_key(npz: np.lib.npyio.NpzFile, candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in npz.files:
            return key
    return None


def as_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        return array[:, None]
    return array


def compute_next_observations(observations: np.ndarray, terminals: np.ndarray) -> np.ndarray:
    next_observations = observations.copy()
    if len(observations) > 1:
        next_observations[:-1] = observations[1:]
        next_observations[np.asarray(terminals).reshape(-1).astype(bool)] = observations[
            np.asarray(terminals).reshape(-1).astype(bool)
        ]
    return next_observations


def dataset_paths(data_folder: str) -> List[Path]:
    folder = Path(data_folder)
    paths = sorted(folder.glob("*.h5"))
    paths.extend(sorted(folder.glob("*.hdf5")))
    h5_stems = {path.stem for path in paths}
    paths.extend(path for path in sorted(folder.glob("*.npz")) if path.stem not in h5_stems)
    if not paths:
        raise FileNotFoundError(f"No .h5, .hdf5, or .npz files found in {folder}")
    return paths


def h5_paths(data_folder: str) -> List[Path]:
    return [path for path in dataset_paths(data_folder) if path.suffix.lower() in {".h5", ".hdf5"}]


def load_npz_transitions(
    path: Path,
    obs_key: Optional[str] = None,
    action_key: Optional[str] = None,
    reward_key: Optional[str] = None,
    done_key: Optional[str] = None,
    next_obs_key: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        ok = obs_key or find_npz_key(data, OBS_KEYS)
        ak = action_key or find_npz_key(data, ACTION_KEYS)
        rk = reward_key or find_npz_key(data, REWARD_KEYS)
        dk = done_key or find_npz_key(data, DONE_KEYS)
        nok = next_obs_key or find_npz_key(data, NEXT_OBS_KEYS)
        if ok is None or ak is None:
            raise KeyError(f"{path} is missing observation or action keys. Keys: {data.files}")

        observations = as_2d(data[ok]).astype(np.float32)
        actions = as_2d(data[ak]).astype(np.float32)
        rewards = (
            np.asarray(data[rk]).reshape(-1).astype(np.float32)
            if rk is not None
            else np.zeros(len(observations), dtype=np.float32)
        )
        terminals = (
            np.asarray(data[dk]).reshape(-1).astype(np.float32)
            if dk is not None
            else np.zeros(len(observations), dtype=np.float32)
        )
        next_observations = (
            as_2d(data[nok]).astype(np.float32)
            if nok is not None
            else compute_next_observations(observations, terminals)
        )

    n = min(len(observations), len(actions), len(rewards), len(terminals), len(next_observations))
    return {
        "observations": observations[:n],
        "actions": actions[:n],
        "rewards": rewards[:n],
        "terminals": terminals[:n],
        "next_observations": next_observations[:n],
    }


def load_h5_transitions(
    data_folder: str,
    obs_key: Optional[str] = None,
    action_key: Optional[str] = None,
    reward_key: Optional[str] = None,
    done_key: Optional[str] = None,
    next_obs_key: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Load offline transition files.

    The name is kept for compatibility with the existing controller scripts, but
    the loader now accepts both legacy HDF5 files and rosbag-generated NPZ files.
    """
    obs_list, action_list, reward_list, done_list, next_obs_list = [], [], [], [], []
    paths = dataset_paths(data_folder)
    for path in paths:
        if path.suffix.lower() == ".npz":
            loaded = load_npz_transitions(path, obs_key, action_key, reward_key, done_key, next_obs_key)
            observations = loaded["observations"]
            actions = loaded["actions"]
            rewards = loaded["rewards"]
            terminals = loaded["terminals"]
            next_observations = loaded["next_observations"]
        else:
            if h5py is None:
                raise ImportError(
                    f"Reading {path.suffix} files requires h5py. Install h5py or use .npz datasets."
                )
            with h5py.File(path, "r") as h5:
                ok = obs_key or find_h5_key(h5, OBS_KEYS)
                ak = action_key or find_h5_key(h5, ACTION_KEYS)
                rk = reward_key or find_h5_key(h5, REWARD_KEYS)
                dk = done_key or find_h5_key(h5, DONE_KEYS)
                nok = next_obs_key or find_h5_key(h5, NEXT_OBS_KEYS)
                if ok is None or ak is None:
                    raise KeyError(f"{path} is missing observation or action keys. Top-level keys: {list(h5.keys())}")

                observations = as_2d(read_h5_array(h5, ok)).astype(np.float32)
                actions = as_2d(read_h5_array(h5, ak)).astype(np.float32)
                rewards = (
                    np.asarray(read_h5_array(h5, rk)).reshape(-1).astype(np.float32)
                    if rk is not None
                    else np.zeros(len(observations), dtype=np.float32)
                )
                terminals = (
                    np.asarray(read_h5_array(h5, dk)).reshape(-1).astype(np.float32)
                    if dk is not None
                    else np.zeros(len(observations), dtype=np.float32)
                )
                next_observations = (
                    as_2d(read_h5_array(h5, nok)).astype(np.float32)
                    if nok is not None
                    else compute_next_observations(observations, terminals)
                )

        n = min(len(observations), len(actions), len(rewards), len(terminals), len(next_observations))
        obs_list.append(observations[:n])
        action_list.append(actions[:n])
        reward_list.append(rewards[:n])
        done_list.append(terminals[:n])
        next_obs_list.append(next_observations[:n])

    dataset = {
        "observations": np.concatenate(obs_list, axis=0),
        "actions": np.concatenate(action_list, axis=0),
        "rewards": np.concatenate(reward_list, axis=0),
        "terminals": np.concatenate(done_list, axis=0),
        "next_observations": np.concatenate(next_obs_list, axis=0),
    }
    print(f"Loaded {len(dataset['observations'])} transitions from {len(paths)} files.")
    return dataset


def normalize_dataset(dataset: Dict[str, np.ndarray], normalize_obs: bool, normalize_actions: bool) -> Dict[str, np.ndarray]:
    stats = {
        "obs_mean": None,
        "obs_std": None,
        "action_mean": None,
        "action_std": None,
    }
    if normalize_obs:
        stats["obs_mean"] = dataset["observations"].mean(axis=0, keepdims=True)
        stats["obs_std"] = dataset["observations"].std(axis=0, keepdims=True) + 1e-6
        dataset["observations"] = (dataset["observations"] - stats["obs_mean"]) / stats["obs_std"]
        dataset["next_observations"] = (dataset["next_observations"] - stats["obs_mean"]) / stats["obs_std"]
    if normalize_actions:
        stats["action_mean"] = dataset["actions"].mean(axis=0, keepdims=True)
        stats["action_std"] = dataset["actions"].std(axis=0, keepdims=True) + 1e-6
        dataset["actions"] = (dataset["actions"] - stats["action_mean"]) / stats["action_std"]
    return stats


class ReplayBuffer:
    def __init__(self, dataset: Dict[str, np.ndarray], device: str = "cpu"):
        self.device = torch.device(device)
        self.observations = torch.as_tensor(dataset["observations"], dtype=torch.float32, device=self.device)
        self.actions = torch.as_tensor(dataset["actions"], dtype=torch.float32, device=self.device)
        self.rewards = torch.as_tensor(dataset["rewards"], dtype=torch.float32, device=self.device).view(-1, 1)
        self.terminals = torch.as_tensor(dataset["terminals"], dtype=torch.float32, device=self.device).view(-1, 1)
        self.next_observations = torch.as_tensor(dataset["next_observations"], dtype=torch.float32, device=self.device)
        self.size = self.observations.shape[0]
        self.obs_dim = self.observations.shape[1]
        self.action_dim = self.actions.shape[1]

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.terminals[indices],
        )


class MLP(nn.Module):
    def __init__(self, dims: Sequence[int], activation=nn.ReLU, output_activation=None):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation is not None:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, hidden_layers: int = 2):
        super().__init__()
        self.net = MLP([obs_dim, *([hidden_dim] * hidden_layers), action_dim], output_activation=nn.Tanh)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, hidden_layers: int = 2):
        super().__init__()
        self.mean = MLP([obs_dim, *([hidden_dim] * hidden_layers), action_dim])
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.distributions.Normal:
        std = torch.exp(self.log_std.clamp(-20.0, 2.0))
        return torch.distributions.Normal(self.mean(obs), std)

    def raw_to_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        # Action convention: speed is [0, 1]; remaining controls are [-1, 1].
        speed = torch.sigmoid(raw_action[..., :1])
        if raw_action.shape[-1] == 1:
            return speed
        steering = torch.tanh(raw_action[..., 1:])
        return torch.cat([speed, steering], dim=-1)

    def action_to_raw(self, action: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        speed = action[..., :1].clamp(eps, 1.0 - eps)
        raw_speed = torch.logit(speed)
        if action.shape[-1] == 1:
            return raw_speed
        steering = action[..., 1:].clamp(-1.0 + eps, 1.0 - eps)
        raw_steering = torch.atanh(steering)
        return torch.cat([raw_speed, raw_steering], dim=-1)

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        raw_action = self.action_to_raw(action, eps)
        transformed_action = self.raw_to_action(raw_action)
        dist = self(obs)
        speed = transformed_action[..., :1].clamp(eps, 1.0 - eps)
        correction = torch.log(speed * (1.0 - speed) + eps)
        if transformed_action.shape[-1] > 1:
            steering = transformed_action[..., 1:].clamp(-1.0 + eps, 1.0 - eps)
            correction = torch.cat([correction, torch.log(1.0 - steering.pow(2) + eps)], dim=-1)
        return (dist.log_prob(raw_action) - correction).sum(dim=-1, keepdim=True)

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.raw_to_action(self.mean(obs))

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self(obs)
        raw_action = dist.rsample()
        action = self.raw_to_action(raw_action)
        speed = action[..., :1]
        correction = torch.log(speed * (1.0 - speed) + 1e-6)
        if action.shape[-1] > 1:
            correction = torch.cat([correction, torch.log(1.0 - action[..., 1:].pow(2) + 1e-6)], dim=-1)
        log_prob = dist.log_prob(raw_action) - correction
        return action, log_prob.sum(dim=-1, keepdim=True)


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, hidden_layers: int = 2):
        super().__init__()
        dims = [obs_dim + action_dim, *([hidden_dim] * hidden_layers), 1]
        self.q1 = MLP(dims)
        self.q2 = MLP(dims)

    def both(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.both(obs, action)
        return torch.min(q1, q2)


class ValueFunction(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 256, hidden_layers: int = 2):
        super().__init__()
        self.net = MLP([obs_dim, *([hidden_dim] * hidden_layers), 1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1.0 - tau) * target_param.data + tau * source_param.data)


def prepare_checkpoint_dir(path: str, config) -> Path:
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(config) if is_dataclass(config) else dict(config)
    with open(checkpoint_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return checkpoint_dir


def save_checkpoint(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
