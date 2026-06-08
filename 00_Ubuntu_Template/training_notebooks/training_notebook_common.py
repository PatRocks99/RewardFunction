from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


NOTEBOOK_DIR = Path(__file__).resolve().parent
ROOT = NOTEBOOK_DIR.parent
CONTROLLERS = ROOT / "Controllers"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS))

from offline_common import DeterministicActor, GaussianActor, load_h5_transitions  # noqa: E402
from simulation_eval_placeholder import simulation_evaluation_placeholder  # noqa: E402


DATA_ROOT = Path(os.environ.get("F110_DATA_ROOT", "/mnt/p/Car/NewCar"))
REWARD_DATASET_DIRS = {
    "reward_v1": Path(
        os.environ.get("F110_REWARD_V1_DIR", str(DATA_ROOT / "processed_reward_V1_datasets"))
    ),
    "reward_v2": Path(
        os.environ.get("F110_REWARD_V2_DIR", str(DATA_ROOT / "processed_reward_V2_datasets"))
    ),
    "reward_v3": Path(
        os.environ.get("F110_REWARD_V3_DIR", str(DATA_ROOT / "processed_reward_V3_datasets"))
    ),
}

SELECTED_DATASET_ROOT = Path(
    os.environ.get("F110_SELECTED_DATASET_ROOT", str(DATA_ROOT / "selected_training_datasets"))
)


FINAL_NAMES = {
    "bc": "final_bc.pt",
    "awac": "final_awac.pt",
    "awac_bc_like": "final_awac.pt",
    "awac_aggressive": "final_awac.pt",
    "iql": "final_iql.pt",
    "td3bc": "final_td3_bc.pt",
}

GAUSSIAN_AWAC_ALGORITHMS = {"awac", "awac_bc_like", "awac_aggressive"}


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def dataset_folder(
    reward_dataset: str,
    split: str,
    lidar_beams: int,
    *,
    data_roots: dict[str, Path] | None = None,
) -> Path:
    roots = data_roots or REWARD_DATASET_DIRS
    base = roots[reward_dataset]
    direct = base / split
    nested = direct / str(lidar_beams)
    source_dirs = [nested, direct] if nested.exists() else [direct]
    selected_files: list[Path] = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        selected_files.extend(sorted(source_dir.glob(f"*lidar_{lidar_beams}.h5")))
        selected_files.extend(sorted(source_dir.glob(f"*lidar_{lidar_beams}.hdf5")))
        selected_files.extend(sorted(source_dir.glob(f"*lidar_{lidar_beams}.npz")))
    if not selected_files:
        raise FileNotFoundError(
            f"No lidar_{lidar_beams} dataset files found under {direct}"
        )
    selected = SELECTED_DATASET_ROOT / f"{reward_dataset}_{split}_lidar_{lidar_beams}"
    selected.mkdir(parents=True, exist_ok=True)
    for existing in selected.glob("*"):
        if existing.is_file():
            existing.unlink()
    for source in selected_files:
        target = selected / source.name
        try:
            os.link(source, target)
        except OSError:
            try:
                target.symlink_to(source)
            except OSError:
                shutil.copy2(source, target)
    return selected


def import_trainer(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import trainer at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def transition_count(data_folder: Path) -> int:
    files = sorted(data_folder.glob("*.h5")) + sorted(data_folder.glob("*.hdf5"))
    if files:
        total = 0
        for path in files:
            with h5py.File(path, "r") as h5:
                total += int(h5["observations"].shape[0])
        return total
    files = sorted(data_folder.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .h5/.hdf5/.npz dataset files found in {data_folder}")
    total = 0
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            total += int(data["observations"].shape[0])
    return total


def steps_for_epochs(num_transitions: int, batch_size: int, epochs: int) -> int:
    return int(math.ceil(num_transitions / float(batch_size)) * epochs)


def config_dict(config: Any) -> dict[str, Any]:
    return asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config)


def train_once(
    *,
    algorithm: str,
    trainer: Any,
    data_folder: Path,
    checkpoint_dir: Path,
    params: dict[str, Any],
    epochs: int,
    device: str,
    seed: int,
    eval_epochs: int = 1,
) -> dict[str, Any]:
    num_transitions = transition_count(data_folder)
    batch_size = int(params.get("batch_size", trainer.Config.batch_size))
    max_steps = steps_for_epochs(num_transitions, batch_size, epochs)
    eval_freq = max(1, steps_for_epochs(num_transitions, batch_size, eval_epochs))
    config_values = {
        **params,
        "data_folder": str(data_folder),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device,
        "seed": seed,
        "max_steps": max_steps,
        "eval_freq": eval_freq,
        "batch_size": batch_size,
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = trainer.Config(**config_values)
    started = time.perf_counter()
    final_path = trainer.train(config)
    train_seconds = time.perf_counter() - started
    metrics = evaluate_checkpoint(
        algorithm=algorithm,
        checkpoint_path=Path(final_path),
        config=config_dict(config),
        data_folder=data_folder,
    )
    result = {
        "algorithm": algorithm,
        "data_folder": str(data_folder),
        "checkpoint_dir": str(checkpoint_dir),
        "final_path": str(final_path),
        "epochs": epochs,
        "max_steps": max_steps,
        "train_seconds": train_seconds,
        "params": config_dict(config),
        **metrics,
    }
    (checkpoint_dir / "offline_eval.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_with_stats(dataset: dict[str, np.ndarray], stats: dict[str, Any]) -> dict[str, np.ndarray]:
    normalized = {key: np.asarray(value).copy() for key, value in dataset.items()}
    obs_mean = stats.get("obs_mean")
    obs_std = stats.get("obs_std")
    if obs_mean is not None and obs_std is not None:
        normalized["observations"] = (
            normalized["observations"] - np.asarray(obs_mean)
        ) / np.asarray(obs_std)
        normalized["next_observations"] = (
            normalized["next_observations"] - np.asarray(obs_mean)
        ) / np.asarray(obs_std)
    action_mean = stats.get("action_mean")
    action_std = stats.get("action_std")
    if action_mean is not None and action_std is not None:
        normalized["actions"] = (normalized["actions"] - np.asarray(action_mean)) / np.asarray(action_std)
    return normalized


def build_actor(algorithm: str, checkpoint: dict[str, Any], config: dict[str, Any], obs_dim: int, action_dim: int):
    hidden_dim = int(config.get("hidden_dim", 256))
    hidden_layers = int(config.get("hidden_layers", 2))
    if algorithm in GAUSSIAN_AWAC_ALGORITHMS or (
        algorithm == "iql" and not bool(config.get("deterministic_actor", False))
    ):
        actor = GaussianActor(obs_dim, action_dim, hidden_dim, hidden_layers)
    else:
        actor = DeterministicActor(obs_dim, action_dim, hidden_dim, hidden_layers)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor


def actor_mean_action(actor, observations: torch.Tensor) -> torch.Tensor:
    if hasattr(actor, "deterministic_action"):
        return actor.deterministic_action(observations)
    output = actor(observations)
    if isinstance(output, torch.distributions.Distribution):
        return output.mean.clamp(-1.0, 1.0)
    return output.clamp(-1.0, 1.0)


def evaluate_checkpoint(
    *,
    algorithm: str,
    checkpoint_path: Path,
    config: dict[str, Any],
    data_folder: Path,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    raw_dataset = load_h5_transitions(str(data_folder))
    dataset = normalize_with_stats(raw_dataset, checkpoint.get("stats", {}))
    observations = torch.as_tensor(dataset["observations"], dtype=torch.float32)
    target_actions = torch.as_tensor(dataset["actions"], dtype=torch.float32)
    actor = build_actor(algorithm, checkpoint, config, observations.shape[1], target_actions.shape[1])
    predictions: list[torch.Tensor] = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(observations), 4096):
            predictions.append(actor_mean_action(actor, observations[start : start + 4096]))
    inference_seconds = time.perf_counter() - started
    policy_actions = torch.cat(predictions, dim=0)
    diff = policy_actions - target_actions
    rewards = raw_dataset["rewards"].reshape(-1)
    terminals = raw_dataset["terminals"].reshape(-1).astype(bool)
    metrics = {
        "offline_action_mse": float(torch.mean(diff.pow(2)).item()),
        "offline_action_mae": float(torch.mean(torch.abs(diff)).item()),
        "mean_policy_speed": float(torch.mean(policy_actions[:, 0]).item()),
        "mean_abs_policy_steering": float(torch.mean(torch.abs(policy_actions[:, 1])).item()),
        "dataset_reward_mean": float(np.mean(rewards)) if rewards.size else 0.0,
        "terminal_rate": float(np.mean(terminals)) if terminals.size else 0.0,
        "inference_ms_per_sample": 1000.0 * inference_seconds / max(1, len(observations)),
    }
    metrics.update(
        simulation_evaluation_placeholder(
            checkpoint_path=checkpoint_path,
            algorithm=algorithm,
            lidar_beams=max(0, observations.shape[1] - 3),
        )
    )
    return metrics


def run_hyperparameter_search(
    *,
    algorithm: str,
    trainer: Any,
    data_folder: Path,
    output_root: Path,
    base_params: dict[str, Any],
    search_space: list[dict[str, Any]],
    quick_epochs: int,
    device: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    for index, overrides in enumerate(search_space, start=1):
        params = {**base_params, **overrides}
        checkpoint_dir = output_root / "search" / f"candidate_{index:02d}"
        print(f"\n=== {algorithm.upper()} candidate {index}/{len(search_space)} ===")
        print(json.dumps(params, indent=2, sort_keys=True))
        result = train_once(
            algorithm=algorithm,
            trainer=trainer,
            data_folder=data_folder,
            checkpoint_dir=checkpoint_dir,
            params=params,
            epochs=quick_epochs,
            device=device,
            seed=seed,
        )
        result["candidate_overrides"] = overrides
        results.append(result)
    best = min(results, key=lambda item: float(item["offline_action_mse"]))
    (output_root / "search_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "best_search_result.json").write_text(
        json.dumps(best, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**base_params, **best["candidate_overrides"]}, results


def train_deploy_ready(
    *,
    algorithm: str,
    trainer: Any,
    data_folder: Path,
    output_root: Path,
    params: dict[str, Any],
    final_epochs: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    checkpoint_dir = output_root / "deploy_ready"
    print(f"\n=== Final deploy-ready {algorithm.upper()} training ===")
    print(json.dumps(params, indent=2, sort_keys=True))
    result = train_once(
        algorithm=algorithm,
        trainer=trainer,
        data_folder=data_folder,
        checkpoint_dir=checkpoint_dir,
        params=params,
        epochs=final_epochs,
        device=device,
        seed=seed,
    )
    (output_root / "deploy_ready_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
