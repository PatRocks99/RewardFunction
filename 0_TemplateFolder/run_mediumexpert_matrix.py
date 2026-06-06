from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

CONTROLLERS_DIR = Path(__file__).resolve().parent / "Controllers"
if str(CONTROLLERS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_DIR))

from offline_common import DeterministicActor, GaussianActor, load_h5_transitions  # noqa: E402
from simulation_eval_placeholder import simulation_evaluation_placeholder  # noqa: E402


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "reward_v1": ROOT / "processed_reward_V1_datasets" / "mediumexpert",
    "reward_v2": ROOT / "processed_reward_V2_datasets" / "mediumexpert",
    "reward_v3": ROOT / "processed_reward_V3_datasets" / "mediumexpert",
}
ALGORITHMS = {
    "awac": {
        "script": ROOT / "Controllers" / "AWAC" / "train_awac.py",
        "final": "final_awac.pt",
        "batch_size": 256,
        "base_args": {
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "awac_lambda": 1.0,
            "max_weight": 20.0,
            "discount": 0.99,
            "normalize_obs": True,
            "normalize_actions": False,
        },
        "search": [
            {"awac_lambda": 0.3},
            {"awac_lambda": 1.0},
            {"awac_lambda": 3.0},
        ],
    },
    "bc": {
        "script": ROOT / "Controllers" / "BC" / "train_bc.py",
        "final": "final_bc.pt",
        "batch_size": 1024,
        "base_args": {
            "lr": 3e-4,
            "normalize_obs": True,
            "normalize_actions": False,
        },
        "search": [
            {"lr": 1e-4},
            {"lr": 3e-4},
            {"lr": 1e-3},
        ],
    },
    "iql": {
        "script": ROOT / "Controllers" / "IQL" / "train_iql.py",
        "final": "final_iql.pt",
        "batch_size": 256,
        "base_args": {
            "actor_lr": 3e-4,
            "qf_lr": 3e-4,
            "vf_lr": 3e-4,
            "iql_tau": 0.7,
            "beta": 3.0,
            "discount": 0.99,
            "normalize_obs": True,
            "normalize_actions": False,
        },
        "search": [
            {"iql_tau": 0.7, "beta": 1.0},
            {"iql_tau": 0.7, "beta": 3.0},
            {"iql_tau": 0.8, "beta": 3.0},
        ],
    },
    "td3bc": {
        "script": ROOT / "Controllers" / "TD3+BC" / "train_td3_bc.py",
        "final": "final_td3_bc.pt",
        "batch_size": 512,
        "base_args": {
            "actor_lr": 3e-4,
            "critic_lr": 1e-4,
            "alpha": 2.5,
            "discount": 0.99,
            "normalize_obs": True,
            "normalize_actions": False,
        },
        "search": [
            {"alpha": 1.0},
            {"alpha": 2.5},
            {"alpha": 5.0},
        ],
    },
}


FOLDER5_BEST_HPARAMS: dict[tuple[str, str], dict[str, Any]] = {
    ("reward_v1", "awac"): {"actor_lr": 3e-4, "critic_lr": 3e-4, "awac_lambda": 0.3, "discount": 0.99, "normalize_obs": True, "normalize_actions": False},
    ("reward_v2", "awac"): {"actor_lr": 3e-4, "critic_lr": 3e-4, "awac_lambda": 3.0, "discount": 0.99, "normalize_obs": True, "normalize_actions": False},
    ("reward_v3", "awac"): {"actor_lr": 3e-4, "critic_lr": 3e-4, "awac_lambda": 3.0, "discount": 0.99, "normalize_obs": True, "normalize_actions": False},
    ("reward_v1", "bc"): {"lr": 1e-3, "normalize_obs": True, "normalize_actions": False},
    ("reward_v2", "bc"): {"lr": 1e-3, "normalize_obs": True, "normalize_actions": False},
    ("reward_v3", "bc"): {"lr": 1e-3, "normalize_obs": True, "normalize_actions": False},
    ("reward_v1", "iql"): {"actor_lr": 3e-4, "qf_lr": 3e-4, "vf_lr": 3e-4, "iql_tau": 0.8, "beta": 3.0, "discount": 0.99, "normalize_obs": True, "normalize_actions": False},
    ("reward_v2", "iql"): {"actor_lr": 3e-4, "qf_lr": 3e-4, "vf_lr": 3e-4, "iql_tau": 0.8, "beta": 3.0, "discount": 0.99, "normalize_obs": True, "normalize_actions": False},
    ("reward_v3", "iql"): {"actor_lr": 3e-4, "qf_lr": 3e-4, "vf_lr": 3e-4, "iql_tau": 0.8, "beta": 3.0, "discount": 0.99, "normalize_obs": True, "normalize_actions": False},
    ("reward_v1", "td3bc"): {"actor_lr": 3e-4, "critic_lr": 1e-4, "alpha": 1.0, "discount": 0.99, "normalize_obs": True, "normalize_actions": False},
    ("reward_v2", "td3bc"): {"actor_lr": 3e-4, "critic_lr": 1e-4, "alpha": 1.0, "discount": 0.99, "normalize_obs": True, "normalize_actions": False},
    ("reward_v3", "td3bc"): {"actor_lr": 3e-4, "critic_lr": 1e-4, "alpha": 1.0, "discount": 0.99, "normalize_obs": True, "normalize_actions": False},
}


@dataclass
class RunSpec:
    dataset_name: str
    algorithm: str
    data_folder: Path
    checkpoint_dir: Path
    max_steps: int
    eval_freq: int
    batch_size: int
    seed: int
    args: dict[str, Any]


def transition_count(data_folder: Path) -> int:
    h5_files = sorted(data_folder.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 dataset file found in {data_folder}")
    total = 0
    for path in h5_files:
        with h5py.File(path, "r") as h5:
            total += int(h5["observations"].shape[0])
    return total


def steps_for_epochs(num_transitions: int, batch_size: int, epochs: int) -> int:
    return int(math.ceil(num_transitions / float(batch_size)) * epochs)


def bool_flag(name: str, value: bool) -> list[str]:
    return [f"--{name.replace('_', '-')}"] if value else [f"--no-{name.replace('_', '-')}"]


def value_arg(name: str, value: Any) -> list[str]:
    if isinstance(value, bool):
        return bool_flag(name, value)
    return [f"--{name.replace('_', '-')}", str(value)]


def build_command(spec: RunSpec) -> list[str]:
    algo = ALGORITHMS[spec.algorithm]
    cmd = [
        sys.executable,
        str(algo["script"]),
        "--data-folder",
        str(spec.data_folder),
        "--checkpoint-dir",
        str(spec.checkpoint_dir),
        "--device",
        "cuda" if torch.cuda.is_available() else "cpu",
        "--seed",
        str(spec.seed),
        "--max-steps",
        str(spec.max_steps),
        "--batch-size",
        str(spec.batch_size),
        "--eval-freq",
        str(spec.eval_freq),
    ]
    for key, value in spec.args.items():
        cmd.extend(value_arg(key, value))
    return cmd


def run_training(spec: RunSpec) -> dict[str, Any]:
    spec.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = spec.checkpoint_dir / "train.log"
    cmd = build_command(spec)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = proc.wait()
    elapsed = time.perf_counter() - started
    if return_code != 0:
        raise RuntimeError(f"Training failed for {spec.dataset_name}/{spec.algorithm}; see {log_path}")
    return {
        "train_log": str(log_path),
        "train_seconds": elapsed,
        "last_training_metrics": parse_last_metrics(log_path),
    }


METRIC_RE = re.compile(r"([a-zA-Z_]+)=(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")


def parse_last_metrics(log_path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        found = {key: float(value) for key, value in METRIC_RE.findall(line)}
        if found:
            metrics = found
    return metrics


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_with_saved_stats(dataset: dict[str, np.ndarray], stats: dict[str, Any]) -> dict[str, np.ndarray]:
    out = {key: np.asarray(value).copy() for key, value in dataset.items()}
    obs_mean = stats.get("obs_mean")
    obs_std = stats.get("obs_std")
    if obs_mean is not None and obs_std is not None:
        out["observations"] = (out["observations"] - np.asarray(obs_mean)) / np.asarray(obs_std)
        out["next_observations"] = (out["next_observations"] - np.asarray(obs_mean)) / np.asarray(obs_std)
    action_mean = stats.get("action_mean")
    action_std = stats.get("action_std")
    if action_mean is not None and action_std is not None:
        out["actions"] = (out["actions"] - np.asarray(action_mean)) / np.asarray(action_std)
    return out


def build_actor(algorithm: str, checkpoint: dict[str, Any], obs_dim: int, action_dim: int, config: dict[str, Any]):
    hidden_dim = int(config.get("hidden_dim", 256))
    hidden_layers = int(config.get("hidden_layers", 2))
    if algorithm in {"awac"}:
        actor = GaussianActor(obs_dim, action_dim, hidden_dim, hidden_layers)
    elif algorithm == "iql" and not bool(config.get("deterministic_actor", False)):
        actor = GaussianActor(obs_dim, action_dim, hidden_dim, hidden_layers)
    else:
        actor = DeterministicActor(obs_dim, action_dim, hidden_dim, hidden_layers)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor


def actor_actions(actor, obs: torch.Tensor) -> torch.Tensor:
    if hasattr(actor, "deterministic_action"):
        return actor.deterministic_action(obs)
    out = actor(obs)
    if isinstance(out, torch.distributions.Distribution):
        return out.mean.clamp(-1.0, 1.0)
    return out.clamp(-1.0, 1.0)


def evaluate_checkpoint(
    dataset_name: str,
    algorithm: str,
    data_folder: Path,
    checkpoint_dir: Path,
    final_name: str,
    train_metadata: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_path = checkpoint_dir / final_name
    config_path = checkpoint_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    checkpoint = load_checkpoint(checkpoint_path)
    raw_dataset = load_h5_transitions(str(data_folder))
    dataset = normalize_with_saved_stats(raw_dataset, checkpoint.get("stats", {}))
    observations = torch.as_tensor(dataset["observations"], dtype=torch.float32)
    target_actions = torch.as_tensor(dataset["actions"], dtype=torch.float32)
    actor = build_actor(algorithm, checkpoint, observations.shape[1], target_actions.shape[1], config)

    preds: list[torch.Tensor] = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(observations), 4096):
            preds.append(actor_actions(actor, observations[start : start + 4096]))
    inference_seconds = time.perf_counter() - started
    policy_actions = torch.cat(preds, dim=0)

    diff = policy_actions - target_actions
    action_mse = float(torch.mean(diff.pow(2)).item())
    action_mae = float(torch.mean(torch.abs(diff)).item())
    mean_action_abs = float(torch.mean(torch.abs(policy_actions)).item())
    mean_speed = float(torch.mean(policy_actions[:, 0]).item())
    mean_abs_steering = float(torch.mean(torch.abs(policy_actions[:, 1])).item())
    if len(policy_actions) > 1:
        steering_smoothness = float(torch.mean(torch.abs(policy_actions[1:, 1] - policy_actions[:-1, 1])).item())
    else:
        steering_smoothness = 0.0

    rewards = raw_dataset["rewards"].reshape(-1)
    terminals = raw_dataset["terminals"].reshape(-1).astype(bool)
    episode_returns = []
    current_return = 0.0
    for reward, terminal in zip(rewards, terminals):
        current_return += float(reward)
        if terminal:
            episode_returns.append(current_return)
            current_return = 0.0
    if rewards.size and (not terminals.size or not terminals[-1]):
        episode_returns.append(current_return)
    episode_returns_arr = np.asarray(episode_returns, dtype=np.float32)

    metrics = {
        "dataset": dataset_name,
        "algorithm": algorithm,
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "train_log": train_metadata.get("train_log", ""),
        "best_hyperparameters": compact_hparams(config),
        "final_training_metrics": json.dumps(train_metadata.get("last_training_metrics", {}), sort_keys=True),
        "offline_action_mse": action_mse,
        "offline_action_mae": action_mae,
        "offline_behavior_episode_return_mean": float(np.mean(episode_returns_arr)) if episode_returns_arr.size else 0.0,
        "offline_behavior_episode_return_median": float(np.median(episode_returns_arr)) if episode_returns_arr.size else 0.0,
        "dataset_reward_mean": float(np.mean(rewards)) if rewards.size else 0.0,
        "terminal_rate": float(np.mean(terminals)) if terminals.size else 0.0,
        "crash_rate_or_terminal_rate": float(np.mean(terminals)) if terminals.size else 0.0,
        "mean_action_magnitude": mean_action_abs,
        "mean_policy_speed": mean_speed,
        "mean_abs_policy_steering": mean_abs_steering,
        "steering_smoothness": steering_smoothness,
        "inference_ms_per_sample": 1000.0 * inference_seconds / max(len(observations), 1),
        "train_seconds": float(train_metadata.get("train_seconds", 0.0)),
    }
    metrics.update(
        simulation_evaluation_placeholder(
            checkpoint_path=checkpoint_path,
            dataset_name=dataset_name,
            algorithm=algorithm,
            lidar_beams=max(0, observations.shape[1] - 3),
        )
    )
    metrics["notes"] = metrics["simulation_notes"]
    return metrics


def compact_hparams(config: dict[str, Any]) -> str:
    keep = [
        "batch_size",
        "lr",
        "actor_lr",
        "critic_lr",
        "vf_lr",
        "qf_lr",
        "awac_lambda",
        "iql_tau",
        "beta",
        "alpha",
        "discount",
        "normalize_obs",
        "normalize_actions",
        "seed",
        "max_steps",
    ]
    return json.dumps({key: config[key] for key in keep if key in config}, sort_keys=True)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, Any]], path: Path) -> None:
    sorted_rows = sorted(rows, key=lambda row: float(row["offline_action_mse"]))
    lines = [
        "# MediumExpert Offline RL Experiment Matrix Report",
        "",
        "This report covers the 12 reward/algorithm combinations trained on the `mediumexpert` split from `6_Split_Data_Better_Steering`.",
        "",
        "Environment evaluation return, normalized return, and lap-completion rate are blank because this repository does not currently provide a real or simulated evaluator for these checkpoints. The comparison below uses offline policy-vs-dataset action metrics plus dataset reward/terminal statistics.",
        "",
        "Hyperparameters were seeded from the best folder-5 experiment summary unless `--quick-search` is used.",
        "",
        "## Ranked By Offline Action MSE",
        "",
        "| Rank | Dataset | Algorithm | Offline action MSE | Mean policy speed | Mean abs steering | Inference ms/sample | Checkpoint |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(sorted_rows, start=1):
        lines.append(
            "| {rank} | {dataset} | {algorithm} | {mse:.6f} | {speed:.4f} | {steer:.4f} | {lat:.6f} | `{ckpt}` |".format(
                rank=rank,
                dataset=row["dataset"],
                algorithm=row["algorithm"],
                mse=float(row["offline_action_mse"]),
                speed=float(row["mean_policy_speed"]),
                steer=float(row["mean_abs_policy_steering"]),
                lat=float(row["inference_ms_per_sample"]),
                ckpt=row["checkpoint_path"],
            )
        )
    lines.extend(
        [
            "",
            "## Best Algorithm Per Reward Dataset",
            "",
        ]
    )
    for dataset_name in sorted({row["dataset"] for row in rows}):
        subset = [row for row in sorted_rows if row["dataset"] == dataset_name]
        if subset:
            best = subset[0]
            lines.append(
                f"- `{dataset_name}`: `{best['algorithm']}` by offline action MSE `{float(best['offline_action_mse']):.6f}`."
            )
    lines.extend(
        [
            "",
            "## Best Reward Dataset Per Algorithm",
            "",
        ]
    )
    for algorithm in sorted({row["algorithm"] for row in rows}):
        subset = [row for row in sorted_rows if row["algorithm"] == algorithm]
        if subset:
            best = subset[0]
            lines.append(
                f"- `{algorithm}`: `{best['dataset']}` by offline action MSE `{float(best['offline_action_mse']):.6f}`."
            )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```powershell",
            r"& 'C:\Users\ppwhi\anaconda3\envs\f110_training\python.exe' 6_Split_Data_Better_Steering/run_mediumexpert_matrix.py --epochs 100 --output-root 6_Split_Data_Better_Steering/experiments_mediumexpert_100 --clean",
            "```",
            "",
            "All final runs use `normalize_obs=True` and `normalize_actions=False`; the rosbag datasets already store actions as normalized `[speed, steering]`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_args(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged


def select_best_candidate(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(candidate_rows, key=lambda row: float(row["offline_action_mse"]))


def run_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.output_root)
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for dataset_name in args.datasets:
        data_folder = DATASETS[dataset_name]
        num_transitions = transition_count(data_folder)
        for algorithm in args.algorithms:
            algo = ALGORITHMS[algorithm]
            final_args = dict(algo["base_args"])
            final_args.update(FOLDER5_BEST_HPARAMS.get((dataset_name, algorithm), {}))
            batch_size = int(final_args.pop("batch_size", algo["batch_size"]))

            if args.quick_search:
                candidate_rows: list[dict[str, Any]] = []
                for index, candidate in enumerate(algo["search"], start=1):
                    candidate_args = merge_args(final_args, candidate)
                    candidate_steps = steps_for_epochs(num_transitions, batch_size, args.quick_epochs)
                    candidate_dir = out_root / dataset_name / algorithm / "search" / f"candidate_{index:02d}"
                    spec = RunSpec(
                        dataset_name=dataset_name,
                        algorithm=algorithm,
                        data_folder=data_folder,
                        checkpoint_dir=candidate_dir,
                        max_steps=candidate_steps,
                        eval_freq=max(1, steps_for_epochs(num_transitions, batch_size, 1)),
                        batch_size=batch_size,
                        seed=args.seed,
                        args=candidate_args,
                    )
                    print(f"\n=== Quick search {dataset_name}/{algorithm} candidate {index} ===")
                    train_metadata = run_training(spec)
                    row = evaluate_checkpoint(
                        dataset_name,
                        algorithm,
                        data_folder,
                        candidate_dir,
                        algo["final"],
                        train_metadata,
                    )
                    row["candidate_args"] = json.dumps(candidate_args, sort_keys=True)
                    candidate_rows.append(row)
                best = select_best_candidate(candidate_rows)
                final_args = json.loads(best["candidate_args"])

            final_steps = steps_for_epochs(num_transitions, batch_size, args.epochs)
            final_dir = out_root / dataset_name / algorithm / "final"
            spec = RunSpec(
                dataset_name=dataset_name,
                algorithm=algorithm,
                data_folder=data_folder,
                checkpoint_dir=final_dir,
                max_steps=final_steps,
                eval_freq=max(1, steps_for_epochs(num_transitions, batch_size, 1)),
                batch_size=batch_size,
                seed=args.seed,
                args=final_args,
            )
            print(f"\n=== Final training {dataset_name}/{algorithm}: {final_steps} steps ===")
            train_metadata = run_training(spec)
            row = evaluate_checkpoint(dataset_name, algorithm, data_folder, final_dir, algo["final"], train_metadata)
            rows.append(row)
            write_csv(rows, out_root / "summary.csv")
            write_report(rows, out_root / "report.md")

    write_csv(rows, out_root / "summary.csv")
    write_report(rows, out_root / "report.md")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mediumexpert 3x4 offline RL experiment matrix.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=list(ALGORITHMS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--quick-search", action="store_true")
    parser.add_argument("--quick-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default=str(ROOT / "experiments_mediumexpert_100"))
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    results = run_matrix(cli_args)
    print(f"\nCompleted {len(results)} final runs.")
    print(f"Summary: {Path(cli_args.output_root) / 'summary.csv'}")
