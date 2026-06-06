# %%
"""
Run deploy-ready offline training across algorithms, lidar lengths, and quality splits.

Workflow:
- Tune hyperparameters on mediumexpert for each algorithm/lidar length.
- Train final deploy-ready checkpoints on mediumexpert, medium, and expert using
  the selected mediumexpert hyperparameters for that algorithm/lidar length.
- Write incremental JSON/CSV summaries so interrupted runs can be resumed.

Run with the f110 training environment:

    & 'C:\\Users\\ppwhi\\anaconda3\\envs\\f110_training\\python.exe' `
      6_Split_Data_Better_Steering\\training_notebooks\\run_deploy_training_matrix.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from training_notebook_common import (
    ROOT,
    dataset_folder,
    default_device,
    import_trainer,
    run_hyperparameter_search,
    train_deploy_ready,
    transition_count,
)


# %%
ALGORITHMS = {
    "bc": {
        "trainer": ROOT / "Controllers" / "BC" / "train_bc.py",
        "module": "matrix_train_bc",
        "base": {
            "batch_size": 1024,
            "lr": 3e-4,
            "normalize_obs": True,
            "normalize_actions": False,
            "hidden_dim": 256,
            "hidden_layers": 2,
        },
        "search": [
            {"lr": 1e-4},
            {"lr": 3e-4},
            {"lr": 1e-3},
            {"lr": 3e-4, "hidden_dim": 512},
        ],
    },
    "awac": {
        "trainer": ROOT / "Controllers" / "AWAC" / "train_awac.py",
        "module": "matrix_train_awac",
        "base": {
            "batch_size": 256,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "discount": 0.99,
            "tau": 0.005,
            "awac_lambda": 1.0,
            "max_weight": 20.0,
            "normalize_obs": True,
            "normalize_actions": False,
            "hidden_dim": 256,
            "hidden_layers": 2,
        },
        "search": [
            {"awac_lambda": 0.3},
            {"awac_lambda": 1.0},
            {"awac_lambda": 3.0},
            {"awac_lambda": 3.0, "max_weight": 50.0},
        ],
    },
    "iql": {
        "trainer": ROOT / "Controllers" / "IQL" / "train_iql.py",
        "module": "matrix_train_iql",
        "base": {
            "batch_size": 256,
            "actor_lr": 3e-4,
            "qf_lr": 3e-4,
            "vf_lr": 3e-4,
            "discount": 0.99,
            "tau": 0.005,
            "beta": 3.0,
            "iql_tau": 0.7,
            "deterministic_actor": False,
            "normalize_obs": True,
            "normalize_actions": False,
            "hidden_dim": 256,
            "hidden_layers": 2,
        },
        "search": [
            {"iql_tau": 0.7, "beta": 1.0},
            {"iql_tau": 0.7, "beta": 3.0},
            {"iql_tau": 0.8, "beta": 3.0},
            {"iql_tau": 0.9, "beta": 3.0},
        ],
    },
    "td3bc": {
        "trainer": ROOT / "Controllers" / "TD3+BC" / "train_td3_bc.py",
        "module": "matrix_train_td3bc",
        "base": {
            "batch_size": 512,
            "actor_lr": 3e-4,
            "critic_lr": 1e-4,
            "discount": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_delay": 2,
            "alpha": 2.5,
            "normalize_obs": True,
            "normalize_actions": False,
            "hidden_dim": 256,
            "hidden_layers": 2,
        },
        "search": [
            {"alpha": 1.0},
            {"alpha": 2.5},
            {"alpha": 5.0},
            {"alpha": 2.5, "critic_lr": 3e-4},
        ],
    },
}

REWARD_DATASETS = ["reward_v1", "reward_v2", "reward_v3"]
LIDAR_BEAMS = [108, 54, 27]
SPLITS = ["mediumexpert", "medium", "expert"]
TUNE_SPLIT = "mediumexpert"


# %%
def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_row(result: dict[str, Any], reward_dataset: str, split: str, lidar_beams: int) -> dict[str, Any]:
    return {
        "reward_dataset": reward_dataset,
        "split": split,
        "lidar_beams": lidar_beams,
        "algorithm": result["algorithm"],
        "offline_action_mse": result["offline_action_mse"],
        "offline_action_mae": result["offline_action_mae"],
        "mean_policy_speed": result["mean_policy_speed"],
        "mean_abs_policy_steering": result["mean_abs_policy_steering"],
        "dataset_reward_mean": result["dataset_reward_mean"],
        "terminal_rate": result["terminal_rate"],
        "inference_ms_per_sample": result["inference_ms_per_sample"],
        "max_steps": result["max_steps"],
        "epochs": result["epochs"],
        "final_path": result["final_path"],
        "params": json.dumps(result["params"], sort_keys=True),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_trainers(algorithms: list[str]) -> dict[str, Any]:
    return {
        algorithm: import_trainer(ALGORITHMS[algorithm]["module"], ALGORITHMS[algorithm]["trainer"])
        for algorithm in algorithms
    }


def count_manifest(reward_datasets: list[str], splits: list[str], lidar_beams: list[int]) -> list[dict[str, Any]]:
    rows = []
    for reward_dataset in reward_datasets:
        for split in splits:
            for beams in lidar_beams:
                folder = dataset_folder(reward_dataset, split, beams)
                rows.append(
                    {
                        "reward_dataset": reward_dataset,
                        "split": split,
                        "lidar_beams": beams,
                        "data_folder": str(folder),
                        "transitions": transition_count(folder),
                    }
                )
    return rows


def run_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    reward_datasets = args.reward_datasets
    algorithms = args.algorithms
    lidar_beams = args.lidar_beams
    splits = args.splits

    trainers = load_trainers(algorithms)
    counts = count_manifest(reward_datasets, splits, lidar_beams)
    write_json(output_root / "dataset_counts.json", counts)

    rows: list[dict[str, Any]] = []
    summary_path = output_root / "summary.csv"

    for reward_dataset in reward_datasets:
        for beams in lidar_beams:
            for algorithm in algorithms:
                algo_spec = ALGORITHMS[algorithm]
                tune_folder = dataset_folder(reward_dataset, TUNE_SPLIT, beams)
                combo_root = output_root / reward_dataset / f"lidar_{beams}" / algorithm
                best_params_path = combo_root / "best_params.json"

                if best_params_path.exists() and args.resume:
                    best_params = read_json(best_params_path)
                    print(f"Reusing selected params for {reward_dataset}/lidar_{beams}/{algorithm}")
                else:
                    best_params, search_results = run_hyperparameter_search(
                        algorithm=algorithm,
                        trainer=trainers[algorithm],
                        data_folder=tune_folder,
                        output_root=combo_root,
                        base_params=algo_spec["base"],
                        search_space=algo_spec["search"],
                        quick_epochs=args.quick_epochs,
                        device=args.device,
                        seed=args.seed,
                    )
                    write_json(best_params_path, best_params)

                for split in splits:
                    final_dir = combo_root / split / "deploy_ready"
                    final_result_path = combo_root / split / "deploy_ready_result.json"
                    if final_result_path.exists() and args.resume:
                        result = read_json(final_result_path)
                        print(f"Reusing final result for {reward_dataset}/lidar_{beams}/{algorithm}/{split}")
                    else:
                        result = train_deploy_ready(
                            algorithm=algorithm,
                            trainer=trainers[algorithm],
                            data_folder=dataset_folder(reward_dataset, split, beams),
                            output_root=combo_root / split,
                            params=best_params,
                            final_epochs=args.final_epochs,
                            device=args.device,
                            seed=args.seed,
                        )
                    rows.append(compact_row(result, reward_dataset, split, beams))
                    write_csv(summary_path, rows)
                    write_json(output_root / "summary.json", rows)
    return rows


# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deploy training matrix.")
    parser.add_argument("--reward-datasets", nargs="+", default=REWARD_DATASETS, choices=REWARD_DATASETS)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=list(ALGORITHMS))
    parser.add_argument("--lidar-beams", nargs="+", type=int, default=LIDAR_BEAMS, choices=LIDAR_BEAMS)
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--quick-epochs", type=int, default=5)
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--output-root", default=str(ROOT / "deploy_training_matrix"))
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


# %%
if __name__ == "__main__":
    cli_args = parse_args()
    print("Python:", sys.executable)
    print("Device:", cli_args.device)
    results = run_matrix(cli_args)
    print(f"Completed {len(results)} final training runs.")
    print(f"Summary: {Path(cli_args.output_root) / 'summary.csv'}")
