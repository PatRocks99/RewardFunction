# %%
"""
Focused TD3+BC refinement.

The first broad matrix showed TD3+BC drifting away from behavior actions on
several reward/lidar combinations. This pass tunes more BC-heavy settings on
mediumexpert, then trains mediumexpert/medium/expert with the selected params.
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
)


REWARD_DATASETS = ["reward_v1", "reward_v2", "reward_v3"]
LIDAR_BEAMS = [108, 54, 27]
SPLITS = ["mediumexpert", "medium", "expert"]
ALGORITHM = "td3bc"

BASE_PARAMS = {
    "batch_size": 512,
    "actor_lr": 3e-4,
    "critic_lr": 1e-4,
    "discount": 0.99,
    "tau": 0.005,
    "policy_noise": 0.2,
    "noise_clip": 0.5,
    "policy_delay": 2,
    "alpha": 0.1,
    "normalize_obs": True,
    "normalize_actions": False,
    "hidden_dim": 256,
    "hidden_layers": 2,
}

SEARCH_SPACE = [
    {"alpha": 0.0, "policy_delay": 1},
    {"alpha": 0.01, "policy_delay": 1},
    {"alpha": 0.05, "policy_delay": 1},
    {"alpha": 0.1, "policy_delay": 1},
    {"alpha": 0.25, "policy_delay": 1},
    {"alpha": 0.0, "policy_delay": 1, "actor_lr": 1e-4},
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
        "final_path": result["final_path"],
        "params": json.dumps(result["params"], sort_keys=True),
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    trainer = import_trainer("td3bc_refine_train", ROOT / "Controllers" / "TD3+BC" / "train_td3_bc.py")
    output_root = Path(args.output_root)
    rows: list[dict[str, Any]] = []
    for reward_dataset in args.reward_datasets:
        for beams in args.lidar_beams:
            combo_root = output_root / reward_dataset / f"lidar_{beams}" / ALGORITHM
            best_params_path = combo_root / "best_params.json"
            if best_params_path.exists() and args.resume:
                best_params = read_json(best_params_path)
                print(f"Reusing refined TD3+BC params for {reward_dataset}/lidar_{beams}")
            else:
                best_params, _ = run_hyperparameter_search(
                    algorithm=ALGORITHM,
                    trainer=trainer,
                    data_folder=dataset_folder(reward_dataset, "mediumexpert", beams),
                    output_root=combo_root,
                    base_params=BASE_PARAMS,
                    search_space=SEARCH_SPACE,
                    quick_epochs=args.quick_epochs,
                    device=args.device,
                    seed=args.seed,
                )
                write_json(best_params_path, best_params)
            for split in args.splits:
                result_path = combo_root / split / "deploy_ready_result.json"
                if result_path.exists() and args.resume:
                    result = read_json(result_path)
                    print(f"Reusing refined TD3+BC result for {reward_dataset}/lidar_{beams}/{split}")
                else:
                    result = train_deploy_ready(
                        algorithm=ALGORITHM,
                        trainer=trainer,
                        data_folder=dataset_folder(reward_dataset, split, beams),
                        output_root=combo_root / split,
                        params=best_params,
                        final_epochs=args.final_epochs,
                        device=args.device,
                        seed=args.seed,
                    )
                rows.append(compact_row(result, reward_dataset, split, beams))
                write_csv(output_root / "summary.csv", rows)
                write_json(output_root / "summary.json", rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused TD3+BC refinement.")
    parser.add_argument("--reward-datasets", nargs="+", default=REWARD_DATASETS, choices=REWARD_DATASETS)
    parser.add_argument("--lidar-beams", nargs="+", type=int, default=LIDAR_BEAMS, choices=LIDAR_BEAMS)
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--quick-epochs", type=int, default=8)
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--output-root", default=str(ROOT / "deploy_training_td3bc_refine"))
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    print("Python:", sys.executable)
    print("Device:", cli_args.device)
    results = run(cli_args)
    print(f"Completed {len(results)} refined TD3+BC final runs.")
    print(f"Summary: {Path(cli_args.output_root) / 'summary.csv'}")
