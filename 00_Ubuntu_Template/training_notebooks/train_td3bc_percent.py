# %%
"""
TD3+BC percent-notebook.

Run cells top-to-bottom in VS Code, or run this file directly:

    python 00_Ubuntu_Template/training_notebooks/train_td3bc_percent.py
"""

from __future__ import annotations

from training_notebook_common import (
    ROOT,
    dataset_folder,
    default_device,
    import_trainer,
    run_hyperparameter_search,
    train_deploy_ready,
)


# %%
# Adjustable experiment parameters.
ALGORITHM = "td3bc"
REWARD_DATASET = "reward_v3"
SPLIT = "mediumexpert"
LIDAR_BEAMS = 108
SEED = 0
DEVICE = default_device()

QUICK_SEARCH_EPOCHS = 5
FINAL_TRAIN_EPOCHS = 100

DATA_FOLDER = dataset_folder(REWARD_DATASET, SPLIT, LIDAR_BEAMS)
OUTPUT_ROOT = ROOT / "deploy_training" / REWARD_DATASET / ALGORITHM

BASE_PARAMS = {
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
}

SEARCH_SPACE = [
    {"alpha": 1.0},
    {"alpha": 2.5},
    {"alpha": 5.0},
    {"alpha": 2.5, "critic_lr": 3e-4},
]


# %%
# Trainer import.
trainer = import_trainer(
    "percent_notebook_train_td3bc",
    ROOT / "Controllers" / "TD3+BC" / "train_td3_bc.py",
)


# %%
def run_search():
    return run_hyperparameter_search(
        algorithm=ALGORITHM,
        trainer=trainer,
        data_folder=DATA_FOLDER,
        output_root=OUTPUT_ROOT,
        base_params=BASE_PARAMS,
        search_space=SEARCH_SPACE,
        quick_epochs=QUICK_SEARCH_EPOCHS,
        device=DEVICE,
        seed=SEED,
    )


# %%
def run_final(best_params):
    return train_deploy_ready(
        algorithm=ALGORITHM,
        trainer=trainer,
        data_folder=DATA_FOLDER,
        output_root=OUTPUT_ROOT,
        params=best_params,
        final_epochs=FINAL_TRAIN_EPOCHS,
        device=DEVICE,
        seed=SEED,
    )


# %%
if __name__ == "__main__":
    best_params, search_results = run_search()
    final_result = run_final(best_params)
    print("Best params:", best_params)
    print("Deploy-ready model:", final_result["final_path"])
