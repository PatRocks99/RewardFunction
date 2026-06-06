# %%
"""
Behavior Cloning percent-notebook.

Run cells top-to-bottom in VS Code, or run this file directly:

    python 6_Split_Data_Better_Steering/training_notebooks/train_bc_percent.py
"""

from __future__ import annotations

from pathlib import Path

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
ALGORITHM = "bc"
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
    "batch_size": 1024,
    "lr": 3e-4,
    "normalize_obs": True,
    "normalize_actions": False,
    "hidden_dim": 256,
    "hidden_layers": 2,
}

SEARCH_SPACE = [
    {"lr": 1e-4},
    {"lr": 3e-4},
    {"lr": 1e-3},
    {"lr": 3e-4, "hidden_dim": 512},
]


# %%
# Trainer import.
trainer = import_trainer(
    "percent_notebook_train_bc",
    ROOT / "Controllers" / "BC" / "train_bc.py",
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
