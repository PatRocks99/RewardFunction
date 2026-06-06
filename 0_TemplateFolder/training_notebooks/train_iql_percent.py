# %%
"""
IQL percent-notebook.

Run cells top-to-bottom in VS Code, or run this file directly:

    python 6_Split_Data_Better_Steering/training_notebooks/train_iql_percent.py
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
ALGORITHM = "iql"
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
}

SEARCH_SPACE = [
    {"iql_tau": 0.7, "beta": 1.0},
    {"iql_tau": 0.7, "beta": 3.0},
    {"iql_tau": 0.8, "beta": 3.0},
    {"iql_tau": 0.9, "beta": 3.0},
]


# %%
# Trainer import.
trainer = import_trainer(
    "percent_notebook_train_iql",
    ROOT / "Controllers" / "IQL" / "train_iql.py",
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
