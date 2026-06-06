# Reward Training Template

This folder mirrors the reusable core infrastructure from `6_Split_Data_Better_Steering`.

Included:
- `rosbag_to_reward_datasets.py` and `process_newcar_lidar_sweeps.py` for data processing.
- `Controllers/` with BC, AWAC, IQL, TD3+BC, reward search models, and shared offline utilities.
- `training_notebooks/` with standalone percent-style training scripts and matrix runners.
- Conda environment files for data processing and training.
- `simulation_eval_placeholder.py`, which keeps simulator rollout metrics as explicit placeholders until the simulator evaluator is added.

Not included:
- Training logs, generated matrix outputs, deploy checkpoints, best-model folders, and generated manifests.
- The full `deploy_training_matrix/` and `deploy_training_td3bc_refine/` checkpoint grids from `6_Split_Data_Better_Steering`.

Start points:
- Data generation: `rosbag_to_reward_datasets.py`
- Per-model training notebooks: `training_notebooks/train_*_percent.py`
- Full training matrix runner: `training_notebooks/run_deploy_training_matrix.py`
