# Ubuntu/WSL Reward Training Template

This folder is a clean Ubuntu-oriented copy of `6_Split_Data_Better_Steering`
without generated training results, checkpoints, logs, or caches.

Use this folder for:
- ROS bag to reward dataset processing on Ubuntu/WSL.
- Offline model training with PyTorch/CUDA.
- Future simulator evaluation integration.

It intentionally excludes:
- `deploy_training_matrix/`
- `final_best_models/`
- training logs
- generated checkpoints
- generated processed datasets

Default paths assume WSL can see the Windows data drive at:

```bash
/mnt/p/Car/NewCar
```

Override paths with environment variables:

```bash
export F110_DATA_ROOT=/mnt/p/Car/NewCar
export F110_OUTPUT_ROOT=/mnt/p/Car/NewCar
export F110_NEW_TRACK_ROOT=/mnt/p/Car/NewCar/NewTrackData
export F110_SECOND_TRACK_ROOT=/mnt/p/Car/NewCar/secondTrack
```

For native Ubuntu, a common layout would be:

```bash
export F110_DATA_ROOT=/data/Car/NewCar
```

Create environments:

```bash
conda env create -f environment_data_processing.yml
conda env create -f environment_training.yml
```

Run data processing:

```bash
conda activate f110_data_processing
python process_newcar_lidar_sweeps.py
```

Run training:

```bash
conda activate f110_training
python training_notebooks/run_deploy_training_matrix.py --device cuda
python training_notebooks/collect_best_models.py
```

Simulator evaluation is still represented by `simulation_eval_placeholder.py`.
The intended next step is to replace that placeholder with an Ubuntu/ROS 2
evaluation node for simulator rollouts.
