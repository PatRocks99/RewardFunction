# Conda Environments

This folder uses two separate environments:

- `f110_data_processing`: ROS bag decoding and reward dataset generation.
- `f110_training`: offline controller training with PyTorch/CUDA.

Create or update them from the repository root on Ubuntu/WSL:

```bash
conda env create -f 00_Ubuntu_Template/environment_data_processing.yml
conda env create -f 00_Ubuntu_Template/environment_training.yml
```

If the environments already exist:

```bash
conda env update -n f110_data_processing -f 00_Ubuntu_Template/environment_data_processing.yml --prune
conda env update -n f110_training -f 00_Ubuntu_Template/environment_training.yml --prune
```

Run data processing:

```bash
conda activate f110_data_processing
export F110_DATA_ROOT=/mnt/p/Car/NewCar
python 00_Ubuntu_Template/process_newcar_lidar_sweeps.py
```

Run training:

```bash
conda activate f110_training
export F110_DATA_ROOT=/mnt/p/Car/NewCar
python 00_Ubuntu_Template/training_notebooks/run_deploy_training_matrix.py --device cuda
```

For machines without an NVIDIA GPU, remove `pytorch-cuda=12.1` from
`environment_training.yml` before creating the training environment.
