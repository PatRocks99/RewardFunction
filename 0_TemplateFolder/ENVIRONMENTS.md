# Conda Environments

This folder uses two separate environments:

- `f110_data_processing`: ROS bag decoding and reward dataset generation.
- `f110_training`: offline controller training with PyTorch/CUDA.

Create or update them from the repository root:

```powershell
conda env create -f 6_Split_Data_Better_Steering\environment_data_processing.yml
conda env create -f 6_Split_Data_Better_Steering\environment_training.yml
```

If the environments already exist:

```powershell
conda env update -n f110_data_processing -f 6_Split_Data_Better_Steering\environment_data_processing.yml --prune
conda env update -n f110_training -f 6_Split_Data_Better_Steering\environment_training.yml --prune
```

Run data processing:

```powershell
conda activate f110_data_processing
python 6_Split_Data_Better_Steering\rosbag_to_reward_datasets.py
```

Run training:

```powershell
conda activate f110_training
python 6_Split_Data_Better_Steering\training_notebooks\run_deploy_training_matrix.py
```

For machines without an NVIDIA GPU, remove `pytorch-cuda=12.1` from
`environment_training.yml` before creating the training environment.
