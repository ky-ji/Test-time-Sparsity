# Test-time Sparsity for Extreme Fast Action Diffusion

<div align="center">

### ⚡ Accelerate Action Diffusion by 5× via Dynamic Test-time Pruning and Omnidirectional Feature Reusing

[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg)]()
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)]()
[![GitHub](https://img.shields.io/badge/GitHub-Code-black.svg)](https://github.com/ky-ji/Test-time-Sparsity)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-purple.svg)](https://cvpr.thecvf.com/)

**CVPR 2026**

[Paper]() | [Code](https://github.com/ky-ji/Test-time-Sparsity)

</div>

---

## Overview

**Test-time Sparsity (TTS)** accelerates action diffusion by dynamically predicting prunable residual computations for each model forward at test time. Our method reduces FLOPs by **92%** and achieves **5× wall-clock speedup**, reaching an inference frequency of **47.5 Hz** on an NVIDIA 4090 GPU without performance degradation.

### Key Features

- **Dynamic Test-time Pruning**: A lightweight pruner that shares the encoder with the diffusion transformer, dynamically predicting skippable residual computations before each forward pass
- **Omnidirectional Feature Reusing**: Achieves **95% sparsity** by selectively reusing features cached from the current forward, previous denoising timesteps, and earlier rollout iterations
- **Highly Parallelized Pipeline**: Decouples encoding and pruning from the autoregressive denoising loop, reducing non-decoder delay to milliseconds via parallel processing and asynchronous execution


---

## Installation

### Prerequisites

- Python 3.9
- CUDA 11.6 or higher
- Conda package manager

### Setup Environment

```bash
git clone https://github.com/ky-ji/Test-time-Sparsity.git
cd Test-time-Sparsity

git submodule update --init --recursive

conda env create -f conda_environment.yaml
conda activate robodiff

pip install -e .
```

---

## Data Preparation

TTS builds on top of [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/). You need to first obtain baseline policy checkpoints before training the pruner.

### Step 1: Download Diffusion Policy datasets

Follow the [Diffusion Policy data download instructions](https://diffusion-policy.cs.columbia.edu/) to download the robosuite / FurnitureBench datasets. By default the code expects data under a `data/` directory (excluded from git).

Supported simulation tasks: `can_ph`, `can_mh`, `lift_ph`, `lift_mh`, `square_ph`, `square_mh`, `transport`, `tool_hang`, `kitchen`.

### Step 2: Train or download the baseline Diffusion Policy checkpoint

Train a Diffusion Transformer policy using the upstream `diffusion_policy` module:

```bash
cd diffusion_policy
python train.py --config-name=train_diffusion_transformer_hybrid_workspace \
  task=<task_name>
```

Or download pre-trained checkpoints from the [Diffusion Policy project page](https://diffusion-policy.cs.columbia.edu/).

### Step 3: Prepare trajectory data

The public simulation training pipeline reads rollout-level trajectory data from `pruner_tra_data_<datatype>/trajectories/<task_name>`. The current training script does **not** generate this data automatically when `--pruner_epoch` is specified; it expects the trajectory directory to already exist.

---

## Reproduce Simulation Results

This section reproduces the main results from Tables 1–3 of the paper using the `TTSInfer` module.

### Directory structure before training

```
Test-time-Sparsity/
├── TTSInfer/
│   └── pruner_config/          # training configs (4 main configs)
├── diffusion_policy/           # submodule: base policy + env runner
└── <your_checkpoint.ckpt>      # baseline DP checkpoint (user-provided)
```

### Stage 1: Train the Pruner

Train the lightweight pruner to predict per-forward pruning decisions:

```bash
python -m TTSInfer.scripts.train_eval.train_pruner_2stage \
  --config_id 2stage2 \
  --task_name can_ph \
  --device cuda:0 \
  --output_dir exp_output \
  --train_version 0
```

**Input**: baseline `.ckpt` checkpoint (loaded automatically based on `task_name`)  
**Output**: `exp_output/stage1ckpt/<timestamp>/<train_id>/<task_name>/pruner_model_<epoch>_<loss>.pt`

Available configs in `TTSInfer/pruner_config/`:

| Config | Description |
|--------|-------------|
| `training_config_2stage2.yaml` | Default main-result config (recommended starting point) |
| `training_config_2stage96.yaml` | Larger Stage 2 trajectory batches |
| `training_config_40ddim96.yaml` | DDIM scheduler variant (50-step, Table 4) |
| `training_config_95pr96.yaml` | 95% sparsity variant |

Additional supplementary configs (for ablation/sensitivity analysis) are in `TTSInfer/pruner_config/supplementary/`.
Main-table config guesses and example commands are summarized in `MAIN_TABLES.md`.

### Trajectory-Based Training

The released simulation code trains a rollout-cache pruner directly from trajectory data. This is the training path used by the main public configs.

```bash
python -m TTSInfer.scripts.train_eval.train_pruner_2stage \
  --config_id 2stage2 \
  --task_name can_ph \
  --device cuda:0 \
  --output_dir exp_stage2 \
  --pruner_epoch 2 \
  --datatype max \
  --train_version 0
```

**Input**: trajectory data under `pruner_tra_data_max/trajectories/<task_name>`  
**Output**: `exp_stage2/stage2ckpt/end2end/<timestamp>/<train_id>/<task_name>/pruner_model_<epoch>_<loss>.pt`

Notes:

- `--pruner_epoch` is kept for compatibility with historical experiment naming and output folder layout.
- The current public configs keep only parameters that are actually consumed by the released simulation training code.

### Evaluate

Evaluate the pruner-accelerated policy on simulation benchmarks:

```bash
python -m TTSInfer.scripts.train_eval.eval_pruner_dp \
  --output_dir exp_stage2 \
  --timestamp <train_timestamp> \
  --task_name can_ph \
  --train_id 0 \
  --epoch <best_epoch> \
  --stage1_epoch <stage1_epoch> \
  --device cuda:0 \
  --rollout_cache True \
  --stage2_mode end2end
```

**Input**: Stage 2 pruner checkpoint  
**Output**: success rate, sparsity ratio, and action error metrics printed to stdout

### Speed Benchmarking

Measure wall-clock inference latency to reproduce the speedup numbers:

```bash
python -m TTSInfer.scripts.exp.eval_speed_only \
  -t can_ph \
  -e <epoch> \
  --train_root <path_to_exp_stage2> \
  --device cuda:0
```

### Supplementary Analysis Scripts

The following scripts in `TTSInfer/scripts/exp/` reproduce supplementary experiments:

| Script | Purpose |
|--------|---------|
| `eval_pruner_ablation.py` | Ablation on omnidirectional cache (Tables 6–8) |
| `eval_ddim_steps.py` | DDIM step count sensitivity |
| `compute_module_flops.py` | FLOPs breakdown (Figure 4) |
| `train_pruner_ablation_single_cache.py` | Single-cache ablation training |

---

## Real-World Robot Pipeline

The `realworld-TTS/` module provides a complete pipeline for deploying TTS on real robots with your own collected data.

### Step 1: Convert collected data to trajectory format

```bash
cd realworld-TTS

ZARR_DATASET=/path/to/your_dataset.zarr \
TRAJECTORY_DIR=/path/to/output_trajectory \
CHECKPOINT=/path/to/dp_policy.ckpt \
bash train/convert_data.sh
```

Training configs for the conversion and pruner training are in `realworld-TTS/train/configs/`. Copy and adapt one:

```bash
cp train/configs/train_pruner_config_pick.yaml train/configs/my_task.yaml
# Edit my_task.yaml: set task_name, hidden_dim, pruning ratio, etc.
```

### Step 2: Train the pruner on real robot data

```bash
cd realworld-TTS

CONFIG=train/configs/my_task.yaml \
CHECKPOINT=/path/to/dp_policy.ckpt \
TRAJECTORY_DIR=/path/to/output_trajectory \
OUTPUT_DIR=./output/pruner_tts \
bash train/train_pruner.sh
```

Multi-GPU training is supported automatically when `NUM_GPUS > 1`.

### Step 3: Run TTS-Accelerated Inference Server

```bash
cd realworld-TTS

# Option A: use a config file
python tts_accelerator/scripts/run_dp_with_tts.py \
  --config tts_accelerator/configs/assembly_bun.yaml

# Option B: specify paths directly
python tts_accelerator/scripts/run_dp_with_tts.py \
  --checkpoint /path/to/dp_policy.ckpt \
  --pruner /path/to/pruner.pt \
  --port 8007
```

### Server Configuration

Edit or copy `realworld-TTS/server/configs/server_config_local.py` to set:

```python
CHECKPOINT_PATH = "/path/to/policy.ckpt"
PRUNER_CHECKPOINT = "/path/to/pruner.pt"
DEVICE = "cuda:0"
SCHEDULER_TYPE = "DDPM"   # or "DDIM"
NUM_INFERENCE_STEPS = 100
TARGET_PRUNE_RATIO = 0.93
```


---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



</div>
