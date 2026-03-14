#!/bin/bash
# TTS Pruner Training Script
# Episode-based training with converted Trajectory data

set -e

# -------------------------
# Configuration
# -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/configs/train_pruner_config_assembly_bun.yaml}"
CHECKPOINT="${CHECKPOINT:-}"  # Set via env: CHECKPOINT=/path/to/dp_policy.ckpt
TRAJECTORY_DIR="${TRAJECTORY_DIR:-}"  # Set via env: TRAJECTORY_DIR=/path/to/trajectory_data
OUTPUT_DIR="${OUTPUT_DIR:-./output/pruner_tts}"
DEVICE="${DEVICE:-cuda:0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-31500}"

REALWORLD_TTS_DIR="${SCRIPT_DIR}"

echo "=================================================="
echo "TTS Pruner Training - Episode-based"
echo "=================================================="
echo "Config:           ${CONFIG}"
echo "Checkpoint:       ${CHECKPOINT}"
echo "Trajectory Dir:   ${TRAJECTORY_DIR}"
echo "Output Dir:       ${OUTPUT_DIR}"
echo "Device:           ${DEVICE}"
echo "Master Port:      ${MASTER_PORT}"
echo "=================================================="
echo ""

# Check if trajectory data exists
if [ ! -d "${TRAJECTORY_DIR}/train" ] || [ ! -d "${TRAJECTORY_DIR}/val" ]; then
    echo "ERROR: Trajectory data not found!"
    echo "Expected directories:"
    echo "  ${TRAJECTORY_DIR}/train"
    echo "  ${TRAJECTORY_DIR}/val"
    echo ""
    echo "Please run convert_data.sh first to convert zarr data to trajectory format."
    exit 1
fi

# Check if config file exists
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    exit 1
fi

# Check if checkpoint exists
if [ -z "${CHECKPOINT}" ]; then
    echo "ERROR: CHECKPOINT is empty."
    echo "Please set CHECKPOINT=/path/to/policy.ckpt"
    exit 1
fi
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint file not found: ${CHECKPOINT}"
    exit 1
fi

echo "=================================================="
echo "Training TTS Pruner (Episode-based)"
echo "=================================================="
echo "Training with:"
echo "  - Real pruner object (not None)"
echo "  - Step-by-step pruner calls (TTSInfer style)"
echo "  - Episode-based training with rollout cache"
echo "  - 4-dimensional gates (compute, 3cache, 24cache, rollout_cache)"
echo ""

if [ "${NUM_GPUS}" -gt 1 ]; then
    torchrun --nproc_per_node "${NUM_GPUS}" \
        --master_port "${MASTER_PORT}" \
        "${REALWORLD_TTS_DIR}/train_pruner_tts_trajectory.py" \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --trajectory_dir "${TRAJECTORY_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --device "${DEVICE}"
else
    python "${REALWORLD_TTS_DIR}/train_pruner_tts_trajectory.py" \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --trajectory_dir "${TRAJECTORY_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --device "${DEVICE}"
fi

echo ""
echo "=================================================="
echo "Training Completed!"
echo "=================================================="
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Summary:"
echo "  ✓ Trained with episode-based approach (TTSInfer style)"
echo "  ✓ Rollout cache enabled for cross-frame reuse"
echo "  ✓ 4-strategy gates (compute, 3cache, 24cache, rollout_cache)"
echo "  ✓ Step-by-step pruner computation (performance optimized)"
echo ""
echo "This training pipeline fully replicates TTSInfer on real robot data!"
echo "=================================================="
