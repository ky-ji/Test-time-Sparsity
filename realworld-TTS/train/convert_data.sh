#!/bin/bash
# TTS Data Conversion Script
# Convert Zarr format data to Trajectory format (for Episode-based training)

set -e

# -------------------------
# Configuration
# -------------------------
CHECKPOINT="${CHECKPOINT:-}"
ZARR_DATASET="${ZARR_DATASET:-}"  # Set via env: ZARR_DATASET=/path/to/dataset.zarr
TRAJECTORY_DIR="${TRAJECTORY_DIR:-}"  # Set via env: TRAJECTORY_DIR=/path/to/output_trajectory_dir
TRAIN_RATIO="${TRAIN_RATIO:-0.9}"
SEED="${SEED:-42}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REALWORLD_TTS_DIR="${SCRIPT_DIR}"

echo "=================================================="
echo "TTS Data Conversion - Zarr to Trajectory"
echo "=================================================="
echo "Checkpoint:       ${CHECKPOINT}"
echo "Zarr Dataset:     ${ZARR_DATASET}"
echo "Trajectory Dir:   ${TRAJECTORY_DIR}"
echo "Train Ratio:      ${TRAIN_RATIO}"
echo "Seed:             ${SEED}"
echo "=================================================="
echo ""

# Check if input files exist
if [ -z "${CHECKPOINT}" ]; then
    echo "ERROR: CHECKPOINT is empty."
    echo "Please set CHECKPOINT=/path/to/policy.ckpt (or set TTS_CHECKPOINT_PATH and pass it here)."
    exit 1
fi
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint file not found: ${CHECKPOINT}"
    exit 1
fi

if [ ! -d "${ZARR_DATASET}" ]; then
    echo "ERROR: Zarr dataset not found: ${ZARR_DATASET}"
    exit 1
fi

echo "=================================================="
echo "Converting Zarr to Trajectory Format"
echo "=================================================="
echo "This converts single-frame zarr data to episode-based"
echo "trajectory format required by TTSInfer's rollout cache training."
echo ""

python "${REALWORLD_TTS_DIR}/convert_zarr_to_trajectory.py" \
    --zarr_path "${ZARR_DATASET}" \
    --output_dir "${TRAJECTORY_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --train_ratio "${TRAIN_RATIO}" \
    --seed "${SEED}"

echo ""
echo "=================================================="
echo "Data Conversion Completed!"
echo "=================================================="
echo "Output directory: ${TRAJECTORY_DIR}"
echo "  Train data: ${TRAJECTORY_DIR}/train"
echo "  Val data:   ${TRAJECTORY_DIR}/val"
echo ""
echo "Next step: Run train_model.sh to train the pruner"
echo "=================================================="
