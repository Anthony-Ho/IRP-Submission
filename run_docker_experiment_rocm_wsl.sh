#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./run_docker_experiment_rocm_wsl.sh [combination_file] [iterations]
#
# Notes:
# - Run this from Linux/WSL (not native PowerShell).
# - Requires ROCm-in-WSL setup with /dev/dxg available.

DEFAULT_COMBINATION_FILE="combinations.csv"
DEFAULT_ITERATIONS=1
DEFAULT_IMAGE_NAME="rl-portfolio-optimization-rocm:latest"
DEFAULT_DOCKERFILE="docker/Dockerfile.rocm-wsl"

COMBINATION_FILE="${1:-$DEFAULT_COMBINATION_FILE}"
ITERATIONS="${2:-$DEFAULT_ITERATIONS}"
IMAGE_NAME="${IMAGE_NAME:-$DEFAULT_IMAGE_NAME}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-$DEFAULT_DOCKERFILE}"
ROCM_DEVICE="${ROCM_DEVICE:-/dev/dxg}"

SRC_DIR="$PWD/src"
DATA_DIR="$PWD/data"
RESULTS_DIR="$PWD/results"
MODEL_DIR="$PWD/models"

if [ ! -e "$ROCM_DEVICE" ]; then
    echo "ROCm device not found at $ROCM_DEVICE"
    echo "Expected to run inside WSL with ROCm support."
    exit 1
fi

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Building image $IMAGE_NAME from $DOCKERFILE_PATH..."
    docker build -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" .
fi

EXTRA_ENV_ARGS=()
if [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
    EXTRA_ENV_ARGS+=(-e "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}")
fi
if [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
    EXTRA_ENV_ARGS+=(-e "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}")
fi

echo "Using combination file: $COMBINATION_FILE"
echo "Running $ITERATIONS iteration(s)"
echo "Image: $IMAGE_NAME"

for (( i=1; i<=ITERATIONS; i++ ))
do
    echo "Starting iteration $i..."

    docker run --rm -it \
        --device="$ROCM_DEVICE" \
        --group-add video \
        --ipc=host \
        "${EXTRA_ENV_ARGS[@]}" \
        -v "$SRC_DIR:/workspace/IRP/src" \
        -v "$DATA_DIR:/workspace/IRP/data" \
        -v "$MODEL_DIR:/workspace/IRP/models" \
        -v "$RESULTS_DIR:/workspace/IRP/results" \
        "$IMAGE_NAME" --combination_file "$COMBINATION_FILE"

    echo "Completed iteration $i."
done

echo "Combining results using combine_results.sh..."
bash "$SRC_DIR/combine_results.sh"

echo "All iterations completed."
