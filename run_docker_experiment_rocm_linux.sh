#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_docker_experiment_rocm_linux.sh [combination_file] [iterations]
#
# Optional env vars:
#   IMAGE_NAME=rl-portfolio-optimization-ai-max395:latest
#   DOCKERFILE_PATH=docker/Dockerfile.rocm-ubuntu24
#   BASE_IMAGE=rocm/pytorch:latest
#   GPU_INDEX=0
#   HIP_VISIBLE_DEVICES=0
#   ROCR_VISIBLE_DEVICES=0
#   HSA_OVERRIDE_GFX_VERSION=11.0.0

DEFAULT_COMBINATION_FILE="combinations.csv"
DEFAULT_ITERATIONS=1
DEFAULT_IMAGE_NAME="rl-portfolio-optimization-ai-max395:latest"
DEFAULT_DOCKERFILE="docker/Dockerfile.rocm-ubuntu24"

COMBINATION_FILE="${1:-$DEFAULT_COMBINATION_FILE}"
ITERATIONS="${2:-$DEFAULT_ITERATIONS}"
IMAGE_NAME="${IMAGE_NAME:-$DEFAULT_IMAGE_NAME}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-$DEFAULT_DOCKERFILE}"
BASE_IMAGE="${BASE_IMAGE:-rocm/pytorch:latest}"
GPU_INDEX="${GPU_INDEX:-0}"

SRC_DIR="$PWD/src"
DATA_DIR="$PWD/data"
RESULTS_DIR="$PWD/results"
MODEL_DIR="$PWD/models"

if [[ ! -e /dev/kfd ]]; then
    echo "ROCm device not found: /dev/kfd"
    echo "Install ROCm on host and run on native Linux."
    exit 1
fi

if [[ ! -e /dev/dri ]]; then
    echo "DRI device path not found: /dev/dri"
    exit 1
fi

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Image not found. Building ${IMAGE_NAME}..."
    IMAGE_NAME="$IMAGE_NAME" DOCKERFILE_PATH="$DOCKERFILE_PATH" BASE_IMAGE="$BASE_IMAGE" \
        ./build_docker_ai_max395.sh
fi

EXTRA_ENV_ARGS=()
if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
    EXTRA_ENV_ARGS+=(-e "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}")
else
    EXTRA_ENV_ARGS+=(-e "HIP_VISIBLE_DEVICES=${GPU_INDEX}")
fi
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    EXTRA_ENV_ARGS+=(-e "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES}")
else
    EXTRA_ENV_ARGS+=(-e "ROCR_VISIBLE_DEVICES=${GPU_INDEX}")
fi
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
    EXTRA_ENV_ARGS+=(-e "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}")
fi

echo "Using combination file: $COMBINATION_FILE"
echo "Running $ITERATIONS iteration(s)"
echo "Image: $IMAGE_NAME"
echo "GPU_INDEX: $GPU_INDEX (iGPU target by default)"

for (( i=1; i<=ITERATIONS; i++ ))
do
    echo "Starting iteration $i..."

    docker run --rm -it \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --group-add render \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
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
