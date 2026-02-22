#!/usr/bin/env bash
set -euo pipefail

# Build script for AMD AI-MAX 395 on Ubuntu 24.04 (ROCm path).
#
# Usage:
#   ./build_docker_ai_max395.sh
#
# Optional env vars:
#   IMAGE_NAME=rl-portfolio-optimization-ai-max395:latest
#   DOCKERFILE_PATH=docker/Dockerfile.rocm-ubuntu24
#   BASE_IMAGE=rocm/pytorch:latest
#   NO_CACHE=0|1
#   BUILD_CONTEXT=.

IMAGE_NAME="${IMAGE_NAME:-rl-portfolio-optimization-ai-max395:latest}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-docker/Dockerfile.rocm-ubuntu24}"
BASE_IMAGE="${BASE_IMAGE:-rocm/pytorch:latest}"
NO_CACHE="${NO_CACHE:-0}"
BUILD_CONTEXT="${BUILD_CONTEXT:-.}"

if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
    echo "Dockerfile not found: ${DOCKERFILE_PATH}"
    exit 1
fi

BUILD_ARGS=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
if [[ "${NO_CACHE}" == "1" ]]; then
    BUILD_ARGS+=(--no-cache)
fi

echo "Building image: ${IMAGE_NAME}"
echo "Dockerfile: ${DOCKERFILE_PATH}"
echo "Base image: ${BASE_IMAGE}"

docker build \
    "${BUILD_ARGS[@]}" \
    -f "${DOCKERFILE_PATH}" \
    -t "${IMAGE_NAME}" \
    "${BUILD_CONTEXT}"

echo "Build completed: ${IMAGE_NAME}"
echo "Next step: run with ./run_docker_experiment_rocm_linux.sh <combination_file> <iterations>"
