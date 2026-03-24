#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="${IMAGE_NAME:-docker.io/kyuz0/amd-strix-halo-toolboxes}"
IMAGE_TAG="${IMAGE_TAG:-vulkan-moe-flash}"
LLAMA_CPP_TAG="${LLAMA_CPP_TAG:-b8298}"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  llama.cpp tag: ${LLAMA_CPP_TAG}"
echo "  patch: patches/0001-moe-flash-complete.patch"

docker build \
    -f "${SCRIPT_DIR}/Dockerfile.vulkan-moe-flash" \
    --build-arg LLAMA_CPP_TAG="${LLAMA_CPP_TAG}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "${PROJECT_DIR}"

echo ""
echo "Built: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Push with: docker push ${IMAGE_NAME}:${IMAGE_TAG}"
