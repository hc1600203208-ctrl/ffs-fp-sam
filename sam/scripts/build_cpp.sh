#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build"
TENSORRT_ROOT="${TENSORRT_ROOT:-}"
BUILD_DINO_ONNXRUNTIME="${BUILD_DINO_ONNXRUNTIME:-OFF}"

cmake -S "${ROOT}" -B "${BUILD_DIR}" \
  -DTENSORRT_ROOT="${TENSORRT_ROOT}" \
  -DBUILD_DINO_ONNXRUNTIME="${BUILD_DINO_ONNXRUNTIME}" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "Built ${BUILD_DIR}/grounded_sam_demo"
