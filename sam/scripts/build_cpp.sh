#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build"
TENSORRT_ROOT="${TENSORRT_ROOT:-/usr/src/tensorrt}"

cmake -S "${ROOT}" -B "${BUILD_DIR}" -DTENSORRT_ROOT="${TENSORRT_ROOT}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "Built ${BUILD_DIR}/grounded_sam_demo"

