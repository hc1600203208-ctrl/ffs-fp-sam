#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
if [[ -z "${TENSORRT_ROOT:-}" && -d /usr/src/tensorrt ]]; then
  TENSORRT_ROOT="/usr/src/tensorrt"
fi
TENSORRT_ROOT="${TENSORRT_ROOT:-}"
BUILD_DINO_ONNXRUNTIME="${BUILD_DINO_ONNXRUNTIME:-OFF}"

if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  set +u
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
  set -u
fi

if ! command -v colcon >/dev/null 2>&1; then
  echo "colcon not found. Source ROS 2 and install colcon before building." >&2
  exit 1
fi

colcon build \
  --base-paths "${ROOT}" \
  --build-base "${ROOT}/ros2_build" \
  --install-base "${ROOT}/ros2_install" \
  --merge-install \
  --packages-select grounded_sam_trt \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORRT_ROOT="${TENSORRT_ROOT}" \
    -DBUILD_DINO_ONNXRUNTIME="${BUILD_DINO_ONNXRUNTIME}" \
    -DBUILD_GROUNDED_SAM_DEMO=OFF

echo "Built ROS 2 package grounded_sam_trt."
echo "Source ${ROOT}/ros2_install/setup.bash before launching FoundationPose."
