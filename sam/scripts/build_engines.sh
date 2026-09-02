#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRTEXEC="${TRTEXEC:-trtexec}"
TENSORRT_ROOT="${TENSORRT_ROOT:-}"
DINO_HEIGHT="${DINO_HEIGHT:-800}"
DINO_WIDTH="${DINO_WIDTH:-1066}"
BUILD_DINO_TRT_ENGINE="${BUILD_DINO_TRT_ENGINE:-1}"

if ! command -v "${TRTEXEC}" >/dev/null 2>&1; then
  for candidate in \
    "${TENSORRT_ROOT}/bin/trtexec" \
    "${TENSORRT_ROOT}/targets/aarch64-linux-gnu/bin/trtexec" \
    "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/bin/trtexec" \
    /usr/src/tensorrt/bin/trtexec \
    /usr/src/tensorrt/targets/aarch64-linux-gnu/bin/trtexec \
    /usr/src/tensorrt/targets/x86_64-linux-gnu/bin/trtexec \
    /usr/bin/trtexec; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      TRTEXEC="${candidate}"
      break
    fi
  done
fi

if ! command -v "${TRTEXEC}" >/dev/null 2>&1; then
  echo "trtexec not found. Set TRTEXEC=/path/to/trtexec or TENSORRT_ROOT=/path/to/tensorrt" >&2
  exit 1
fi

mkdir -p "${ROOT}/engines"

if [[ "${BUILD_DINO_TRT_ENGINE}" == "1" ]]; then
  "${TRTEXEC}" \
    --onnx="${ROOT}/models/grounding_dino_fixed_prompt.onnx" \
    --saveEngine="${ROOT}/engines/grounding_dino_fixed_prompt.engine" \
    --noTF32 \
    --stronglyTyped
else
  echo "Skipping DINO TensorRT engine build."
  echo "Set BUILD_DINO_TRT_ENGINE=1 to build the DINO engine for validation on another target."
fi

"${TRTEXEC}" \
  --onnx="${ROOT}/models/sam_image_encoder.onnx" \
  --saveEngine="${ROOT}/engines/sam_image_encoder.engine" \
  --minShapes=image:1x3x1024x1024 \
  --optShapes=image:1x3x1024x1024 \
  --maxShapes=image:1x3x1024x1024 \
  --fp16

"${TRTEXEC}" \
  --onnx="${ROOT}/models/sam_mask_decoder.onnx" \
  --saveEngine="${ROOT}/engines/sam_mask_decoder.engine" \
  --minShapes=image_embeddings:1x256x64x64,point_coords:1x2x2,point_labels:1x2,mask_input:1x1x256x256,has_mask_input:1 \
  --optShapes=image_embeddings:1x256x64x64,point_coords:1x2x2,point_labels:1x2,mask_input:1x1x256x256,has_mask_input:1 \
  --maxShapes=image_embeddings:1x256x64x64,point_coords:1x2x2,point_labels:1x2,mask_input:1x1x256x256,has_mask_input:1 \
  --fp16
