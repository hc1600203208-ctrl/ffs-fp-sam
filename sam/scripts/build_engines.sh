#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRTEXEC="${TRTEXEC:-trtexec}"
DINO_HEIGHT="${DINO_HEIGHT:-800}"
DINO_WIDTH="${DINO_WIDTH:-1066}"

if ! command -v "${TRTEXEC}" >/dev/null 2>&1; then
  if [[ -x /usr/src/tensorrt/targets/x86_64-linux-gnu/bin/trtexec ]]; then
    TRTEXEC=/usr/src/tensorrt/targets/x86_64-linux-gnu/bin/trtexec
  else
    echo "trtexec not found. Set TRTEXEC=/path/to/trtexec" >&2
    exit 1
  fi
fi

mkdir -p "${ROOT}/engines"

"${TRTEXEC}" \
  --onnx="${ROOT}/models/grounding_dino_fixed_prompt.onnx" \
  --saveEngine="${ROOT}/engines/grounding_dino_fixed_prompt.engine" \
  --minShapes=images:1x3:${DINO_HEIGHT}x${DINO_WIDTH} \
  --optShapes=images:1x3:${DINO_HEIGHT}x${DINO_WIDTH} \
  --maxShapes=images:1x3:${DINO_HEIGHT}x${DINO_WIDTH} \
  --noTF32 \
  --stronglyTyped

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
