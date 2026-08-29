#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/anaconda3}"
CONDA_ENV="${CONDA_ENV:-sam}"
PROMPT="${PROMPT:-blue carton}"
PROMPT_NAME="${PROMPT_NAME:-blue_carton}"
DINO_HEIGHT="${DINO_HEIGHT:-800}"
DINO_WIDTH="${DINO_WIDTH:-1066}"
TRTEXEC="${TRTEXEC:-trtexec}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${ROOT}/models"

python "${ROOT}/scripts/export_onnx.py" grounding_dino \
  --device cpu \
  --text_prompt "${PROMPT}" \
  --dino_height "${DINO_HEIGHT}" \
  --dino_width "${DINO_WIDTH}" \
  --fixed_mask \
  --output "${ROOT}/models/grounding_dino_fixed_${PROMPT_NAME}.onnx"

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
  --onnx="${ROOT}/models/grounding_dino_fixed_${PROMPT_NAME}.onnx" \
  --saveEngine="${ROOT}/engines/grounding_dino_fixed_${PROMPT_NAME}.engine" \
  --minShapes=images:1x3x${DINO_HEIGHT}x${DINO_WIDTH} \
  --optShapes=images:1x3x${DINO_HEIGHT}x${DINO_WIDTH} \
  --maxShapes=images:1x3x${DINO_HEIGHT}x${DINO_WIDTH} \
  --noTF32 \
  --stronglyTyped
