#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/anaconda3}"
CONDA_ENV="${CONDA_ENV:-sam}"
PROMPT="${PROMPT:-blue carton}"
DINO_HEIGHT="${DINO_HEIGHT:-800}"
DINO_WIDTH="${DINO_WIDTH:-1066}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${ROOT}/models"

python "${ROOT}/scripts/export_onnx.py" grounding_dino \
  --device cpu \
  --text_prompt "${PROMPT}" \
  --dino_height "${DINO_HEIGHT}" \
  --dino_width "${DINO_WIDTH}" \
  --fixed_mask \
  --output "${ROOT}/models/grounding_dino_fixed_prompt.onnx"

python "${ROOT}/scripts/export_onnx.py" sam_encoder \
  --device cpu \
  --output "${ROOT}/models/sam_image_encoder.onnx"

python "${ROOT}/scripts/export_onnx.py" sam \
  --device cpu \
  --output "${ROOT}/models/sam_mask_decoder.onnx"

echo "Exported ONNX models into ${ROOT}/models"
