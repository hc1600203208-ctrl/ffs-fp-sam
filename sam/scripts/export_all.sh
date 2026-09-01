#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/anaconda3}"
CONDA_ENV="${CONDA_ENV:-sam}"
PROMPT="${PROMPT:-blue carton}"
DINO_HEIGHT="${DINO_HEIGHT:-800}"
DINO_WIDTH="${DINO_WIDTH:-1066}"
BERT_BASE_UNCASED_PATH="${BERT_BASE_UNCASED_PATH:-}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE HF_HUB_OFFLINE

resolve_bert_path() {
  local path="$1"
  local snapshot=""

  if [[ -d "${path}/snapshots" ]]; then
    if [[ -f "${path}/refs/main" ]]; then
      snapshot="$(cat "${path}/refs/main")"
      if [[ -d "${path}/snapshots/${snapshot}" ]]; then
        echo "${path}/snapshots/${snapshot}"
        return 0
      fi
    fi
    snapshot="$(find "${path}/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)"
    if [[ -n "${snapshot}" ]]; then
      echo "${snapshot}"
      return 0
    fi
  fi

  echo "${path}"
}

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${ROOT}/models"

if [[ -z "${BERT_BASE_UNCASED_PATH}" ]]; then
  if [[ -d "${ROOT}/models/bert-base-uncased" ]]; then
    BERT_BASE_UNCASED_PATH="${ROOT}/models/bert-base-uncased"
  else
    BERT_BASE_UNCASED_PATH="${HOME}/.cache/huggingface/hub/models--bert-base-uncased"
  fi
fi

BERT_BASE_UNCASED_PATH="$(resolve_bert_path "${BERT_BASE_UNCASED_PATH}")"

if [[ ! -f "${BERT_BASE_UNCASED_PATH}/config.json" ||
      ! -f "${BERT_BASE_UNCASED_PATH}/vocab.txt" ||
      ! -f "${BERT_BASE_UNCASED_PATH}/tokenizer_config.json" ]] ||
   [[ ! -f "${BERT_BASE_UNCASED_PATH}/pytorch_model.bin" &&
      ! -f "${BERT_BASE_UNCASED_PATH}/model.safetensors" ]]; then
  echo "Local bert-base-uncased files are missing under: ${BERT_BASE_UNCASED_PATH}" >&2
  echo "Run ./scripts/download_bert_base_uncased.sh first, or set BERT_BASE_UNCASED_PATH=/path/to/bert-base-uncased." >&2
  exit 1
fi

python "${ROOT}/scripts/export_onnx.py" grounding_dino \
  --device cpu \
  --text_prompt "${PROMPT}" \
  --bert_base_uncased_path "${BERT_BASE_UNCASED_PATH}" \
  --dino_height "${DINO_HEIGHT}" \
  --dino_width "${DINO_WIDTH}" \
  --fixed_mask \
  --static_batch \
  --output "${ROOT}/models/grounding_dino_fixed_prompt.onnx"

python "${ROOT}/scripts/export_onnx.py" sam_encoder \
  --device cpu \
  --output "${ROOT}/models/sam_image_encoder.onnx"

python "${ROOT}/scripts/export_onnx.py" sam \
  --device cpu \
  --output "${ROOT}/models/sam_mask_decoder.onnx"

echo "Exported ONNX models into ${ROOT}/models"
