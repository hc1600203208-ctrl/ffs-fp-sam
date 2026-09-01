#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/anaconda3}"
CONDA_ENV="${CONDA_ENV:-sam}"
PROMPT="${PROMPT:-golden object}"
PROMPT_NAME="${PROMPT_NAME:-golden_object}"
DINO_HEIGHT="${DINO_HEIGHT:-800}"
DINO_WIDTH="${DINO_WIDTH:-1066}"
BERT_BASE_UNCASED_PATH="${BERT_BASE_UNCASED_PATH:-}"
BUILD_DINO_TRT_ENGINE="${BUILD_DINO_TRT_ENGINE:-1}"
TRTEXEC="${TRTEXEC:-trtexec}"
TENSORRT_ROOT="${TENSORRT_ROOT:-}"
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
  --output "${ROOT}/models/grounding_dino_fixed_${PROMPT_NAME}.onnx"

if [[ "${BUILD_DINO_TRT_ENGINE}" != "1" ]]; then
  echo "Exported ${ROOT}/models/grounding_dino_fixed_${PROMPT_NAME}.onnx"
  echo "Skipping DINO TensorRT engine build. Set BUILD_DINO_TRT_ENGINE=1 to build it for validation on another target."
  exit 0
fi

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

"${TRTEXEC}" \
  --onnx="${ROOT}/models/grounding_dino_fixed_${PROMPT_NAME}.onnx" \
  --saveEngine="${ROOT}/engines/grounding_dino_fixed_${PROMPT_NAME}.engine" \
  --noTF32 \
  --stronglyTyped
