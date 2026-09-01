#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/anaconda3}"
CONDA_ENV="${CONDA_ENV:-sam}"
BERT_BASE_UNCASED_PATH="${BERT_BASE_UNCASED_PATH:-${ROOT}/models/bert-base-uncased}"
HF_MODEL_ID="${HF_MODEL_ID:-bert-base-uncased}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${BERT_BASE_UNCASED_PATH}"

python - "${HF_MODEL_ID}" "${BERT_BASE_UNCASED_PATH}" <<'PY'
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is not installed. Install it in the active conda env with: "
        "pip install huggingface_hub"
    ) from exc

model_id = sys.argv[1]
local_dir = Path(sys.argv[2]).resolve()

snapshot_download(
    repo_id=model_id,
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
    allow_patterns=[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
        "pytorch_model.bin",
        "model.safetensors",
    ],
)

required = ["config.json", "vocab.txt", "tokenizer_config.json"]
missing = [name for name in required if not (local_dir / name).exists()]
has_weights = (local_dir / "pytorch_model.bin").exists() or (local_dir / "model.safetensors").exists()
if missing or not has_weights:
    detail = []
    if missing:
        detail.append("missing files: " + ", ".join(missing))
    if not has_weights:
        detail.append("missing BERT weights: pytorch_model.bin or model.safetensors")
    raise SystemExit("; ".join(detail))

print(f"Downloaded {model_id} to {local_dir}")
PY

echo "Use this path for offline export:"
echo "  BERT_BASE_UNCASED_PATH=${BERT_BASE_UNCASED_PATH}"
