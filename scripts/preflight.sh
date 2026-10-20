#!/usr/bin/env bash
set -euo pipefail

need() {
  if [[ -z "${!1:-}" ]]; then
    echo "Required environment variable is unset: $1" >&2
    exit 2
  fi
}

for name in XAI_PROJECT_ROOT XAI_FORMAL_ROOT XAI_PREPARED_ROOT XAI_TRAIN_ANNOTATIONS XAI_TRAIN_VENV XAI_OFFICIAL_VENV CUDA_HOME; do
  need "$name"
done

for path in "$XAI_PROJECT_ROOT" "$XAI_FORMAL_ROOT" "$XAI_PREPARED_ROOT" "$XAI_TRAIN_ANNOTATIONS" "$XAI_TRAIN_VENV" "$XAI_OFFICIAL_VENV" "$CUDA_HOME"; do
  if [[ ! -e "$path" ]]; then
    echo "Configured path does not exist: $path" >&2
    exit 2
  fi
done

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required on the training host" >&2; exit 2; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 2; }
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
echo "Path/dependency presence preflight complete. This does not validate model hashes, GPU exclusivity, or training readiness."
