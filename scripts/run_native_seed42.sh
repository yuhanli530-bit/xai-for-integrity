#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 GPU_IDS_FOUR_COMMA_SEPARATED NEW_RUN_ROOT" >&2
  exit 2
fi

gpu_ids="$1"
run_root="$2"
[[ ! -e "$run_root" ]] || { echo "Run root must be new: $run_root" >&2; exit 2; }
"$(dirname "$0")/preflight.sh"

reason_module="$(cd "$(dirname "$0")/../src/xai_integrity" && pwd)/reason_grounding.py"
reason_sha="$(sha256sum "$reason_module" | awk '{print $1}')"
export PYTHONPATH="$(cd "$(dirname "$0")/../src/xai_integrity" && pwd):${PYTHONPATH:-}"
export XAI_GPU_IDS="$gpu_ids"
export XAI_EXPLANATION_DATA="$(cd "$(dirname "$0")/../src/xai_integrity" && pwd)/explanation_data.py"

python3 "$(dirname "$0")/launch_native.py" \
  --mode main \
  --reason-module "$reason_module" \
  --reason-module-sha256 "$reason_sha" \
  --gpus "$gpu_ids" \
  --run-root "$run_root" \
  --master-port "${MASTER_PORT:-31443}" \
  --execute-gpu
