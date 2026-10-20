#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 TRAINING_RUN_DIR TRAINING_STATUS NEW_EVAL_DIR GPU_IDS_FOUR_COMMA_SEPARATED" >&2
  exit 2
fi
training_run="$1"
training_status="$2"
output_dir="$3"
gpu_ids="$4"
[[ -f "$training_status" ]] || { echo "Training status file not found" >&2; exit 2; }
[[ ! -e "$output_dir" ]] || { echo "Evaluation output must be new" >&2; exit 2; }
"$(dirname "$0")/preflight.sh"

export XAI_GPU_IDS="$gpu_ids"
python3 "$(dirname "$0")/build_eval_snapshot.py"
snapshot_sha="$(sha256sum "$(dirname "$0")/SNAPSHOT.json" | awk '{print $1}')"
python3 "$(dirname "$0")/evaluate_official.py" \
  --candidate G \
  --training-run-dir "$training_run" \
  --training-status "$training_status" \
  --output-dir "$output_dir" \
  --gpus "$gpu_ids" \
  --snapshot-sha256 "$snapshot_sha" \
  --wait-seconds 0 \
  --evaluation-timeout-seconds 86400 \
  --execute-gpu
