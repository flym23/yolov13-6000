#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT='/home/room305/ZZF/yolov11n-6000'
PYTHON='/home/room305/.conda/envs/yolov13/bin/python3'
VARIANT="${1:?Specify A0/A1/A2/A3/A4}"
RUN_ID="${2:?Specify immutable run_id}"
shift 2
export OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_DISABLED=true
export PIN_MEMORY=false
MANAGER="$PROJECT/tools/uw_chain.py"
if [[ "$VARIANT" == 'A4' && "$RUN_ID" == 'urpc2020_20260916_r1' ]]; then
  MANAGER="$PROJECT/tools/uw180/uw_chain.py"
fi
exec "$PYTHON" -u "$MANAGER" run --variant "$VARIANT" --run-id "$RUN_ID" "$@"
