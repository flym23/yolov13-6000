#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT='/home/room305/ZZF/yolov11n-6000'
PYTHON='/home/room305/.conda/envs/yolov13/bin/python3'
VARIANT="${1:?Specify C0/C1/C2/C3}"
RUN_ID="${2:?Specify immutable run_id}"
shift 2
export OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_DISABLED=true
export PIN_MEMORY=false
exec "$PYTHON" -u "$PROJECT/tools/uwv3/chain.py" run --variant "$VARIANT" --run-id "$RUN_ID" "$@"
