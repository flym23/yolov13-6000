#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT='/home/room305/ZZF/yolov11n-6000'
PYTHON='/home/room305/.conda/envs/yolov13/bin/python3'
VARIANT="${1:?Specify downstream variant}"
RUN_ID="${2:?Specify immutable run_id}"
UPSTREAM="${3:?Specify absolute upstream state.json}"
# The Python waiter logs terminal status/reason, then execs run_yolo11uw_v2.sh.
exec "$PYTHON" -u "$PROJECT/tools/uwv2/chain.py" wait --variant "$VARIANT" --run-id "$RUN_ID" --upstream "$UPSTREAM"
