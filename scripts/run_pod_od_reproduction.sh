#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/home/room305/ZZF/yolov13-6000"
PYTHON_BIN="/home/room305/.conda/envs/yolov13/bin/python"
RUN_ID="pod_od_repro_urpc2019_20260913_r2"
RUN_ROOT="${PROJECT}/runs/${RUN_ID}"
LAUNCHER_LOG="${RUN_ROOT}/train/launcher.log"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "Run root already exists: ${RUN_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" || ! -f "${PROJECT}/tools/run_pod_od_reproduction.py" ]]; then
  echo "Required Python runtime or manager is missing." >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}/train"
echo "$$" > "${RUN_ROOT}/launcher.pid"
exec "${PYTHON_BIN}" "${PROJECT}/tools/run_pod_od_reproduction.py" --run-root "${RUN_ROOT}" >>"${LAUNCHER_LOG}" 2>&1
