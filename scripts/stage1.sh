#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPI4PY_MPIABI=openmpi
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
for degree in 1 2 3; do
    .venv/bin/python -m dg_wave.benchmark --degree "$degree" --copies 1024 --repeats 10
done
