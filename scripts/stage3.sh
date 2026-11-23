#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPI4PY_MPIABI=openmpi
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
mpiexec --nooversubscribe --bind-to core --map-by core -n 4 \
    .venv/bin/python -m dg_wave run elastic_box.toml
