#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPI4PY_MPIABI=openmpi
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q -o junit_family=xunit1 --junitxml=output/stage5-tests.xml
.venv/bin/python -m dg_wave run attenuating_box.toml
mpiexec --nooversubscribe --bind-to core --map-by core -n 4 \
    .venv/bin/python -m dg_wave run attenuating_box.toml
