#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPI4PY_MPIABI=openmpi
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q -o junit_family=xunit1 --junitxml=output/stage7-tests.xml
.venv/bin/python -m dg_wave.volcano volcano.msh
.venv/bin/python -m dg_wave run volcano.toml
mpiexec --nooversubscribe --bind-to core --map-by core -n 4 \
    .venv/bin/python -m dg_wave run volcano.toml
.venv/bin/python - <<'PY'
from pathlib import Path
root = Path.cwd()
text = (root / "volcano.toml").read_text()
text = text.replace('path = "volcano.msh"', f'path = "{root / "volcano.msh"}"')
text = text.replace('directory = "output/volcano"', 'directory = "volcano_restart"')
(root / "output" / "volcano-restart.toml").write_text(text)
PY
.venv/bin/python -m dg_wave run output/volcano-restart.toml \
    --checkpoint output/volcano_checkpoints/serial --stop-after 1337
.venv/bin/python -m dg_wave run output/volcano-restart.toml \
    --restart output/volcano_checkpoints/serial
mpiexec --nooversubscribe --bind-to core --map-by core -n 4 \
    .venv/bin/python -m dg_wave run output/volcano-restart.toml \
    --checkpoint output/volcano_checkpoints/mpi4 --stop-after 1337
mpiexec --nooversubscribe --bind-to core --map-by core -n 4 \
    .venv/bin/python -m dg_wave run output/volcano-restart.toml \
    --restart output/volcano_checkpoints/mpi4
