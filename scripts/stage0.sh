#!/usr/bin/env bash
# Run from WSL Ubuntu 22.04: bash scripts/stage0.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export MPI4PY_MPIABI=openmpi
python3 -c 'import sys; assert sys.version_info[:2] == (3, 10), "Lock targets Python 3.10"'
python3 -m venv .venv
# Avoid bytecode compilation races on the Windows/OneDrive mount.
.venv/bin/python -m pip install --quiet --disable-pip-version-check --no-compile -r requirements.lock
.venv/bin/python -m pip install --quiet --disable-pip-version-check --no-compile --no-build-isolation --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python -m dg_wave --help
.venv/bin/python -m pytest -q
.venv/bin/python -c 'import jax, numpy, modepy, mpi4py.MPI, meshio'
for ranks in 1 4; do
    mpiexec --nooversubscribe --bind-to core --map-by core -n "$ranks" \
        .venv/bin/python -m dg_wave --config stage0.toml --diagnostics
done
