# DGWave

**CPU/MPI build phase complete:** a single Python/JAX implementation of three-dimensional tetrahedral DG isotropic elastic and generalized-Maxwell viscoelastic waves. **GPU execution is not implemented or validated.** Four-rank MPI was verified on one Ubuntu/WSL2 CPU host, not multiple machines.

Open [index.html](index.html) for the physics overview, the existing interactive [Rainier model](results/site/model-view.html), and the latest [R03 animation](results/site/rainier-r03-10s.gif). All showcase assets are local; no CDN or equation-rendering service is required.

Source repository: [obriengareth/DGWave](https://github.com/obriengareth/DGWave).

## Implemented

- Modepy nodal tetrahedral bases, differentiation/mass/lifting, affine geometry, stable element IDs, connectivity and triangular-face permutations; focused degree 1-3 operator checks.
- FP64 batched, matrix-free JAX velocity-stress residuals and RK4; heterogeneous welded impedance fluxes, free surfaces, periodic fixtures and local absorbing boundaries. Optional artificial sponge damping is distinct from physical attenuation; absorption is **not a PML**.
- Positive generalized-Maxwell branch stiffness, positive relaxed moduli and per-material Qp/Qs fitting. Inputs are storage moduli at the configured reference frequency, not exact finite-Q phase-speed moduli.
- CPU MPI face-trace exchange through host buffers, receivers, CSV/surface/VTU/PVTU output and atomic hash-checked same-partition full-state checkpoints.
- Synthetic P/S, convergence, interface, attenuation, restart and MPI checks; saved Rainier field/rank/hash/finite-output audits. These do not establish real-world prediction accuracy.

## Environment and setup

Verified baseline: Ubuntu 22.04 under WSL2, x86_64, Python 3.10.12, Open MPI 4.1.2; CPU JAX/JAXlib 0.6.2, NumPy 2.2.6, modepy 2026.1, mpi4py 4.1.2, meshio 5.3.5 and pytest 9.2.0. `requirements.lock` pins the complete CPU Python environment. This is not a claim of portability to every Python, MPI or operating-system version.

For a fresh checkout, run these commands **inside Ubuntu/WSL**, with Python 3.10, its venv support, Git and Open MPI already available. Clone first, then install only into the local project environment:

```bash
git clone https://github.com/obriengareth/DGWave.git
cd DGWave
export MPI4PY_MPIABI=openmpi
python3 -m venv .venv
.venv/bin/python -m pip install --no-compile -r requirements.lock
.venv/bin/python -m pip install --no-compile --no-build-isolation --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python -m dg_wave --help
```

`requirements-geo.lock` is the separate terrain-preparation environment (rasterio/pyproj); it is not needed to view the bundled results or run the small existing box examples. Native Windows startup is not equivalent to the verified WSL/MPI workflow.

## CLI

These are actual supported commands, not requests to rerun the archived Rainier job:

```bash
# FP64/backend/host-buffer diagnostics, with explicit no-oversubscription binding
mpiexec --nooversubscribe --bind-to core --map-by core -n 4 \
  .venv/bin/python -m dg_wave --config stage0.toml --diagnostics

# Small serial example
.venv/bin/python -m dg_wave run elastic_box.toml

# Four-rank viscoelastic example, saving a full final checkpoint
mpiexec --nooversubscribe --bind-to core --map-by core -n 4 \
  .venv/bin/python -m dg_wave run attenuating_box.toml \
  --checkpoint output/demo_checkpoint

# Regression suite (includes MPI subprocess checks)
.venv/bin/python -m pytest -q
```

`--config FILE` validates TOML parameters only; it does not load the mesh or prove CFL safety. The solve path performs geometry/material/timestep checks. Ensure enough physical cores and memory are available; the ordinary CLI is not the laptop-specific guarded Rainier launcher.

### Restart and end-time extension

`run FILE --restart DIR --checkpoint NEW_DIR` resumes a compatible full checkpoint with the **same process count/partition**. `--stop-after STEP` is an absolute step and requires `--checkpoint`. Checkpoints are written at the invocation's final step, not as a promise of periodic crash recovery.

To continue beyond a configured end, first copy the original TOML to `continuation.toml` **in the same directory**, increase only `[time].steps` among the checked physics/schedule settings, and choose a separate `[output].directory`. Then:

```bash
mpiexec --nooversubscribe --bind-to core --map-by core -n 4 \
  .venv/bin/python -m dg_wave run continuation.toml \
  --restart output/demo_checkpoint --extend-run-end \
  --checkpoint output/demo_continued_checkpoint
```

Keep the mesh bytes, timestep, materials, attenuation, receiver/output cadence and source history unchanged. In particular, **do not extend a source's duration/tapers or reapply it** just because the run end increases. `--extend-run-end` permits only a monotonic increase of the checked step limit; it is not repartitioning or arbitrary physics editing. Surface-frame history is stored separately from checkpoints: preserving/combining earlier surface frames requires an explicit copy/manifest audit, as done for Rainier.

## Code layout

| Location | Responsibility |
|---|---|
| `dg_wave/config.py`, `__main__.py`, `runtime.py` | TOML/CLI, FP64 backend, MPI and CPU affinity |
| `mesh.py`, `reference.py`, `operators.py`, `geometry.py`, `connectivity.py` | Tetrahedral DG geometry and operators |
| `physics.py`, `solver.py`, `attenuation.py`, `sources.py` | Elastic fluxes, shared RK4 kernels, Maxwell memory and sources |
| `partition.py`, `halo.py`, `distributed.py`, `mpi_run.py` | CPU MPI partitioning, face traces and distributed runner |
| `run.py`, `checkpoint.py`, `surface.py`, `snapshots.py` | Serial runner, state/history persistence and exports |
| `terrain.py`, `rainier_layers.py`, `scripts/` | Terrain/material preparation and numerical studies |
| `tests/`, root `*.toml` and small mesh fixtures | Focused verification and runnable examples |
| `results/site/`, `index.html` | Portable existing mesh, animation, still frame and public provenance |

## Rainier demonstration and limitations

The completed run uses 150,000 affine tetrahedra, p=2 (10 nodes/element), six simplified effective materials and five Maxwell mechanisms: **39 FP64 components per node**. The footprint is 25 x 25 km; the retained summit-to-flat-base extent is 6 km, with variable local depth. The simulation terrain has 500 m vertices, derived from Copernicus GLO-30 DSM, not a 30 m simulation mesh or bare-earth survey.

- Four pinned CPU MPI ranks on one WSL2 host. `dt = 0.004/17` s; 42,500 steps to 10 s, continued from the saved 6 s state.
- Ten stations each have 2,501 finite three-component velocity samples at 0.004 s cadence; 101 finite ordered surface frames at 0.1 s include 0 and 10 s.
- Source: double couple strike/dip/rake 0/90/+90 degrees, net moment `1e12 N m`, Gaussian sigma 750 m, 3 km below the retained summit. Its original normalized/tapered **six-second** history is exactly zero from 6 to 10 s.
- Nominal source/Q band 0.01-2 Hz, Q reference 0.5 Hz; maximum realized Q-fit relative errors approximately 2.73% Qp and 2.34% Qs. This fit is not a mesh-resolution or physical-accuracy certificate.
- **Intentionally exploratory/under-resolved above approximately 0.23 Hz.** The 0.01 Hz period is 100 s, unresolved by this short source/record. Finite-duration tapering introduces leakage; the pulse is not strictly band-limited and causal at once.
- Layers, terrain-following contacts, density/Vs/Q and effective glacier/vegetation surface are modelling assumptions, **not calibrated Rainier geology**. Local absorbing boundaries can reflect oblique waves.

The latest GIF uses **R03 (CSV index 2)**, all 101 frames at **5 fps / 200 ms**, infinite looping, and a fixed **+/-1 micrometre/s** surface scale (**not mm/s**). Larger amplitudes deliberately saturate the colour mapping; raw full-run extrema are approximately -6.799383 to +7.174364 micrometres/s. The R03 seismogram independently spans +/-2.2 micrometres/s and covers its 1.967000104 micrometre/s peak without clipping. Illumination changes brightness, not the reported velocity. Playback per loop is 20.2 s.

## Portable showcase and source distribution

Open `index.html` directly, or serve the project locally:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8000/`. The animation is opt-in to avoid automatic motion and a 54.3 MiB download. The model page bundles actual geometry; its viewer is not a volume-field export.

The source distribution contains solver code, tests, locked requirements, package metadata, small example configs/meshes, this README/index and `results/site/`. Large runtime checkpoints, full-volume outputs and raw terrain caches are not distributed. Generate example outputs locally using the CLI; the Rainier showcase is a visualization bundle, not a complete restart archive.

The GIF is 56,920,810 bytes (54.3 MiB). The presentation can be served as static files. GitHub's repository file view displays HTML source; use the local server above or a static website host to view the interactive showcase.

## Attribution and licence

Terrain is a derived Copernicus GLO-30 product: produced using Copernicus WorldDEM-30 (C) DLR e.V. 2010-2014 and (C) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. [Dataset DOI](https://doi.org/10.5270/ESA-c5d3d65) and [dataset/licence information](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM). No code licence has been selected; public visibility alone does not grant an open-source licence.
