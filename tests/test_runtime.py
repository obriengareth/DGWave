import json
from pathlib import Path
import subprocess
import sys

import pytest

from dg_wave.runtime import check_affinities, physical_core_cpus


def test_physical_core_selection_excludes_smt(monkeypatch):
    from dg_wave import runtime

    monkeypatch.setattr(runtime.os, "sched_getaffinity", lambda pid: set(range(8)))
    monkeypatch.setattr(runtime, "cpu_topology", lambda cpu: [0, cpu//2])
    assert physical_core_cpus(4) == [0, 2, 4, 6]
    with pytest.raises(ValueError, match="unavailable"):
        physical_core_cpus(5)


@pytest.mark.parametrize("cores", [1, 2, 4])
def test_explicit_threaded_mode_stays_bound_and_refuses_mpi(cores):
    code = """
import json,os
from dg_wave.runtime import initialize_threaded_cpu,physical_core_cpus,configure_cpu_environment,initialize_mpi_cpu
record=initialize_threaded_cpu(physical_core_cpus(CORES))
configure_cpu_environment()
assert os.environ['TF_NUM_INTRAOP_THREADS']==str(CORES)
assert os.environ['OPENBLAS_NUM_THREADS']==os.environ['OMP_NUM_THREADS']=='1'
try:
    initialize_mpi_cpu()
except RuntimeError as error:
    assert 'cannot initialize MPI' in str(error)
else:
    raise AssertionError('threaded process initialized MPI')
import jax
import jax.numpy as jnp
value=jax.jit(lambda x:x*2)(jnp.ones(8,dtype=jnp.float64)).block_until_ready()
assert str(value.dtype)=='float64' and jax.default_backend()=='cpu'
from pathlib import Path
allowed=set(record['logical_cpu_mask'])
assert all(set(os.sched_getaffinity(int(p.name)))<=allowed for p in Path('/proc/self/task').iterdir())
print(json.dumps(record))
""".replace("CORES", str(cores))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=True)
    record = json.loads(result.stdout)
    assert len(record["logical_cpu_mask"]) == cores
    assert len(set(map(tuple, record["socket_cores"]))) == cores


@pytest.mark.parametrize("setup", ["import numpy", "import os;os.environ['OMPI_COMM_WORLD_SIZE']='1'"])
def test_threaded_mode_rejects_late_selection_and_mpi_launcher(setup):
    code = setup + "\nfrom dg_wave.runtime import initialize_threaded_cpu,physical_core_cpus\ninitialize_threaded_cpu(physical_core_cpus(1))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "before JAX/NumPy/MPI" in result.stderr or "without an MPI launcher" in result.stderr


@pytest.mark.parametrize("masks", [
    [[0], [1], [2], [3]], [[0, 1], [2, 3]], [None],
])
def test_safe_affinities(masks):
    check_affinities(masks)


@pytest.mark.parametrize("masks", [
    [[0, 1], [1, 2]], [[0], [0]], [[], [1]], [None, None],
])
def test_oversubscribed_or_unverifiable_affinities(masks):
    with pytest.raises(RuntimeError, match="affinit"):
        check_affinities(masks)


@pytest.mark.parametrize("ranks", [1, 4])
def test_cpu_fp64_mpi_startup(ranks):
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            "mpiexec", "--nooversubscribe", "--bind-to", "core", "--map-by", "core",
            "-n", str(ranks), sys.executable, "-m", "dg_wave",
            "--config", str(root / "stage0.toml"), "--diagnostics",
        ],
        capture_output=True, text=True, timeout=60, check=True,
    )
    records = json.loads(result.stdout)["ranks"]
    assert len(records) == ranks
    assert [record["rank"] for record in records] == list(range(ranks))
    assert [record["local_rank"] for record in records] == list(range(ranks))
    for record in records:
        assert record["mpi_size"] == ranks
        assert record["backend"] == "cpu"
        assert record["dtype"] == record["host_buffer_dtype"] == "float64"
        assert record["host_buffer_reduction"] == [2.0 * ranks, 4.0 * ranks]
        assert record["local_device_count"] == 1
        assert len(record["cpu_affinity"]) == 1
        assert "Open MPI" in record["mpi_library"]
    assert len({record["cpu_affinity"][0] for record in records}) == ranks


def test_unbound_mpi_rejected():
    result = subprocess.run(
        [
            "mpiexec", "--nooversubscribe", "--bind-to", "none", "-n", "2",
            sys.executable, "-m", "dg_wave", "--diagnostics",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "CPU affinities overlap" in result.stderr


@pytest.mark.parametrize("spatial_q", [False, True])
def test_threaded_viscoelastic_numerical_integrity_across_core_masks(tmp_path, spatial_q):
    import numpy as np

    code = """
import os,sys
from dg_wave.runtime import initialize_threaded_cpu,physical_core_cpus
record=initialize_threaded_cpu(physical_core_cpus(int(sys.argv[1])))
import numpy as np
from dg_wave.config import Material,Attenuation
from dg_wave.mesh import box_mesh,TetraMesh
from dg_wave.operators import prepare_mesh
from dg_wave.solver import ElasticSolver
mesh=box_mesh()
targets=()
if SPATIAL_Q:
    mesh=TetraMesh(mesh.vertices,mesh.cells,mesh.element_ids,np.arange(len(mesh.cells))%2+1)
    targets=((1,60.,40.),(2,150.,100.))
solver=ElasticSolver(prepare_mesh(mesh,2),Material(1,8/3,1),attenuation=Attenuation((5,20),10,60,40,3,targets))
initial=np.random.default_rng(42).normal(size=solver.state_shape)
final=np.asarray(solver.advance(initial,.2*solver.max_timestep,4))
np.save(sys.argv[2],final)
from pathlib import Path
assert all(set(os.sched_getaffinity(int(p.name)))<=set(record['logical_cpu_mask']) for p in Path('/proc/self/task').iterdir())
""".replace("SPATIAL_Q", repr(spatial_q))
    results = []
    for cores in (1, 4):
        path = tmp_path / f"cores{cores}.npy"
        subprocess.run(
            [sys.executable, "-c", code, str(cores), str(path)],
            capture_output=True, text=True, timeout=40, check=True,
        )
        results.append(np.load(path))
    assert results[0].shape[-1] == 27 and np.max(np.abs(results[0][..., 9:])) > 0
    np.testing.assert_allclose(results[1], results[0], atol=1e-10, rtol=1e-10)
