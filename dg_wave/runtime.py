"""CPU FP64 diagnostics; future MPI exchanges use contiguous NumPy host buffers."""

import os
import socket
import sys
from pathlib import Path
from typing import Any


_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
)
_THREADED_CPUS: tuple[int, ...] | None = None


def cpu_topology(cpu: int) -> list[int]:
    directory = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    return [int((directory / name).read_text()) for name in ("physical_package_id", "core_id")]


def physical_core_cpus(count: int) -> list[int]:
    if type(count) is not int or count < 1:
        raise ValueError("physical core count must be a positive integer")
    result, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        core = tuple(cpu_topology(cpu))
        if core not in seen:
            result.append(cpu)
            seen.add(core)
        if len(result) == count:
            return result
    raise ValueError(f"{count} distinct physical cores unavailable within current affinity")


def initialize_threaded_cpu(cpus: list[int]) -> dict[str, Any]:
    """Explicit single-process mode; affinity precedes numerical library imports."""
    global _THREADED_CPUS
    if any(name in sys.modules for name in ("jax", "numpy", "mpi4py.MPI")):
        raise RuntimeError("threaded CPU mode must be selected before JAX/NumPy/MPI imports")
    if any(name in os.environ for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK")):
        raise RuntimeError("threaded CPU mode must run directly, without an MPI launcher")
    available = os.sched_getaffinity(0)
    if not cpus or len(set(cpus)) != len(cpus) or not set(cpus) <= available:
        raise ValueError("threaded CPU mask must be nonempty, distinct and inside current affinity")
    cores = [cpu_topology(cpu) for cpu in cpus]
    check_physical_cores(cores)
    flags = "--xla_cpu_multi_thread_eigen=true"
    if os.environ.get("XLA_FLAGS", flags) != flags:
        raise ValueError("threaded CPU mode requires an unset XLA_FLAGS or its explicit supported Eigen flag")
    os.sched_setaffinity(0, set(cpus))
    _THREADED_CPUS = tuple(cpus)
    os.environ["XLA_FLAGS"] = flags
    configure_cpu_environment()
    return {
        "mode": "single_process_threaded_cpu", "pid": os.getpid(),
        "logical_cpu_mask": sorted(os.sched_getaffinity(0)), "socket_cores": cores,
        "thread_environment": {key: os.environ[key] for key in _THREAD_VARIABLES},
        "xla_flags": os.environ["XLA_FLAGS"],
    }


def require_cpu_backend(backend: str) -> None:
    if backend != "cpu":
        raise ValueError(
            f"runtime.backend={backend!r} is unavailable in this CPU-only build; "
            "refusing CPU fallback"
        )


def check_affinities(affinities: list[list[int] | None]) -> None:
    if len(affinities) > 1 and any(mask is None for mask in affinities):
        raise RuntimeError("CPU affinity unavailable: cannot verify MPI oversubscription")
    used: set[int] = set()
    for mask in affinities:
        if mask is None:
            continue
        if not mask or used.intersection(mask):
            raise RuntimeError(
                "MPI CPU affinities overlap or are empty; launch with "
                "mpiexec --nooversubscribe --bind-to core --map-by core"
            )
        used.update(mask)


def check_physical_cores(cores) -> None:
    if len(set(map(tuple, cores))) != len(cores):
        raise RuntimeError("MPI ranks share a physical CPU core (including SMT siblings)")


def physical_binding(comm, local_rank):
    """Verify this job's effective one-logical-CPU binding and node-local cores."""
    from mpi4py import MPI

    affinity = sorted(os.sched_getaffinity(0))
    record, error = None, None
    try:
        if len(affinity) != 1:
            raise RuntimeError("wavefield runs require one pinned logical CPU per MPI rank")
        core = cpu_topology(affinity[0])
        threads = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
        if any(value != "1" for value in threads.values()):
            raise RuntimeError("native thread limits must all equal one")
        record = {
            "rank": comm.rank, "local_rank": local_rank, "host": socket.gethostname(),
            "logical_cpu_mask": affinity, "socket_core": core, "thread_environment": threads,
        }
    except (OSError, ValueError, RuntimeError) as exc:
        error = str(exc)
    errors = comm.allgather(error)
    if any(errors):
        raise RuntimeError("; ".join(item for item in errors if item))
    local = comm.Split_type(MPI.COMM_TYPE_SHARED)
    cores = local.allgather(record["socket_core"])
    local.Free()
    error = None
    try:
        check_physical_cores(cores)
    except RuntimeError as exc:
        error = str(exc)
    errors = comm.allgather(error)
    if any(errors):
        raise RuntimeError("; ".join(item for item in errors if item))
    return record


def configure_cpu_environment() -> None:
    """Call before importing JAX/NumPy to constrain their compute thread pools."""
    for variable in _THREAD_VARIABLES:
        os.environ[variable] = "1"
    if _THREADED_CPUS is not None:
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(len(_THREADED_CPUS))
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"


def initialize_mpi_cpu():
    """Verify disjoint shared-node MPI affinities, then pin one CPU per rank."""
    if _THREADED_CPUS is not None:
        raise RuntimeError("threaded CPU mode cannot initialize MPI; use a fresh MPI process")
    configure_cpu_environment()
    # This WSL environment has both MPICH and Open MPI; the launcher is Open MPI.
    os.environ["MPI4PY_MPIABI"] = "openmpi"

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    local = comm.Split_type(MPI.COMM_TYPE_SHARED)
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    local_affinities = local.allgather(affinity)
    error = None
    try:
        check_affinities(local_affinities)
        if affinity is not None:
            os.sched_setaffinity(0, {affinity[0]})
    except (RuntimeError, OSError) as exc:
        error = str(exc)
    errors = comm.allgather(error)
    local_rank = local.Get_rank()
    local.Free()
    if any(errors):
        raise RuntimeError("; ".join(sorted({item for item in errors if item})))
    return comm, affinity, local_rank


def require_single_process() -> None:
    """Convergence analysis is serial; production solves support MPI."""
    comm, _, _ = initialize_mpi_cpu()
    if comm.Get_size() != 1:
        raise RuntimeError("this analysis command supports exactly one MPI rank")


def diagnostics() -> list[dict[str, Any]] | None:
    comm, affinity, local_rank = initialize_mpi_cpu()
    from mpi4py import MPI

    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    value = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    result = jax.jit(lambda array: array * 2.0)(value)
    host = np.ascontiguousarray(jax.device_get(result), dtype=np.float64)
    device = next(iter(result.devices()))
    if jax.default_backend() != "cpu" or str(result.dtype) != "float64":
        raise RuntimeError("Stage 0 requires an actual CPU backend and float64 array")
    if not np.array_equal(host, [2.0, 4.0]):
        raise RuntimeError("FP64 startup computation failed")
    # Exercise a host-buffer collective, not a device-aware MPI code path.
    total = np.empty_like(host)
    comm.Allreduce(host, total, op=MPI.SUM)
    if not np.array_equal(total, host * comm.Get_size()):
        raise RuntimeError("MPI host-buffer reduction failed")
    record = {
        "rank": comm.Get_rank(),
        "mpi_size": comm.Get_size(),
        "local_rank": local_rank,
        "host": socket.gethostname(),
        "backend": jax.default_backend(),
        "dtype": str(result.dtype),
        "device": str(device),
        "local_device_count": jax.local_device_count(),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0)) if affinity is not None else None
        ),
        "launcher_cpu_affinity": affinity,
        "thread_env_limit": 1,
        "host_buffer_dtype": str(host.dtype),
        "host_buffer_reduction": total.tolist(),
        "mpi_library": MPI.Get_library_version().rstrip("\0 \n"),
        "jax_version": jax.__version__,
    }
    return comm.gather(record, root=0)
