"""Blocking host-buffer MPI scheduling around the shared JAX elastic kernels."""

import math
import time

import numpy as np

from dg_wave.halo import HaloPlan
from dg_wave.attenuation import fit_relaxation
from dg_wave.materials import element_material
from dg_wave.operators import _jax_runtime
from dg_wave.partition import Partition
from dg_wave.solver import ElasticSolver

jax = _jax_runtime()
import jax.numpy as jnp


@jax.jit
def _stage(base, derivative, factor):
    return base + factor * derivative


@jax.jit
def _combine(base, k1, k2, k3, k4, dt):
    return base + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def collective_error(comm, error: str | None) -> None:
    errors = comm.allgather(error)
    if any(errors):
        raise ValueError("; ".join(sorted({item for item in errors if item})))


class DistributedSolver:
    def __init__(
        self, partition: Partition, material, comm, boundaries, source=None,
        *, free_surface=False, sponge=None, attenuation=None, free_surface_tags=(),
    ):
        from mpi4py import MPI

        self.comm = comm
        self.partition = partition
        self.prepared = partition.prepare()
        material = partition.material if partition.material is not None else material
        if material is None:
            raise ValueError("distributed solve requires assigned element materials")
        reference = element_material(material, len(partition.mesh.cells))
        model = fit_relaxation(reference, attenuation, physical_tags=partition.mesh.physical_tags)
        self.material = model.unrelaxed if model is not None else reference
        self.halo = HaloPlan(partition, comm)
        neighbour_material = self.halo.exchange_materials(self.material)
        self.kernel = ElasticSolver(
            self.prepared, reference, boundaries, source, connection=partition.connection,
            neighbour_material=neighbour_material, free_surface=free_surface,
            sponge=sponge, domain_bounds=partition.bounds, relaxation=model,
            free_surface_tags=free_surface_tags,
        )
        self.local_timestep = self.kernel.max_timestep
        global_dt = np.empty((), dtype=np.float64)
        comm.Allreduce(np.asarray(self.local_timestep), global_dt, op=MPI.MIN)
        self.max_timestep = float(global_dt)
        self.counts = np.asarray(comm.allgather(len(partition.mesh.cells)), dtype=np.int64)
        self.displacements = np.concatenate(([0], np.cumsum(self.counts[:-1])))
        self.compiled = False
        self.closed = False
        self.volume_gathers = 0

    def compile(self) -> float:
        from mpi4py import MPI

        if self.compiled:
            return 0.0
        state = jnp.zeros(self.kernel.state_shape, dtype=jnp.float64)
        faces = np.zeros(self.halo.shape, dtype=np.float64)
        start = time.perf_counter()
        self._trace = self.kernel.remote_trace_kernel.lower(state).compile()
        self._residual = self.kernel.distributed_residual_kernel.lower(state, 0.0, faces).compile()
        self._stage = _stage.lower(state, state, 0.0).compile()
        self._combine = _combine.lower(state, state, state, state, state, 0.0).compile()
        self._energy = self.kernel.energy.lower(state).compile()
        elapsed = time.perf_counter() - start
        self.compiled = True
        return self.comm.allreduce(elapsed, op=MPI.MAX)

    def rhs(self, state, t):
        if not self.compiled or self.closed:
            raise RuntimeError("distributed kernels must be compiled and open before stepping")
        traces = np.asarray(self._trace(state).block_until_ready())
        right = self.halo.exchange(traces)
        # A CPU transfer may alias the NumPy receive buffer. Finish its consumer
        # before the next RK stage can overwrite any exchange buffers.
        return self._residual(state, t, right).block_until_ready()

    def step(self, state, t, dt):
        k1 = self.rhs(state, t)
        k2 = self.rhs(self._stage(state, k1, dt / 2), t + dt / 2)
        k3 = self.rhs(self._stage(state, k2, dt / 2), t + dt / 2)
        k4 = self.rhs(self._stage(state, k3, dt), t + dt)
        return self._combine(state, k1, k2, k3, k4, dt).block_until_ready()

    def validate_advance(self, state, dt, steps, t):
        error = None
        if not math.isfinite(dt) or dt <= 0 or dt > self.max_timestep * (1 + 1e-14):
            error = f"dt must be positive and <= global conservative limit {self.max_timestep:.8g}"
        elif type(steps) is not int or steps < 1:
            error = "steps must be a positive integer"
        elif np.shape(state) != self.kernel.state_shape or not math.isfinite(t):
            error = f"state must have local shape {self.kernel.state_shape} and finite time"
        collective_error(self.comm, error)
        parameters = self.comm.allgather((float(dt), steps, float(t)))
        if len(set(parameters)) != 1:
            raise ValueError("all ranks must advance with identical timestep, step count and time")

    def advance(self, state, dt: float, steps: int, t: float = 0.0):
        self.validate_advance(state, dt, steps, t)
        if not self.compiled:
            self.compile()
        state = jnp.asarray(state, dtype=jnp.float64)
        for step in range(steps):
            state = self.step(state, t + step * dt, dt)
        return state

    def energy(self, state) -> float:
        from mpi4py import MPI

        local = np.asarray(self._energy(state), dtype=np.float64)
        total = np.empty((), dtype=np.float64)
        self.comm.Allreduce(local, total, op=MPI.SUM)
        return float(total)

    def gather_state(self, state):
        from mpi4py import MPI

        self.volume_gathers += 1
        ids = np.empty(int(self.counts.sum()), dtype=np.int64) if self.comm.rank == 0 else None
        host = np.ascontiguousarray(state, dtype=np.float64)
        stride = int(np.prod(host.shape[1:]))
        data = np.empty((int(self.counts.sum()), *host.shape[1:])) if self.comm.rank == 0 else None
        self.comm.Gatherv(
            [self.partition.mesh.element_ids, MPI.INT64_T],
            [ids, self.counts, self.displacements, MPI.INT64_T] if ids is not None else None,
            root=0,
        )
        self.comm.Gatherv(
            [host.ravel(), MPI.DOUBLE],
            [data, self.counts * stride, self.displacements * stride, MPI.DOUBLE] if data is not None else None,
            root=0,
        )
        if ids is None:
            return None
        order = np.argsort(ids)
        return ids[order], data[order]

    def close(self):
        if not self.closed:
            self.halo.close()
            self.closed = True


class SurfacePlan:
    """Gather selected face nodes, never a volume state, in stable ID/face order."""

    def __init__(self, prepared, mask, comm, *, components=slice(None)):
        from mpi4py import MPI
        from dg_wave.surface import surface_trace_kernel

        self.comm = comm
        self.local_faces = np.argwhere(mask)
        self.counts = np.asarray(comm.allgather(len(self.local_faces)), dtype=np.int64)
        self.displacements = np.concatenate(([0], np.cumsum(self.counts[:-1])))
        self._extract = surface_trace_kernel(prepared, mask, components=components)
        self.compiled = False
        rows, faces = self.local_faces.T
        keys = self._gather(np.column_stack((prepared.mesh.element_ids[rows], faces)), MPI.INT64_T)
        positions = self._gather(prepared.geometry.face_nodes[mask], MPI.DOUBLE)
        self.element_ids, self.face_indices, self.positions, self.order = (None,)*4
        error = None
        if comm.rank == 0:
            if len(np.unique(keys, axis=0)) != len(keys):
                error = "free-surface face identities must have unique owning ranks"
            self.order = np.lexsort((keys[:, 1], keys[:, 0]))
            self.element_ids, self.face_indices = keys[self.order].T
            self.positions = positions[self.order]
        collective_error(comm, error)

    def _gather(self, local, datatype):
        local = np.ascontiguousarray(local)
        stride = int(np.prod(local.shape[1:]))
        data = np.empty((int(self.counts.sum()), *local.shape[1:]), dtype=local.dtype) if self.comm.rank == 0 else None
        self.comm.Gatherv(
            [local.ravel(), datatype],
            [data, self.counts*stride, self.displacements*stride, datatype] if data is not None else None,
            root=0,
        )
        return data

    def compile(self, state):
        from mpi4py import MPI

        if self.compiled:
            return 0.0
        started = time.perf_counter()
        self._extract = self._extract.lower(state).compile()
        self.compiled = True
        return self.comm.allreduce(time.perf_counter()-started, op=MPI.MAX)

    def sample(self, state):
        from mpi4py import MPI

        local = np.asarray(self._extract(state).block_until_ready())
        data = self._gather(local, MPI.DOUBLE)
        return data[self.order] if data is not None else None


class ReceiverPlan:
    def __init__(self, partition: Partition, comm):
        from mpi4py import MPI

        self.comm = comm
        self.indices = partition.receiver_indices
        self.rows = partition.receiver_rows
        self.weights = partition.receiver_weights
        count = len(partition.receiver_element_ids)
        ownership = np.zeros(count, dtype=np.int32)
        ownership[self.indices] = 1
        total = np.empty_like(ownership)
        comm.Allreduce(ownership, total, op=MPI.SUM)
        if not np.all(total == 1):
            raise ValueError("every receiver must have exactly one owning rank")
        self.local = np.zeros((count, 9), dtype=np.float64)
        self.result = np.empty_like(self.local) if comm.rank == 0 else None

    def sample(self, state):
        from mpi4py import MPI

        self.local.fill(0.0)
        if len(self.indices):
            self.local[self.indices] = np.einsum(
                "ri,ric->rc", self.weights, np.asarray(state)[self.rows, :, :9],
            )
        self.comm.Reduce(self.local, self.result, op=MPI.SUM, root=0)
        return self.result.copy() if self.result is not None else None
