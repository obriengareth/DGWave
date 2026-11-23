"""Shared matrix-free elastic DG kernels and the fast serial RK4 schedule."""

import math

import numpy as np

from dg_wave.config import Material
from dg_wave.attenuation import fit_relaxation
from dg_wave.connectivity import Connectivity, pair_periodic_box
from dg_wave.materials import element_material, select_material, shaped_material, sponge_rates
from dg_wave.operators import PreparedMesh, _jax_runtime
from dg_wave.physics import constitutive_rate, interface_states, surface_correction, volume_rhs, wave_speeds
from dg_wave.reference import FACE_VERTICES

jax = _jax_runtime()
import jax.numpy as jnp


def conservative_timestep(
    prepared: PreparedMesh, material: Material, cfl: float = 0.5, damping=None,
) -> float:
    """Safety estimate r_in / (cp*(p+1)^2); cfl <= 0.5, validated by energy/wave tests."""
    if not math.isfinite(cfl) or not 0 < cfl <= 0.5:
        raise ValueError("cfl must lie in (0, 0.5]")
    geometry = prepared.geometry
    inradius = 3 * geometry.volumes / np.sum(2 * geometry.surface_jacobians, axis=1)
    local = cfl * inradius / (wave_speeds(material)[0] * (prepared.reference.degree + 1)**2)
    if damping is not None:
        local = local / (1 + local * damping)
    return float(np.min(local))


def top_free_faces(prepared, connection, bounds, *, required=False):
    vertices = prepared.mesh.vertices[prepared.mesh.cells[:, FACE_VERTICES]]
    tolerance = 1e-12 * max(1.0, float(np.max(bounds[1] - bounds[0])))
    mask = (connection.neighbours == -1) & np.all(
        np.abs(vertices[..., 2] - bounds[1][2]) <= tolerance, axis=-1,
    )
    if required and not np.any(mask):
        raise ValueError("top free surface requires planar z-max physical boundary faces")
    return mask


def free_boundary_faces(prepared, connection, bounds, tags=(), *, required=False):
    if not tags:
        return top_free_faces(prepared, connection, bounds, required=required)
    available = prepared.mesh.boundary_tags
    if available is None:
        raise ValueError("tag-selected free surface requires physical boundary triangle tags")
    if required and not set(tags) <= set(np.unique(available)):
        raise ValueError("requested free-surface tags are missing from the mesh")
    mask = np.isin(available, tags)
    if np.any(mask & (connection.neighbours != -1)):
        raise ValueError("free-surface tags must select physical exterior faces")
    return mask


def rk4(rhs, state, time, dt):
    k1 = rhs(state, time)
    k2 = rhs(state + dt * k1 / 2, time + dt / 2)
    k3 = rhs(state + dt * k2 / 2, time + dt / 2)
    k4 = rhs(state + dt * k3, time + dt)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def elastic_connectivity(prepared: PreparedMesh, boundaries: tuple[str, str, str]) -> Connectivity:
    if len(boundaries) != 3 or any(kind not in {"periodic", "absorbing"} for kind in boundaries):
        raise ValueError("boundaries must be three 'periodic' or 'absorbing' axis types")
    axes = tuple(axis for axis, kind in enumerate(boundaries) if kind == "periodic")
    return (
        pair_periodic_box(
            prepared.mesh, prepared.reference, prepared.geometry, prepared.connectivity, axes,
        ) if axes else prepared.connectivity
    )


class ElasticSolver:
    def __init__(
        self, prepared: PreparedMesh, material: Material,
        boundaries: tuple[str, str, str] = ("periodic", "periodic", "periodic"), source=None,
        *, connection: Connectivity | None = None,
        neighbour_material: Material | None = None, free_surface=False,
        sponge=None, domain_bounds=None,
        attenuation=None, relaxation=None, free_surface_tags=(),
    ):
        self.prepared = prepared
        self.reference_material = element_material(material, len(prepared.mesh.cells))
        if attenuation is not None and relaxation is not None:
            raise ValueError("supply attenuation settings or an already fitted model, not both")
        self.relaxation = relaxation if relaxation is not None else fit_relaxation(
            self.reference_material, attenuation, physical_tags=prepared.mesh.physical_tags,
        )
        self.material = self.relaxation.unrelaxed if self.relaxation is not None else self.reference_material
        model = self.relaxation
        self.boundaries = boundaries
        self.connection = elastic_connectivity(prepared, boundaries) if connection is None else connection
        bounds = domain_bounds if domain_bounds is not None else (
            prepared.mesh.vertices.min(axis=0), prepared.mesh.vertices.max(axis=0),
        )
        if free_surface and boundaries[2] == "periodic":
            raise ValueError("top free surface cannot be paired periodically in z")
        if sponge is not None:
            for face in sponge.faces:
                if boundaries["xyz".index(face[0])] == "periodic" or (free_surface and face == "zmax"):
                    raise ValueError("sponge cannot cover periodic faces or the physical free surface")
        self.damping = sponge_rates(prepared, sponge, bounds)
        self.elastic_timestep = conservative_timestep(prepared, self.reference_material, damping=self.damping)
        self.max_timestep = conservative_timestep(
            prepared, self.material, damping=self.damping+(model.rate_limit if model is not None else 0),
        )
        volume_material = shaped_material(self.material, (-1, 1))
        minus_material = shaped_material(self.material, (-1, 1, 1))
        if neighbour_material is None:
            if np.any(self.connection.neighbours == -2):
                raise ValueError("remote face material coefficients must be exchanged before compilation")
            rows = np.where(
                self.connection.neighbours >= 0, self.connection.neighbours,
                np.arange(len(prepared.mesh.cells))[:, None],
            )
            neighbour_material = select_material(self.material, rows)
        self.neighbour_material = neighbour_material
        plus_material = shaped_material(neighbour_material, (len(prepared.mesh.cells), 4, 1))
        self.free_faces = (
            free_boundary_faces(
                prepared, self.connection, bounds, free_surface_tags,
                required=domain_bounds is None,
            )
            if free_surface else np.zeros(self.connection.neighbours.shape, dtype=bool)
        )
        free_mask = jnp.asarray(self.free_faces)
        damping = jnp.asarray(self.damping[:, None, None])
        ref, geometry = prepared.reference, prepared.geometry
        derivatives = jnp.asarray(ref.differentiation)
        inverse = prepared.device_inverse_jacobians
        trace = jnp.asarray(ref.trace)
        lift = jnp.asarray(ref.lift)
        normals = jnp.asarray(geometry.normals[:, :, None, :])
        factors = jnp.asarray(geometry.surface_factors[:, :, None, None])
        nelements, nnodes = geometry.nodes.shape[:2]
        nface_nodes = len(ref.face_barycentric)
        mechanisms = model.mechanisms if model is not None else 0
        if model is not None:
            tau = jnp.asarray(model.tau[None, None, :, None])
            branch_bulk = jnp.asarray(model.bulk[:, None, :])
            branch_shear = jnp.asarray(model.shear[:, None, :])
        connection = self.connection
        boundary = connection.neighbours < 0
        face_indices = (
            (np.maximum(connection.neighbours, 0) * 4 + np.maximum(connection.neighbour_faces, 0))
            [..., None] * nface_nodes + connection.permutations
        )
        face_indices = jnp.asarray(face_indices)
        boundary = jnp.asarray(boundary[:, :, None, None])

        def traces(state):
            return jnp.einsum("fij,ejc->efic", trace, state[..., :9])

        def residual(state, time, left, right):
            reference_gradient = jnp.einsum("dij,ejc->edic", derivatives, state[..., :9])
            gradient = jnp.einsum("edic,edk->eick", reference_gradient, inverse)
            volume = volume_rhs(gradient, volume_material)
            correction = surface_correction(
                left, right, normals, minus_material, plus_material, free_mask,
            ) * factors
            result = volume + jnp.einsum(
                "ij,ejc->eic", lift, correction.reshape(nelements, 4 * nface_nodes, 9),
            )
            if source is not None:
                forcing = source(time)
                if forcing.shape[-1] == 3:
                    result = result.at[..., :3].add(forcing)
                elif forcing.shape[-1] == 9:
                    result = result+forcing
                else:
                    raise ValueError("source must supply three accelerations or nine physical state rates")
            if model is not None:
                memory = state[..., 9:].reshape(nelements, nnodes, mechanisms, 6)
                velocity, _ = interface_states(
                    left, right, normals, minus_material, plus_material, free_mask,
                )
                jump = velocity-left[..., :3]
                face_gradient = jump[..., :, None]*normals[..., None, :]*factors[..., None]
                corrected = gradient[..., :3, :] + jnp.einsum(
                    "ij,ejab->eiab", lift,
                    face_gradient.reshape(nelements, 4*nface_nodes, 3, 3),
                )
                memory_rate = constitutive_rate(
                    corrected[:, :, None], branch_bulk, branch_shear,
                ) - memory/tau
                result = result.at[..., 3:9].add(-jnp.sum(memory/tau, axis=2))
                result = jnp.concatenate((result, memory_rate.reshape(nelements, nnodes, -1)), axis=-1)
            return result - damping * state

        has_remote = bool(np.any(connection.neighbours == -2))
        remote_rows, remote_faces = np.nonzero(connection.neighbours == -2)
        remote_trace = trace[remote_faces]
        remote_rows, remote_faces = jnp.asarray(remote_rows), jnp.asarray(remote_faces)

        def mapped_right(left):
            right = left.reshape(-1, 9)[face_indices]
            return jnp.where(boundary, 0.0, right)

        def remote_traces(state):
            return jnp.einsum("fij,fjc->fic", remote_trace, state[remote_rows, :, :9])

        def distributed_rhs(state, time, remote_right):
            left = traces(state)
            right = mapped_right(left).at[remote_rows, remote_faces].set(remote_right)
            return residual(state, time, left, right)

        def rhs(state, time):
            if has_remote:
                raise ValueError("partitioned residual requires externally exchanged right traces")
            left = traces(state)
            # Zero exterior state gives t*=-Z(n)v*: no incoming characteristics.
            return residual(state, time, left, mapped_right(left))

        self.trace_kernel = jax.jit(traces)
        self.remote_trace_kernel = jax.jit(remote_traces)
        self.distributed_residual_kernel = jax.jit(distributed_rhs)
        self.residual_kernel = jax.jit(residual)
        self.rhs = jax.jit(rhs)
        self.step = jax.jit(lambda state, time, dt: rk4(rhs, state, time, dt))

        def advance(state, time, dt, *, steps):
            def body(carry, _):
                current, t = carry
                return (rk4(rhs, current, t, dt), t + dt), None
            return jax.lax.scan(body, (state, time), None, length=steps)[0][0]

        self.advance_kernel = jax.jit(advance, static_argnames=("steps",))
        mass = jnp.asarray(ref.mass)
        determinants = jnp.asarray(geometry.determinants)
        weights = jnp.asarray([1, 1, 1, 2, 2, 2])
        rho = jnp.asarray(volume_material.density)

        def compliance_inner(left, right, bulk, mu):
            stress = right * weights * mu[..., None] / 2
            trace_stress = jnp.sum(right[..., :3], axis=-1)
            stress = stress.at[..., :3].add(
                trace_stress[..., None] * (bulk/9-mu/6)[..., None]
            )
            return jnp.einsum("eic,ij,ejc,e->", left, mass, stress, determinants)

        if model is None:
            inverse_bulk = 1/jnp.asarray(volume_material.bulk_modulus)
            inverse_mu = 1/jnp.asarray(volume_material.shear_modulus)
        else:
            inverse_bulk = 1/jnp.asarray(model.relaxed.bulk_modulus[:, None])
            inverse_mu = 1/jnp.asarray(model.relaxed.shear_modulus[:, None])
            inverse_branch_bulk = jnp.asarray(np.divide(
                1, model.bulk, out=np.zeros_like(model.bulk), where=model.bulk > 0,
            ))
            inverse_branch_mu = jnp.asarray(np.divide(
                1, model.shear, out=np.zeros_like(model.shear), where=model.shear > 0,
            ))

        def inner(left, right):
            kinetic = jnp.einsum(
                "eic,ij,ejc,e->", left[..., :3], mass, rho[..., None]*right[..., :3], determinants,
            )
            ls, rs = left[..., 3:9], right[..., 3:9]
            if model is not None:
                lm = left[..., 9:].reshape(nelements, nnodes, mechanisms, 6)
                rm = right[..., 9:].reshape(nelements, nnodes, mechanisms, 6)
                ls, rs = ls-jnp.sum(lm, axis=2), rs-jnp.sum(rm, axis=2)
                for branch in range(mechanisms):
                    kinetic += compliance_inner(
                        lm[:, :, branch], rm[:, :, branch],
                        inverse_branch_bulk[:, branch, None], inverse_branch_mu[:, branch, None],
                    )
            return kinetic + compliance_inner(ls, rs, inverse_bulk, inverse_mu)

        self.inner_product = jax.jit(inner)
        self.energy = jax.jit(lambda state: inner(state, state) / 2)
        self.state_shape = (nelements, nnodes, 9+6*mechanisms)

    def validate_timestep(self, dt: float) -> None:
        if not math.isfinite(dt) or dt <= 0 or dt > self.max_timestep * (1 + 1e-14):
            raise ValueError(f"dt must be positive and <= conservative limit {self.max_timestep:.8g}")

    def advance(self, state, dt: float, steps: int, time: float = 0.0):
        self.validate_timestep(dt)
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        if np.shape(state) != self.state_shape or not math.isfinite(time):
            raise ValueError(f"state must have shape {self.state_shape} and time must be finite")
        return self.advance_kernel(jnp.asarray(state, dtype=jnp.float64), time, dt, steps=steps)
