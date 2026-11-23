"""Reusable immutable mesh data and batched physical-gradient operators."""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dg_wave.connectivity import Connectivity, build_connectivity
from dg_wave.geometry import Geometry, build_geometry
from dg_wave.mesh import TetraMesh
from dg_wave.reference import ReferenceOperators, reference_operators
from dg_wave.runtime import configure_cpu_environment

if TYPE_CHECKING:
    import jax


@lru_cache(maxsize=1)
def _jax_runtime():
    configure_cpu_environment()
    import jax

    jax.config.update("jax_enable_x64", True)
    return jax


@dataclass(frozen=True)
class PreparedMesh:
    mesh: TetraMesh
    reference: ReferenceOperators
    geometry: Geometry
    connectivity: Connectivity

    @cached_property
    def device_inverse_jacobians(self) -> "jax.Array":
        jax = _jax_runtime()
        return jax.device_put(self.geometry.inverse_jacobians, jax.devices("cpu")[0])


@lru_cache(maxsize=4, typed=True)
def prepare_mesh(mesh: TetraMesh, degree: int) -> PreparedMesh:
    """Cache by immutable mesh identity and degree, retaining at most four meshes."""
    ref = reference_operators(degree)
    geometry = build_geometry(mesh, ref)
    connectivity = build_connectivity(mesh, ref)
    return PreparedMesh(mesh, ref, geometry, connectivity)


@lru_cache(maxsize=8, typed=True)
def differentiation_kernel(degree: int):
    """JIT kernel: (element, node) scalars -> (element, node, xyz) gradients."""
    jax = _jax_runtime()
    import jax.numpy as jnp

    derivatives = jax.device_put(
        reference_operators(degree).differentiation, jax.devices("cpu")[0],
    )

    @jax.jit
    def gradient(values, inverse_jacobians):
        reference_gradient = jnp.einsum(
            "dij,ej->edi", derivatives, jnp.asarray(values, dtype=jnp.float64),
        )
        return jnp.einsum("edi,edk->eik", reference_gradient, inverse_jacobians)

    return gradient


def _check_shape(shape: tuple[int, ...], prepared: PreparedMesh) -> None:
    expected = (len(prepared.mesh.cells), len(prepared.reference.nodes))
    if shape != expected:
        raise ValueError(f"scalar nodal values must have shape {expected}, got {shape}")


def differentiate_host(values: ArrayLike, prepared: PreparedMesh) -> NDArray[np.float64]:
    data = np.asarray(values, dtype=np.float64)
    _check_shape(data.shape, prepared)
    reference_gradient = np.einsum(
        "dij,ej->edi", prepared.reference.differentiation, data, optimize=True,
    )
    return np.einsum(
        "edi,edk->eik", reference_gradient, prepared.geometry.inverse_jacobians, optimize=True,
    )


def differentiate(values: ArrayLike, prepared: PreparedMesh) -> "jax.Array":
    _check_shape(np.shape(values), prepared)
    return differentiation_kernel(prepared.reference.degree)(
        values, prepared.device_inverse_jacobians,
    )
