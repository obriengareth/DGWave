"""Host-side elementwise elastic coefficients and artificial sponge profiles."""

import numpy as np

from dg_wave.config import Config, Material, Sponge


COEFFICIENTS = ("density", "bulk_modulus", "shear_modulus")


def element_material(material: Material, count: int) -> Material:
    values = []
    for name in COEFFICIENTS:
        value = np.asarray(getattr(material, name), dtype=np.float64)
        if value.shape not in ((), (count,)) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError(f"{name} must be positive finite scalars or one value per element")
        coefficient = np.broadcast_to(value, (count,)).copy()
        coefficient.setflags(write=False)
        values.append(coefficient)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        cp2 = values[1] / values[0] + (4 / 3) * values[2] / values[0]
        cs2 = values[2] / values[0]
    if not np.isfinite(cp2).all() or not np.isfinite(cs2).all() or np.any(cs2 <= 0):
        raise ValueError("material wave speeds must be positive finite")
    return Material(*values)


def select_material(material: Material, indices) -> Material:
    return Material(*(getattr(material, name)[indices] for name in COEFFICIENTS))


def shaped_material(material: Material, shape) -> Material:
    return Material(*(np.reshape(getattr(material, name), shape) for name in COEFFICIENTS))


def assign_materials(mesh, config: Config) -> Material:
    if config.material is not None:
        if mesh.physical_tags is not None:
            raise ValueError("tagged meshes require a materials table; physical tags cannot be ignored")
        return element_material(config.material, len(mesh.cells))
    if mesh.physical_tags is None:
        raise ValueError("materials table requires Gmsh physical tags on every tetrahedron")
    configured = dict(config.materials)
    used = set(map(int, mesh.physical_tags))
    if used != set(configured):
        raise ValueError(
            f"material tags mismatch: missing materials {sorted(used - set(configured))}; "
            f"unused materials {sorted(set(configured) - used)}"
        )
    return element_material(Material(*(
        np.array([getattr(configured[int(tag)], name) for tag in mesh.physical_tags])
        for name in COEFFICIENTS
    )), len(mesh.cells))


def sponge_rates(prepared, sponge: Sponge | None, bounds) -> np.ndarray:
    """Element-constant quadratic damping: E'_sponge = -2 sum_e sigma_e E_e."""
    rate = np.zeros(len(prepared.mesh.cells))
    if sponge is None:
        return rate
    if not np.isfinite(sponge.rate) or sponge.rate <= 0 or not np.isfinite(sponge.width):
        raise ValueError("sponge rate and width must be positive finite")
    lower, upper = bounds
    centers = prepared.mesh.vertices[prepared.mesh.cells].mean(axis=1)
    for face in sponge.faces:
        axis = "xyz".index(face[0])
        if not 0 < sponge.width < (upper[axis] - lower[axis]) / 2:
            raise ValueError("sponge.width must be smaller than half each selected domain extent")
        distance = centers[:, axis] - lower[axis] if face.endswith("min") else upper[axis] - centers[:, axis]
        rate += sponge.rate * np.maximum(0, 1 - distance / sponge.width)**2
    return rate
