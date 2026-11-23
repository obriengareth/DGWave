"""Nested affine refinement and measured DG free-surface sampling."""

import modepy as mp
import numpy as np

from dg_wave.mesh import TetraMesh
from dg_wave.reference import FACE_VERTICES
from dg_wave.solver import free_boundary_faces


def refine_tetrahedra(mesh):
    """Eight children per cell; common edge midpoints preserve all planar facets."""
    if np.max(mesh.element_ids) > (np.iinfo(np.int64).max-7)//8:
        raise ValueError("refined stable element IDs exceed int64")
    vertices = mesh.vertices.tolist()
    midpoints, cells, face_tags = {}, [], []
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    template = np.array([
        [0, 4, 5, 6], [1, 4, 7, 8], [2, 5, 7, 9], [3, 6, 8, 9],
        [4, 9, 5, 6], [4, 9, 6, 8], [4, 9, 8, 7], [4, 9, 7, 5],
    ])
    bary = np.concatenate((np.eye(4), np.array([(np.eye(4)[i]+np.eye(4)[j])/2 for i, j in pairs])))
    for row, parent in enumerate(mesh.cells):
        nodes = list(parent)
        for i, j in pairs:
            key = tuple(sorted((int(parent[i]), int(parent[j]))))
            if key not in midpoints:
                midpoints[key] = len(vertices)
                vertices.append(((mesh.vertices[key[0]]+mesh.vertices[key[1]])/2).tolist())
            nodes.append(midpoints[key])
        for child in template:
            child = child.copy()
            xyz = np.asarray([vertices[nodes[i]] for i in child])
            if np.linalg.det((xyz[1:]-xyz[0]).T) < 0:
                child[[0, 1]] = child[[1, 0]]
            cells.append(np.asarray(nodes)[child])
            tags = np.zeros(4, dtype=np.int64)
            if mesh.boundary_tags is not None:
                for face, face_nodes in enumerate(FACE_VERTICES):
                    opposite = np.flatnonzero(np.all(bary[child[face_nodes]] == 0, axis=0))
                    if len(opposite):
                        tags[face] = mesh.boundary_tags[row, opposite[0]]
            face_tags.append(tags)
    parents = np.repeat(np.arange(len(mesh.cells)), 8)
    return TetraMesh(
        vertices, np.asarray(cells), np.repeat(mesh.element_ids*8, 8)+np.tile(np.arange(8), len(mesh.cells)),
        None if mesh.physical_tags is None else mesh.physical_tags[parents],
        None if mesh.boundary_tags is None else np.asarray(face_tags),
    ), parents


def free_surface_mask(prepared, tags):
    bounds = (prepared.mesh.vertices.min(axis=0), prepared.mesh.vertices.max(axis=0))
    return free_boundary_faces(prepared, prepared.connectivity, bounds, tags, required=True)


def surface_quadrature(prepared, mask):
    q = mp.quadrature_for_space(mp.PN(2, 6), mp.Simplex(2))
    uv = (q.nodes.T+1)/2
    bary = np.column_stack((1-uv.sum(axis=1), uv))
    xyz = prepared.mesh.vertices[prepared.mesh.cells[:, FACE_VERTICES]][mask]
    points = np.einsum("ni,fij->fnj", bary, xyz)
    jacobian = np.linalg.norm(np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), axis=-1)/4
    weights = jacobian[:, None]*q.weights
    if np.any(weights <= 0):
        raise ValueError("surface norm requires positive quadrature weights")
    return points.reshape(-1, 3), weights.ravel()


def surface_trace(prepared, state, mask):
    if np.shape(state)[:2] != prepared.geometry.nodes.shape[:2]:
        raise ValueError("surface extraction requires the full element/nodal DG state")
    rows, faces = np.nonzero(mask)
    values = np.einsum("fij,fjc->fic", prepared.reference.trace[faces], state[rows])
    return prepared.geometry.face_nodes[mask].copy(), values


def surface_trace_kernel(prepared, mask, *, components=slice(None)):
    from dg_wave.operators import _jax_runtime

    jax = _jax_runtime()
    import jax.numpy as jnp

    rows, faces = np.nonzero(mask)
    trace = jnp.asarray(prepared.reference.trace[faces])
    rows = jnp.asarray(rows)

    @jax.jit
    def extract(state):
        return jnp.einsum("fij,fjc->fic", trace, state[rows, :, components])

    return extract


def display_triangles(reference, count):
    if reference.degree != 2:
        raise ValueError("this initial surface view requires degree 2")
    target = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    indices = np.argmin(np.linalg.norm(target[:, None]-reference.face_barycentric, axis=-1), axis=1)
    if np.max(np.abs(reference.face_barycentric[indices]-target)) > 1e-12:
        raise ValueError("unexpected quadratic face nodes")
    local = indices[np.array([[0, 3, 4], [1, 5, 3], [2, 4, 5], [3, 5, 4]])]
    return (np.arange(count)[:, None, None]*6+local).reshape(-1, 3)


def snapshot_steps(times, dt, total):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("snapshot timestep must be positive finite")
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("snapshot times must be finite, increasing and distinct")
    steps = np.rint(times/dt).astype(np.int64)
    if np.any(steps < 0) or np.any(steps > total) or np.any(np.abs(steps*dt-times) > 64*np.finfo(float).eps*np.maximum(1, np.abs(times))):
        raise ValueError("snapshot time must lie on a timestep within the simulation")
    return steps


def field_comparison(actual, reference, weights, times):
    actual, reference, weights, times = map(np.asarray, (actual, reference, weights, times))
    if actual.shape != reference.shape or actual.shape != (len(times), len(weights), 3) or np.any(weights <= 0):
        raise ValueError("wavefield comparison requires matched time/point/velocity arrays and positive area weights")
    difference = np.sum(weights[None, :, None]*(actual-reference)**2, axis=(1, 2))
    norm = np.sum(weights[None, :, None]*reference**2, axis=(1, 2))
    if np.any((norm == 0) & (difference != 0)):
        raise ValueError("relative wavefield norm has a zero reference with nonzero difference")
    relative = np.sqrt(np.divide(difference, norm, out=np.zeros_like(norm), where=norm > 0))
    total_norm = float(np.trapezoid(norm, times))
    if total_norm <= 0:
        raise ValueError("space-time reference wavefield norm must be positive")
    return {
        "times_s": times.tolist(), "per_time_relative_l2": relative.tolist(),
        "reference_rms_velocity_m_s": np.sqrt(norm/weights.sum()).tolist(),
        "difference_rms_velocity_m_s": np.sqrt(difference/weights.sum()).tolist(),
        "spacetime_relative_l2": float(np.sqrt(np.trapezoid(difference, times)/total_norm)),
        "spacetime_difference_norm_squared": float(np.trapezoid(difference, times)),
        "method": "area-weighted all-three-component surface velocity L2 at identical quadrature points; trapezoidal integration over saved times; denominator is reference field, not an exact solution",
    }
