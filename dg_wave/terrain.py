"""Affine geometry from sampled terrain; vertical mesh levels are not geology."""

import numpy as np

from dg_wave.geometry import build_geometry
from dg_wave.mesh import TetraMesh, box_mesh
from dg_wave.reference import FACE_VERTICES, reference_operators


def terrain_mesh(elevations, width=25000., height=6000., layers=10):
    top = np.asarray(elevations, dtype=np.float64)
    if top.ndim != 2 or min(top.shape) < 2 or not np.isfinite(top).all():
        raise ValueError("terrain elevations must be a finite two-dimensional grid")
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0 or type(layers) is not int or layers < 1:
        raise ValueError("positive finite width/height and positive integer layers required")
    bottom = float(top.max()-height)
    if np.min(top) <= bottom:
        raise ValueError("flat bottom must lie strictly below every terrain column")
    nx, ny = np.asarray(top.shape)-1
    base = box_mesh((int(nx), int(ny), layers), (width, width, 1.))
    vertices = base.vertices.copy()
    levels = vertices[:, 2].copy()
    vertices[:, :2] -= width/2
    vertices[:, 2] = bottom+levels*(np.repeat(top.ravel(), layers+1)-bottom)
    faces = base.vertices[base.cells[:, FACE_VERTICES]]
    tags = np.zeros((len(base.cells), 4), dtype=np.int64)
    for axis, length in enumerate((width, width, 1.)):
        for side in (0, 1):
            selected = np.all(np.isclose(faces[..., axis], side*length, rtol=0, atol=1e-8), axis=-1)
            tags[selected] = 10 if axis == 2 and side == 1 else 20
    mesh = TetraMesh(vertices, base.cells, np.arange(len(base.cells)),
                    np.ones(len(base.cells), dtype=np.int64), tags)
    build_geometry(mesh, reference_operators(1))
    return mesh


def surface_stations(mesh, count=10, inset=500., seed=20261120):
    if type(count) is not int or count < 1 or not np.isfinite(inset) or inset <= 0:
        raise ValueError("station count and edge inset must be positive")
    lower, upper = mesh.vertices[:, :2].min(axis=0)+inset, mesh.vertices[:, :2].max(axis=0)-inset
    if np.any(lower >= upper) or mesh.boundary_tags is None:
        raise ValueError("stations require tagged terrain and a usable inset footprint")
    rows, faces = np.nonzero(mesh.boundary_tags == 10)
    if not len(rows):
        raise ValueError("station placement requires terrain tag10")
    triangles = mesh.vertices[mesh.cells[rows[:, None], FACE_VERTICES[faces]]]
    inverse = np.linalg.inv(np.transpose(triangles[:, 1:, :2]-triangles[:, :1, :2], (0, 2, 1)))
    points = np.random.default_rng(seed).uniform(lower, upper, (count, 2))
    stations = []
    for index, xy in enumerate(points):
        uv = np.einsum("fij,fj->fi", inverse, xy-triangles[:, 0, :2])
        weights = np.column_stack((1-uv.sum(axis=1), uv))
        inside = np.flatnonzero(weights.min(axis=1) >= -1e-12)
        if not len(inside):
            raise ValueError("station does not lie on a tagged terrain triangle")
        selected = inside[np.argmin(mesh.element_ids[rows[inside]])]
        xyz = weights[selected] @ triangles[selected]
        stations.append({
            "id": f"R{index+1:02d}", "xyz_m": xyz.tolist(),
            "element_id": int(mesh.element_ids[rows[selected]]), "face": int(faces[selected]),
            "triangle_weights": weights[selected].tolist(),
        })
    return stations


def faceted_elevation(top, xy, width):
    """Evaluate the actual structured top-face diagonal, not a smoothed surface."""
    top, xy = np.asarray(top), np.asarray(xy)
    if xy.ndim != 2 or xy.shape[1] != 2 or np.any(np.abs(xy) > width/2+1e-8):
        raise ValueError("surface samples must lie within the terrain square")
    scaled = np.clip((xy/width+.5)*(np.asarray(top.shape)-1), 0, np.asarray(top.shape)-1)
    ij = np.minimum(np.floor(scaled).astype(int), np.asarray(top.shape)-2)
    u, v = (scaled-ij).T
    i, j = ij.T
    z00, z10, z01, z11 = top[i, j], top[i+1, j], top[i, j+1], top[i+1, j+1]
    return np.where(u >= v, (1-u)*z00+(u-v)*z10+v*z11,
                    (1-v)*z00+u*z11+(v-u)*z01)


def quality_report(mesh):
    geometry = build_geometry(mesh, reference_operators(1))
    points = mesh.vertices[mesh.cells]
    edges = np.stack([np.linalg.norm(points[:, j]-points[:, i], axis=1)
                      for i in range(4) for j in range(i+1, 4)], axis=1)
    mean_ratio = 12*(3*geometry.volumes)**(2/3)/np.sum(edges**2, axis=1)
    altitude = 3*geometry.volumes/np.max(2*geometry.surface_jacobians, axis=1)
    aspect = edges.max(axis=1)/altitude
    if mean_ratio.min() < .05 or aspect.max() > 20:
        raise ValueError("terrain mesh contains an excessively distorted/sliver cell")
    all_faces = np.sort(mesh.cells[:, FACE_VERTICES], axis=-1).reshape(-1, 3)
    _, inverse, counts = np.unique(all_faces, axis=0, return_inverse=True, return_counts=True)
    exterior = (counts[inverse] == 1).reshape(-1, 4)
    if np.any(counts > 2) or not np.array_equal(exterior, mesh.boundary_tags > 0):
        raise ValueError("exterior tags do not match conforming tetrahedral connectivity")
    terrain = mesh.boundary_tags == 10
    if not np.all(geometry.normals[..., 2][terrain] > 0):
        raise ValueError("terrain normals do not point upward/outward")
    return {
        "elements": len(mesh.cells), "vertices": len(mesh.vertices),
        "min_volume_m3": float(geometry.volumes.min()), "max_volume_m3": float(geometry.volumes.max()),
        "total_volume_m3": float(geometry.volumes.sum()),
        "mean_ratio_min_p01_median_max": np.quantile(mean_ratio, [0, .01, .5, 1]).tolist(),
        "max_edge_over_min_altitude": float(aspect.max()),
        "edge_min_max_m": [float(edges.min()), float(edges.max())],
        "terrain_faces": int(terrain.sum()), "outer_faces": int((mesh.boundary_tags == 20).sum()),
        "min_terrain_upward_normal_z": float(geometry.normals[..., 2][terrain].min()),
        "quality_guards": "positive volumes; conforming1/2-face incidence; complete exterior tags; mean-ratio>=.05, edge/min-altitude<=20 (distortion guards, not accuracy acceptance)",
    }
