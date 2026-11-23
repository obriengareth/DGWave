"""Agreed effective Rainier modelling inputs, not measured stratigraphy."""

import numpy as np

from dg_wave.config import Material
from dg_wave.materials import element_material
from dg_wave.mesh import TetraMesh
from dg_wave.reference import FACE_VERTICES
from dg_wave.terrain import faceted_elevation, terrain_mesh


INTERFACES_M = (500., 875., 1250., 1625., 2000.)
INTERVALS_TOP_DOWN = (3, 1, 1, 1, 1, 3)


def effective_materials():
    inputs = (
        ("Weathered/fractured effective rock", 2200., 2500., 1400., 60., 40., "#d6ac57"),
        ("Intermediate effective rock I", 2300., 3000., 1700., 75., 50., "#a6bf64"),
        ("Intermediate effective rock II", 2400., 3500., 2000., 90., 60., "#63b69e"),
        ("Intermediate effective rock III", 2500., 4000., 2300., 100., 70., "#4b99b6"),
        ("Intermediate effective rock IV", 2600., 4500., 2600., 125., 85., "#657eb8"),
        ("Basement effective approximation", 2700., 5000., 2900., 150., 100., "#9469ac"),
    )
    result = []
    bounds = (0., *INTERFACES_M, None)
    for tag, (name, rho, vp, vs, qp, qs, color) in enumerate(inputs, 1):
        k, mu = rho*(vp*vp-4*vs*vs/3), rho*vs*vs
        element_material(Material(rho, k, mu), 1)
        inverse_bulk_q = ((k+4*mu/3)/qp-4*mu/(3*qs))/k
        if not np.isfinite(inverse_bulk_q) or inverse_bulk_q <= 0:
            raise ValueError("material Qp/Qs require nonpositive bulk dissipation")
        result.append({
            "tag": tag, "name": name, "depth_top_m": bounds[tag-1], "depth_bottom_m": bounds[tag],
            "rho_kg_m3": rho, "vp_m_s": vp, "vs_m_s": vs, "qp": qp, "qs": qs,
            "bulk_modulus_Pa": k, "shear_modulus_Pa": mu, "lame_lambda_Pa": k-2*mu/3,
            "poisson_ratio": (vp*vp-2*vs*vs)/(2*(vp*vp-vs*vs)),
            "necessary_inverse_bulk_q": inverse_bulk_q, "implied_bulk_q": 1/inverse_bulk_q,
            "color": color,
        })
    return result


def material_inputs():
    table = effective_materials()
    return {
        "materials": {
            str(row["tag"]): {"rho": row["rho_kg_m3"], "vp": row["vp_m_s"], "vs": row["vs_m_s"]}
            for row in table
        },
        "attenuation": {
            "materials": {str(row["tag"]): {"qp": row["qp"], "qs": row["qs"]} for row in table},
        },
    }


def layered_terrain_mesh(top, width=25000., height=6000.):
    base = terrain_mesh(top, width, height, sum(INTERVALS_TOP_DOWN))
    top = np.asarray(top, dtype=np.float64)
    bottom = float(base.vertices[:, 2].min())
    if top.min()-INTERFACES_M[-1] <= bottom:
        raise ValueError("deepest material interface must remain strictly above the flat base")
    bounds = [np.full_like(top, bottom), *(top-depth for depth in reversed(INTERFACES_M)), top]
    levels = [bounds[0]]
    for lower, upper, count in zip(bounds[:-1], bounds[1:], reversed(INTERVALS_TOP_DOWN), strict=True):
        levels.extend(lower+(upper-lower)*step/count for step in range(1, count+1))
    vertices = base.vertices.copy()
    vertices[:, 2] = np.stack(levels, axis=-1).ravel()
    vertices[::11, 2] = base.vertices[::11, 2]
    vertices[10::11, 2] = base.vertices[10::11, 2]
    interval_tags = np.repeat(np.arange(6, 0, -1), INTERVALS_TOP_DOWN[::-1])
    tags = np.tile(np.repeat(interval_tags, 6), int(np.prod(np.asarray(top.shape)-1)))
    return TetraMesh(vertices, base.cells, base.element_ids, tags, base.boundary_tags)


def interface_report(mesh, top, width):
    depth = faceted_elevation(top, mesh.vertices[:, :2], width)-mesh.vertices[:, 2]
    per_cell = depth[mesh.cells]
    materials = effective_materials()
    for material in materials:
        selected = per_cell[mesh.physical_tags == material["tag"]]
        if not len(selected) or selected.min() < material["depth_top_m"]-1e-8:
            raise ValueError("material cell crosses its upper depth interface")
        if material["depth_bottom_m"] is not None and selected.max() > material["depth_bottom_m"]+1e-8:
            raise ValueError("material cell crosses its lower depth interface")
    faces = mesh.cells[:, FACE_VERTICES].reshape(-1, 3)
    _, first, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_index=True, return_inverse=True, return_counts=True,
    )
    if np.any(counts > 2):
        raise ValueError("non-manifold layered mesh")
    owner_tags = np.repeat(mesh.physical_tags, 4)
    first_tags = owner_tags[first]
    sums = np.bincount(inverse, weights=owner_tags).astype(np.int64)
    is_interface = (counts == 2) & (sums != 2*first_tags)
    triangles = faces[first[is_interface]]
    pairs = np.sort(np.column_stack((first_tags[is_interface], sums[is_interface]-first_tags[is_interface])), axis=1)
    reports = []
    for tag, expected_depth in enumerate(INTERFACES_M, 1):
        selected = (pairs == [tag, tag+1]).all(axis=1)
        error = float(np.max(np.abs(depth[triangles[selected]]-expected_depth))) if selected.any() else float("inf")
        expected_faces = 2*int(np.prod(np.asarray(top.shape)-1))
        if selected.sum() != expected_faces or error > 1e-8:
            raise ValueError("material interface is not an exact conforming terrain-offset surface")
        reports.append({"tags": [tag, tag+1], "depth_below_faceted_terrain_m": expected_depth,
                        "triangles": int(selected.sum()), "max_depth_error_m": error})
    if len(triangles) != sum(item["triangles"] for item in reports):
        raise ValueError("unexpected material adjacency, not the proposed six-layer stack")
    return reports, triangles, pairs
