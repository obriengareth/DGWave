"""Parameterised synthetic terrain: affine tetrahedra, not curved elements."""

import argparse
from pathlib import Path

import meshio
import numpy as np

from dg_wave.geometry import build_geometry
from dg_wave.mesh import TetraMesh, box_mesh
from dg_wave.reference import FACE_VERTICES, reference_operators


def volcano_mesh(divisions=(4, 4, 4), lengths=(1600., 1600., 1000.),
                 height=300., radius=450., interface_layer=2):
    base = box_mesh(divisions, lengths)
    if (
        not np.isfinite([height, radius]).all() or height < 0 or radius <= 0
        or type(interface_layer) is not int or not 0 < interface_layer < divisions[2]
    ):
        raise ValueError("terrain requires nonnegative height, positive radius and an interior interface layer")
    points = base.vertices.copy()
    radial = np.sum((points[:, :2]-np.asarray(lengths[:2])/2)**2, axis=1)
    points[:, 2] *= 1 + height/lengths[2]*np.exp(-radial/radius**2)
    tags = np.where(
        base.vertices[base.cells, 2].mean(axis=1) < lengths[2]*interface_layer/divisions[2],
        1, 2,
    )
    face_tags = np.zeros((len(base.cells), 4), dtype=np.int64)
    faces = base.vertices[base.cells[:, FACE_VERTICES]]
    for axis in range(3):
        for side in range(2):
            mask = np.all(np.isclose(faces[..., axis], side*lengths[axis], rtol=0, atol=1e-10), axis=-1)
            face_tags[mask] = 10 if axis == 2 and side == 1 else 20
    mesh = TetraMesh(points, base.cells, base.element_ids, tags, face_tags)
    build_geometry(mesh, reference_operators(1))
    return mesh


def write_volcano(path, **parameters):
    mesh = volcano_mesh(**parameters)
    elements, faces = np.nonzero(mesh.boundary_tags)
    triangles = mesh.cells[elements[:, None], FACE_VERTICES[faces]]
    boundary_tags = mesh.boundary_tags[elements, faces]
    meshio.write_points_cells(
        path, mesh.vertices, [("tetra", mesh.cells), ("triangle", triangles)],
        file_format="gmsh22", binary=False,
        cell_data={
            "gmsh:physical": [mesh.physical_tags, boundary_tags],
            "gmsh:geometrical": [mesh.physical_tags, boundary_tags],
        },
        field_data={"basement": [1, 3], "edifice": [2, 3], "terrain": [10, 2], "outer": [20, 2]},
    )
    return mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--divisions", type=int, nargs=3, default=(4, 4, 4))
    parser.add_argument("--lengths", type=float, nargs=3, default=(1600., 1600., 1000.))
    parser.add_argument("--height", type=float, default=300.)
    parser.add_argument("--radius", type=float, default=450.)
    parser.add_argument("--interface-layer", type=int, default=2)
    args = vars(parser.parse_args())
    output = args.pop("output")
    write_volcano(output, **args)


if __name__ == "__main__":
    main()
