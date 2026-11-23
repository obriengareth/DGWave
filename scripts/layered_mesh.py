"""Generate the conforming two-layer Gmsh 2.2 fixture using existing meshio."""

from pathlib import Path

import meshio
import numpy as np

from dg_wave.mesh import box_mesh


if __name__ == "__main__":
    mesh = box_mesh(4, (1000.0, 1000.0, 1000.0))
    tags = np.where(mesh.vertices[mesh.cells, 2].mean(axis=1) < 500.0, 1, 2)
    meshio.write_points_cells(
        Path(__file__).parents[1] / "layered_box.msh", mesh.vertices,
        [("tetra", mesh.cells)], file_format="gmsh22", binary=False,
        cell_data={"gmsh:physical": [tags], "gmsh:geometrical": [tags]},
        field_data={"lower": np.array([1, 3]), "upper": np.array([2, 3])},
    )
