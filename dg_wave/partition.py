"""Balanced deterministic graph partitions and explicit local/remote face identities."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from dg_wave.connectivity import Connectivity
from dg_wave.config import Material
from dg_wave.materials import element_material, select_material
from dg_wave.mesh import TetraMesh
from dg_wave.operators import PreparedMesh, prepare_mesh


@dataclass(frozen=True)
class Partition:
    mesh: TetraMesh
    degree: int
    connection: Connectivity
    global_rows: np.ndarray
    global_vertex_ids: np.ndarray
    remote_ranks: np.ndarray
    receiver_indices: np.ndarray
    receiver_rows: np.ndarray
    receiver_weights: np.ndarray
    receiver_element_ids: np.ndarray
    bounds: tuple[np.ndarray, np.ndarray]
    material: Material | None = None

    def prepare(self) -> PreparedMesh:
        local = prepare_mesh(self.mesh, self.degree)
        return PreparedMesh(local.mesh, local.reference, local.geometry, self.connection)


def graph_owners(connection: Connectivity, element_ids: np.ndarray, parts: int) -> np.ndarray:
    """Split a stable-ID BFS traversal evenly; no minimum-edge-cut claim or extra dependency."""
    count = len(element_ids)
    if type(parts) is not int or parts < 1:
        raise ValueError("partition count must be a positive integer")
    if parts > count:
        raise ValueError(f"empty partitions unsupported: {count} tetrahedra for {parts} MPI ranks")
    seen = np.zeros(count, dtype=bool)
    order = []
    for seed in np.argsort(element_ids):
        if seen[seed]:
            continue
        queue = deque([int(seed)])
        seen[seed] = True
        while queue:
            element = queue.popleft()
            order.append(element)
            adjacent = sorted(
                set(connection.neighbours[element][connection.neighbours[element] >= 0]),
                key=lambda row: int(element_ids[row]),
            )
            for other in adjacent:
                if not seen[other]:
                    seen[other] = True
                    queue.append(int(other))
    owners = np.empty(count, dtype=np.int64)
    for rank, rows in enumerate(np.array_split(order, parts)):
        owners[rows] = rank
    return owners


def partition_mesh(
    prepared: PreparedMesh, connection: Connectivity, parts: int,
    receiver_rows: np.ndarray, receiver_weights: np.ndarray,
    *, material: Material | None = None,
) -> list[Partition]:
    mesh = prepared.mesh
    if material is not None:
        material = element_material(material, len(mesh.cells))
    owners = graph_owners(connection, mesh.element_ids, parts)
    receiver_ids = mesh.element_ids[receiver_rows]
    bounds = (mesh.vertices.min(axis=0), mesh.vertices.max(axis=0))
    result = []
    for rank in range(parts):
        rows = np.flatnonzero(owners == rank)
        rows = rows[np.argsort(mesh.element_ids[rows])]
        vertex_ids = np.unique(mesh.cells[rows])
        local_mesh = TetraMesh(
            mesh.vertices[vertex_ids], np.searchsorted(vertex_ids, mesh.cells[rows]),
            mesh.element_ids[rows],
            None if mesh.physical_tags is None else mesh.physical_tags[rows],
            None if mesh.boundary_tags is None else mesh.boundary_tags[rows],
        )
        local_index = np.full(len(mesh.cells), -1, dtype=np.int64)
        local_index[rows] = np.arange(len(rows))
        global_neighbours = connection.neighbours[rows]
        physical = global_neighbours < 0
        safe = np.maximum(global_neighbours, 0)
        remote = ~physical & (owners[safe] != rank)
        neighbours = np.where(physical, -1, np.where(remote, -2, local_index[safe]))
        remote_ranks = np.where(remote, owners[safe], -1)
        local_connection = Connectivity(
            neighbours, connection.neighbour_ids[rows].copy(),
            connection.neighbour_faces[rows].copy(), connection.permutations[rows].copy(),
        )
        selected = np.flatnonzero(owners[receiver_rows] == rank)
        result.append(Partition(
            local_mesh, prepared.reference.degree, local_connection, rows, vertex_ids,
            remote_ranks, selected, local_index[receiver_rows[selected]],
            receiver_weights[selected].copy(), receiver_ids.copy(), bounds,
            select_material(material, rows) if material is not None else None,
        ))
    return result
