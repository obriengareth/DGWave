import pickle

import numpy as np
import pytest

from dg_wave.config import Material
from dg_wave.mesh import TetraMesh, box_mesh
from dg_wave.operators import prepare_mesh
from dg_wave.partition import graph_owners, partition_mesh
from dg_wave.run import receiver_interpolation
from dg_wave.solver import ElasticSolver


@pytest.mark.parametrize("parts", [1, 2, 4])
def test_graph_partition_identity_and_connectivity(parts):
    prepared = prepare_mesh(box_mesh(2), 2)
    connection = ElasticSolver(prepared, Material(1, 8 / 3, 1)).connection
    positions = [[0.5, 0.5, 0.5], [0.17, 0.28, 0.36]]
    rows, weights = receiver_interpolation(prepared, positions)
    partitions = partition_mesh(prepared, connection, parts, rows, weights, material=Material(1, 8 / 3, 1))
    owners = graph_owners(connection, prepared.mesh.element_ids, parts)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([part.mesh.element_ids for part in partitions])),
        np.sort(prepared.mesh.element_ids),
    )
    assert max(map(lambda p: len(p.mesh.cells), partitions)) - min(map(lambda p: len(p.mesh.cells), partitions)) <= 1
    assert sorted(np.concatenate([part.receiver_indices for part in partitions])) == [0, 1]
    for rank, part in enumerate(partitions):
        part = pickle.loads(pickle.dumps(part))
        assert not any(
            array.flags.writeable for array in (part.mesh.vertices, part.mesh.cells, part.mesh.element_ids)
        )
        local = part.prepare()
        np.testing.assert_allclose(local.geometry.nodes, prepared.geometry.nodes[part.global_rows], atol=1e-13)
        for element, face in np.ndindex(part.remote_ranks.shape):
            other = connection.neighbours[part.global_rows[element], face]
            if owners[other] == rank:
                assert part.connection.neighbours[element, face] >= 0
            else:
                assert part.connection.neighbours[element, face] == -2
                assert part.remote_ranks[element, face] == owners[other]
            assert part.connection.neighbour_ids[element, face] == prepared.mesh.element_ids[other]


def test_partition_is_stable_under_cell_reordering_and_rejects_empty():
    box = box_mesh()
    material = Material(1, 8 / 3, 1)
    original = prepare_mesh(box, 1)
    owners = graph_owners(ElasticSolver(original, material).connection, box.element_ids, 4)
    order = np.array([3, 0, 5, 1, 4, 2])
    reordered = prepare_mesh(TetraMesh(box.vertices, box.cells[order], box.element_ids[order]), 1)
    other = graph_owners(ElasticSolver(reordered, material).connection, reordered.mesh.element_ids, 4)
    np.testing.assert_array_equal(other, owners[order])
    with pytest.raises(ValueError, match="empty partitions unsupported"):
        graph_owners(original.connectivity, box.element_ids, 7)
