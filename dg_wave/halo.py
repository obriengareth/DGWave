"""Preallocated NumPy trace buffers; all requests finish before buffers are reused."""

from dataclasses import dataclass

import numpy as np

from dg_wave.config import Material
from dg_wave.materials import COEFFICIENTS
from dg_wave.partition import Partition


@dataclass
class PeerBuffers:
    rank: int
    faces: np.ndarray
    permutations: np.ndarray
    slots: np.ndarray
    send: np.ndarray
    receive: np.ndarray


class HaloPlan:
    def __init__(self, partition: Partition, comm):
        self.comm = comm.Dup()
        connection = partition.connection
        nface_nodes = connection.permutations.shape[2]
        self.remote_faces = np.argwhere(partition.remote_ranks >= 0)
        self.shape = (len(self.remote_faces), nface_nodes, 9)
        self.right = np.empty(self.shape, dtype=np.float64)
        remote_slots = np.full(partition.remote_ranks.shape, -1, dtype=np.int64)
        if len(self.remote_faces):
            remote_slots[tuple(self.remote_faces.T)] = np.arange(len(self.remote_faces))
        self.local_faces = np.argwhere(connection.neighbours >= 0)
        self.connection = connection
        self.peers = []
        keys = [[] for _ in range(comm.Get_size())]
        for rank in sorted(set(partition.remote_ranks[partition.remote_ranks >= 0])):
            faces = np.argwhere(partition.remote_ranks == rank)

            def key(face):
                element, side = face
                first = (int(partition.mesh.element_ids[element]), int(side))
                second = (int(connection.neighbour_ids[element, side]),
                          int(connection.neighbour_faces[element, side]))
                return tuple(sorted((first, second)))

            faces = np.array(sorted(faces, key=key), dtype=np.int64)
            keys[int(rank)] = [key(face) for face in faces]
            count = len(faces)
            self.peers.append(PeerBuffers(
                int(rank), faces, connection.permutations[faces[:, 0], faces[:, 1]],
                remote_slots[faces[:, 0], faces[:, 1]],
                np.empty((count, nface_nodes, 9), dtype=np.float64),
                np.empty((count, nface_nodes, 9), dtype=np.float64),
            ))
        advertised = self.comm.alltoall(keys)
        error = None
        if any(advertised[rank] != keys[rank] for rank in range(comm.Get_size())):
            error = "remote face ordering/identity mismatch"
        errors = self.comm.allgather(error)
        if any(errors):
            self.comm.Free()
            raise ValueError("; ".join(sorted({item for item in errors if item})))
        self.exchanges = 0
        self.material_exchanges = 0
        self.closed = False

    def exchange_materials(self, material: Material) -> Material:
        """One static coefficient exchange; element-constant data needs no node permutation."""
        from mpi4py import MPI

        values = np.column_stack([getattr(material, name) for name in COEFFICIENTS])
        right = np.broadcast_to(values[:, None, :], (len(values), 4, 3)).copy()
        if len(self.local_faces):
            elements, faces = self.local_faces.T
            right[elements, faces] = values[self.connection.neighbours[elements, faces]]
        buffers, requests = [], []
        for peer in self.peers:
            send = np.ascontiguousarray(values[peer.faces[:, 0]])
            receive = np.empty_like(send)
            buffers.append((peer, send, receive))
            requests.append(self.comm.Irecv(receive, source=peer.rank, tag=1))
        for peer, send, _ in buffers:
            requests.append(self.comm.Isend(send, dest=peer.rank, tag=1))
        MPI.Request.Waitall(requests)
        for peer, _, receive in buffers:
            right[peer.faces[:, 0], peer.faces[:, 1]] = receive
        self.material_exchanges += 1
        return Material(*(right[..., index] for index in range(3)))

    def exchange(self, traces: np.ndarray) -> np.ndarray:
        from mpi4py import MPI

        if self.closed or traces.shape != self.shape or traces.dtype != np.float64:
            raise ValueError("halo exchange requires an open plan and correctly shaped float64 traces")
        requests = []
        # Post all receives before sends; packing finishes before the corresponding Isend.
        for peer in self.peers:
            requests.append(self.comm.Irecv(peer.receive, source=peer.rank, tag=0))
        for peer in self.peers:
            np.copyto(peer.send, traces[peer.slots])
            requests.append(self.comm.Isend(peer.send, dest=peer.rank, tag=0))
        MPI.Request.Waitall(requests)
        # Requests are complete; the caller must finish consuming right before reuse.
        for peer in self.peers:
            self.right[peer.slots] = peer.receive[
                np.arange(len(peer.faces))[:, None], peer.permutations,
            ]
        self.exchanges += 1
        return self.right

    def close(self):
        if not self.closed:
            self.comm.Free()
            self.closed = True
