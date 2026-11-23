"""Actual MPI integration worker; launched only by timeout-protected subprocess tests."""

import json
import os
from pathlib import Path
import sys
import traceback

from dg_wave.runtime import initialize_mpi_cpu

comm, _, _ = initialize_mpi_cpu()

from mpi4py import MPI
import numpy as np

from dg_wave.config import Attenuation, Config, Material, Source, Sponge
from dg_wave.attenuation import fit_relaxation
from dg_wave.distributed import DistributedSolver, ReceiverPlan, SurfacePlan
from dg_wave.halo import HaloPlan
from dg_wave.mesh import TetraMesh, box_mesh
from dg_wave.materials import COEFFICIENTS, element_material
from dg_wave.operators import prepare_mesh
from dg_wave.partition import partition_mesh
from dg_wave.run import receiver_interpolation, smooth_source, validate_source
from dg_wave.solver import ElasticSolver, elastic_connectivity
from dg_wave.waves import plane_wave


def disconnected_mesh():
    reference = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vertices = np.concatenate([
        reference * scale + [3 * index, 0, 0]
        for index, scale in enumerate((0.7, 1.0, 1.3, 1.6))
    ])
    return TetraMesh(vertices, np.arange(16).reshape(4, 4), [301, 17, 95, 602])


def run():
    reports = []
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    material = Material(1, 8 / 3, 1)
    kinds = ("periodic", "absorbing", "isolated", "layered")
    if comm.size in (1, 4):
        kinds += ("attenuating",)
    for kind in kinds:
        partitions = None
        prepared = None
        cfg = None
        materials_by_id = None
        field_material = None
        if comm.rank == 0:
            prepared = prepare_mesh(box_mesh(2) if kind == "periodic" else disconnected_mesh(), 2)
            if kind == "absorbing":
                prepared = prepare_mesh(box_mesh(2), 2)
            if kind in ("layered", "attenuating"):
                mesh = box_mesh(2)
                tags = np.where(mesh.vertices[mesh.cells, 2].mean(axis=1) < 0.5, 1, 2)
                prepared = prepare_mesh(TetraMesh(mesh.vertices, mesh.cells, mesh.element_ids, tags), 2)
            boundaries = {
                "periodic": ("periodic",) * 3,
                "absorbing": ("absorbing", "periodic", "absorbing"),
                "isolated": ("absorbing",) * 3,
                "layered": ("absorbing", "periodic", "absorbing"),
                "attenuating": ("absorbing", "periodic", "absorbing"),
            }[kind]
            positions = (
                ((0.5, 0.5, 0.5), (0.25, 0.5, 0.5), (0.17, 0.28, 0.36))
                if kind != "isolated" else tuple(map(tuple, prepared.mesh.vertices[prepared.mesh.cells].mean(axis=1)))
            )
            cfg = Config(
                Path("unused"), material, 2, Source(
                    (0.5, 0.5, 0.5), 20.0, 0.0 if kind == "isolated" else 0.3,
                    0.18, 0.01,
                ),
                0.001, 6, "cpu", Path("unused"), 1, boundaries, positions,
                free_surface=kind in ("layered", "attenuating"),
                sponge=Sponge(0.2, 5, ("xmin", "xmax", "zmin")) if kind in ("layered", "attenuating") else None,
                attenuation=Attenuation((5, 20), 10, 60, 40, 3) if kind == "attenuating" else None,
            )
            field_material = element_material(material, len(prepared.mesh.cells))
            if kind in ("layered", "attenuating"):
                upper = prepared.mesh.physical_tags == 2
                field_material = Material(
                    np.where(upper, 2.0, 1.0), np.where(upper, 14.16, 8 / 3),
                    np.where(upper, 2.88, 1.0),
                )
            model = fit_relaxation(field_material, cfg.attenuation)
            flux_material = model.unrelaxed if model is not None else field_material
            materials_by_id = {
                int(element_id): tuple(getattr(flux_material, name)[row] for name in COEFFICIENTS)
                for row, element_id in enumerate(prepared.mesh.element_ids)
            }
            connection = elastic_connectivity(prepared, boundaries)
            receiver_rows, receiver_weights = receiver_interpolation(prepared, positions)
            validate_source(prepared, cfg)
            partitions = partition_mesh(
                prepared, connection, comm.size, receiver_rows, receiver_weights, material=field_material,
            )
        cfg = comm.bcast(cfg)
        materials_by_id = comm.bcast(materials_by_id)
        partition = comm.scatter(partitions)
        plan = HaloPlan(partition, comm)
        indices = np.indices(plan.shape)
        elements, faces = plan.remote_faces.T
        assert plan.shape == (np.count_nonzero(partition.remote_ranks >= 0),
                              len(partition.prepare().reference.face_barycentric), 9)
        traces = (
            partition.mesh.element_ids[elements, None, None] * 10000
            + faces[:, None, None] * 1000 + indices[1] * 10 + indices[2]
        ).astype(np.float64)
        buffer_ids = [(id(peer.send), id(peer.receive)) for peer in plan.peers]
        for stage in range(2):
            right = plan.exchange(traces + 0.5 * stage)
            for slot, (element, face) in enumerate(plan.remote_faces):
                other_id = partition.connection.neighbour_ids[element, face]
                other_face = partition.connection.neighbour_faces[element, face]
                permutation = partition.connection.permutations[element, face]
                expected = other_id * 10000 + other_face * 1000 + permutation[:, None] * 10 + np.arange(9)
                np.testing.assert_array_equal(right[slot], expected + 0.5 * stage)
        assert [(id(peer.send), id(peer.receive)) for peer in plan.peers] == buffer_ids
        peer_counts = comm.gather(len(plan.peers))
        if kind == "isolated":
            assert len(plan.peers) == 0
        plan.close()
        local = partition.prepare()
        solver = DistributedSolver(
            partition, material, comm, cfg.boundaries,
            smooth_source(local, cfg, domain_bounds=partition.bounds),
            free_surface=cfg.free_surface, sponge=cfg.sponge,
            attenuation=cfg.attenuation,
        )
        heterogeneous_remote = 0
        for element, face in np.ndindex(partition.remote_ranks.shape):
            other_id = partition.connection.neighbour_ids[element, face]
            expected_id = int(other_id if other_id >= 0 else partition.mesh.element_ids[element])
            actual = tuple(getattr(solver.kernel.neighbour_material, name)[element, face] for name in COEFFICIENTS)
            np.testing.assert_array_equal(actual, materials_by_id[expected_id])
            if partition.remote_ranks[element, face] >= 0:
                heterogeneous_remote += actual != materials_by_id[int(partition.mesh.element_ids[element])]
        remote_material_faces = comm.gather(heterogeneous_remote)
        assert solver.halo.material_exchanges == 1
        receivers = ReceiverPlan(partition, comm)
        solver.compile()

        def initial_field(points, components):
            physical = (
                plane_wave(points, 0.03, material, "p", (1, 1, 1))
                + 0.4 * plane_wave(points, 0.02, material, "s", (1, 0, 1))
            )
            return np.pad(physical, ((0, 0), (0, 0), (0, components-9)))

        state = initial_field(local.geometry.nodes, solver.kernel.state_shape[-1])
        surface_mask = solver.kernel.free_faces.copy()
        if kind == "isolated" and comm.rank == 0:
            # One selected exterior face exercises ranks with no surface ownership.
            surface_mask[0, 0] = True
        surfaces = SurfacePlan(local, surface_mask, comm)
        surfaces.compile(state)
        surface_counts = comm.gather(len(surfaces.local_faces))
        surfaces.sample(state)
        dt = 0.3 * solver.max_timestep
        steps = 6
        history = [receivers.sample(state)]
        for step in range(steps):
            state = solver.step(state, step * dt, dt)
            history.append(receivers.sample(state))
        surface = surfaces.sample(state)
        gathered = solver.gather_state(state)
        energy = solver.energy(state)
        local_limits = comm.gather(solver.local_timestep)
        exchanges = comm.gather(solver.halo.exchanges)
        assert solver.halo.exchanges == 4 * steps
        solver.close()
        if comm.rank == 0:
            if kind == "periodic" and comm.size == 4:
                assert max(peer_counts) >= 2
            reference = ElasticSolver(
                prepared, field_material, cfg.boundaries, smooth_source(prepared, cfg),
                free_surface=cfg.free_surface, sponge=cfg.sponge,
                attenuation=cfg.attenuation,
            )
            assert reference.max_timestep == solver.max_timestep
            expected = initial_field(prepared.geometry.nodes, reference.state_shape[-1])
            expected_history = [np.einsum("ri,ric->rc", receiver_weights, expected[receiver_rows, :, :9])]
            for step in range(steps):
                expected = reference.step(expected, step * dt, dt)
                expected_history.append(np.einsum("ri,ric->rc", receiver_weights, np.asarray(expected)[receiver_rows, :, :9]))
            order = np.argsort(prepared.mesh.element_ids)
            ids, final = gathered
            np.testing.assert_array_equal(ids, prepared.mesh.element_ids[order])
            expected = np.asarray(expected)[order]
            surface_rows = np.searchsorted(ids, surfaces.element_ids)
            expected_surface = np.einsum(
                "fij,ejc->efic", prepared.reference.trace, expected,
            )[surface_rows, surfaces.face_indices]
            np.testing.assert_allclose(surface, expected_surface, atol=1e-10, rtol=1e-10)
            global_rows = {int(value): row for row, value in enumerate(prepared.mesh.element_ids)}
            position_rows = np.array([global_rows[int(value)] for value in surfaces.element_ids], dtype=int)
            np.testing.assert_allclose(
                surfaces.positions, prepared.geometry.face_nodes[position_rows, surfaces.face_indices], atol=1e-12,
            )
            field_error = float(np.max(np.abs(final - expected) / np.maximum(1, np.abs(expected))))
            expected_history = np.asarray(expected_history)
            history = np.asarray(history)
            receiver_error = float(np.max(np.abs(history - expected_history) / np.maximum(1, np.abs(expected_history))))
            assert field_error <= 1e-10
            assert receiver_error <= 1e-10
            assert solver.max_timestep == min(local_limits)
            if kind == "isolated" and comm.size == 4:
                assert max(local_limits) > min(local_limits)
            if kind in ("layered", "attenuating") and comm.size == 4:
                assert sum(remote_material_faces) > 0
            if kind == "attenuating":
                assert final.shape[-1] == 27 and solver.halo.shape[-1] == 9
                assert np.linalg.norm(final[..., 9:]) > 0
            if output is not None:
                output.mkdir(parents=True, exist_ok=True)
                np.savez(
                    output / f"{kind}.npz", element_ids=ids, state=final,
                    receivers=history, receiver_element_ids=partition.receiver_element_ids,
                )
            reports.append({
                "case": kind, "peer_counts": peer_counts, "halo_exchanges": exchanges,
                "field_scaled_error": field_error, "receiver_scaled_error": receiver_error,
                "global_dt_limit": solver.max_timestep, "local_dt_limits": local_limits,
                "global_energy": energy,
                "heterogeneous_remote_faces": list(map(int, remote_material_faces)),
                "state_components": final.shape[-1], "trace_components": solver.halo.shape[-1],
                "surface_scaled_error": (
                    float(np.max(np.abs(surface-expected_surface)/np.maximum(1, np.abs(expected_surface))))
                    if surface.size else 0.0
                ),
                "surface_local_face_counts": surface_counts,
                "memory_scaled_error": (
                    float(np.max(np.abs(final[..., 9:]-expected[..., 9:])/np.maximum(1, np.abs(expected[..., 9:]))))
                    if kind == "attenuating" else None
                ),
            })
    affinities = comm.gather(sorted(os.sched_getaffinity(0)))
    if comm.rank == 0:
        assert all(len(mask) == 1 for mask in affinities)
        assert len({mask[0] for mask in affinities}) == comm.size
        print(json.dumps({"mpi_size": comm.size, "affinities": affinities, "cases": reports}))


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        comm.Abort(1)
