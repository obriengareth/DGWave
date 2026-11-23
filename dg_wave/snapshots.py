"""Incremental, surface-only velocity output; no history-sized volume buffers."""

import json

import numpy as np

from dg_wave.surface import display_triangles, surface_trace_kernel


class SurfaceOutput:
    def __init__(self, prepared, mask, config, *, comm=None, restart_step=None):
        self.comm, self.config = comm, config
        self.directory = config.output / "surface"
        self.plan = None
        if comm is None:
            self.extract = surface_trace_kernel(prepared, mask, components=slice(0, 3))
            rows, faces = np.nonzero(mask)
            self.geometry = {
                "face_positions": prepared.geometry.face_nodes[mask],
                "face_element_ids": prepared.mesh.element_ids[rows], "face_indices": faces,
            }
        else:
            from dg_wave.distributed import SurfacePlan

            self.plan = SurfacePlan(prepared, mask, comm, components=slice(0, 3))
            self.geometry = {
                "face_positions": self.plan.positions, "face_element_ids": self.plan.element_ids,
                "face_indices": self.plan.face_indices,
            }

        def initialise():
            path = self.directory / "manifest.json"
            self.directory.mkdir(parents=True, exist_ok=True)
            if restart_step is None:
                if path.exists() or any(self.directory.iterdir()):
                    raise ValueError("refusing to overwrite existing surface snapshots")
                self.steps = []
                np.savez_compressed(
                    self.directory / "geometry.npz", **self.geometry,
                    triangles=display_triangles(prepared.reference, len(self.geometry["face_indices"])),
                )
            else:
                manifest = json.loads(path.read_text())
                self.steps = manifest["steps"]
                expected = list(range(0, restart_step+1, config.surface_interval))
                if (
                    self.steps != expected or manifest["dt_s"] != config.dt
                    or manifest["interval_steps"] != config.surface_interval
                    or any(not (self.directory/f"step{step:08d}.npz").is_file() for step in expected)
                ):
                    raise ValueError("surface restart requires matching committed snapshot history in the output directory")

        self._root(initialise)

    def _root(self, operation):
        if self.comm is None:
            return operation()
        from dg_wave.mpi_run import root_operation

        return root_operation(self.comm, operation)

    def compile(self, state):
        if self.plan is None:
            self.extract = self.extract.lower(state).compile()
        else:
            self.plan.compile(state)

    def record(self, state, step):
        if self.plan is None:
            velocity = np.asarray(self.extract(state).block_until_ready())
        else:
            velocity = self.plan.sample(state)

        def write():
            if not np.isfinite(velocity).all():
                raise RuntimeError("nonfinite surface velocity")
            path = self.directory / f"step{step:08d}.npz"
            if path.exists() or (self.steps and step <= self.steps[-1]):
                raise ValueError("refusing to replace or reorder a surface snapshot")
            np.savez_compressed(path, step=step, time_s=step*self.config.dt, velocity_m_s=velocity)
            self.steps.append(step)
            manifest = {
                "dt_s": self.config.dt, "interval_steps": self.config.surface_interval,
                "steps": self.steps, "times_s": [value*self.config.dt for value in self.steps],
                "components": ["vx", "vy", "vz"], "geometry": "geometry.npz",
                "strategy": "selected free-face velocity nodes only; no volume state gather",
            }
            from dg_wave.checkpoint import _atomic

            _atomic(self.directory/"manifest.json", lambda stream: stream.write(json.dumps(manifest).encode()))

        self._root(write)
