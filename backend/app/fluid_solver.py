"""Fluid solver contract with boundary coupling and internal substeps."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from . import config


class FluidSolver:
    def initialize(self, world) -> None: ...
    def set_boundaries(self, terrain, obstacles: dict) -> None: ...
    def advance(self, global_dt: float, max_substeps: int, stability_dt: float) -> int: ...
    def sample_for_bodies(self, positions: np.ndarray) -> dict: ...
    def reset(self) -> None: ...
    def get_water_height(self, x: float = 0.0, z: float = 0.0) -> float: ...
    def get_velocity_field(self) -> Optional[np.ndarray]: ...


class PlaceholderFluidSolver(FluidSolver):
    def __init__(self) -> None:
        self._world = None
        self._terrain = None
        self._obstacles = {}
        self._level = 0.5
        self._time = 0.0
        self.last_substeps = 0

    def initialize(self, world) -> None:
        self._world = world
        self._terrain = world.terrain
        self._level = world.water.level
        self._time = 0.0

    def set_boundaries(self, terrain, obstacles: dict) -> None:
        self._terrain = terrain
        self._obstacles = obstacles

    def advance(self, global_dt: float, max_substeps: int, stability_dt: float) -> int:
        substeps = max(1, min(max_substeps, math.ceil(global_dt / stability_dt)))
        dt = global_dt / substeps
        for _ in range(substeps):
            self._time += dt
            if self._world is not None:
                self._level = self._world.water.level
        self.last_substeps = substeps
        return substeps

    def sample_for_bodies(self, positions: np.ndarray) -> dict:
        count = len(positions)
        depths = np.zeros(count, dtype=np.float32)
        if self._terrain is not None:
            for i, position in enumerate(positions):
                terrain_h = self._terrain.height_at(float(position[0]), float(position[2]))
                depths[i] = max(0.0, self._level - terrain_h)
        velocities = np.tile(np.array([1.5, 0.0, 0.0], dtype=np.float32), (count, 1))
        forces = velocities * depths[:, None]
        return {"depths": depths, "velocities": velocities, "forces": forces}

    def reset(self) -> None:
        self._time = 0.0
        if self._world is not None:
            self._level = self._world.water.level

    def set_level(self, level: float) -> None:
        self._level = level

    def get_water_height(self, x: float = 0.0, z: float = 0.0) -> float:
        return self._level

    def get_velocity_field(self) -> Optional[np.ndarray]:
        return np.array([1.5, 0.0, 0.0], dtype=np.float32)


def _bilinear(grid: np.ndarray, gx: np.ndarray, gz: np.ndarray) -> np.ndarray:
    """Sample a (ny, nx) grid at fractional (gx, gz) cell coordinates."""
    ny, nx = grid.shape
    gx = np.clip(gx, 0, nx - 1)
    gz = np.clip(gz, 0, ny - 1)
    i0 = np.floor(gx).astype(np.int64)
    j0 = np.floor(gz).astype(np.int64)
    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    fx = gx - i0
    fz = gz - j0
    return ((1 - fx) * (1 - fz) * grid[j0, i0] + fx * (1 - fz) * grid[j0, i1]
             + (1 - fx) * fz * grid[j1, i0] + fx * fz * grid[j1, i1])


class ShallowWaterFluidSolver(FluidSolver):
    """Height-field shallow water via outflow-limited cell exchange.

    Each cell exchanges water with its 4 neighbours proportionally to the
    total-height difference (terrain + depth), clamped so a cell can never
    drain more water than it holds. This keeps the scheme unconditionally
    mass-conservative (no water created or destroyed away from an explicit
    source) and correctly diverts flow around obstacles/terrain changes,
    which is the causal-realism bar this project actually requires (see
    docs/04_TZ_v0.3_roadmap.md section 3) rather than a full Navier-Stokes
    solver. World edges are closed (zero-flux) boundaries.

    Obstacle footprint is currently a fixed radius per registered rigid
    body (config.FLUID_OBSTACLE_RADIUS_M) rather than the object's real
    scale — RigidStateBuffer does not carry scale yet. That is a follow-up,
    not silently assumed correct.
    """

    def __init__(self) -> None:
        self._world = None
        self._terrain = None
        self._depth: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        self._flow_x: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        self._flow_z: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        self._obstacle_mask: np.ndarray = np.zeros((1, 1), dtype=bool)
        self._time = 0.0
        self.last_substeps = 0

    # ------------------------------------------------------------------ setup
    def initialize(self, world) -> None:
        self._world = world
        self._terrain = world.terrain
        shape = self._terrain.heights.shape
        self._depth = np.maximum(
            0.0, world.water.level - self._terrain.heights).astype(np.float32)
        self._flow_x = np.zeros(shape, dtype=np.float32)
        self._flow_z = np.zeros(shape, dtype=np.float32)
        self._obstacle_mask = np.zeros(shape, dtype=bool)
        self._time = 0.0

    def set_boundaries(self, terrain, obstacles: dict) -> None:
        self._terrain = terrain
        if self._depth.shape != terrain.heights.shape:
            # terrain resized (e.g. loaded world) -> reinitialise the field flat
            self._depth = np.zeros(terrain.heights.shape, dtype=np.float32)
        self._obstacle_mask = self._rasterize_obstacles(terrain, obstacles)
        self._depth[self._obstacle_mask] = 0.0

    def _rasterize_obstacles(self, terrain, obstacles: dict) -> np.ndarray:
        mask = np.zeros(terrain.heights.shape, dtype=bool)
        positions = obstacles.get("positions") if obstacles else None
        if positions is None or len(positions) == 0:
            return mask
        ny, nx = mask.shape
        r_cells = max(1.0, config.FLUID_OBSTACLE_RADIUS_M / terrain.cell_size)
        yy, xx = np.mgrid[0:ny, 0:nx]
        for pos in positions:
            gx = float(pos[0]) / terrain.cell_size + terrain.width / 2
            gz = float(pos[2]) / terrain.cell_size + terrain.height / 2
            mask |= (xx - gx) ** 2 + (yy - gz) ** 2 <= r_cells ** 2
        return mask

    # ------------------------------------------------------------------ step
    def advance(self, global_dt: float, max_substeps: int, stability_dt: float) -> int:
        substeps = max(1, min(max_substeps, math.ceil(global_dt / stability_dt)))
        dt = global_dt / substeps
        for _ in range(substeps):
            self._step(dt)
            self._time += dt
        self.last_substeps = substeps
        return substeps

    def _step(self, dt: float) -> None:
        terrain_h = self._terrain.heights
        depth = self._depth
        effective = terrain_h.astype(np.float32).copy()
        effective[self._obstacle_mask] += 1e4  # solid: never a flow target
        total = effective + depth

        padded = np.pad(total, 1, mode="edge")
        h_right = padded[1:-1, 2:]
        h_left = padded[1:-1, :-2]
        h_down = padded[2:, 1:-1]
        h_up = padded[:-2, 1:-1]

        gain = config.FLUID_FLOW_GAIN
        flow_right = np.maximum(0.0, total - h_right) * gain
        flow_left = np.maximum(0.0, total - h_left) * gain
        flow_down = np.maximum(0.0, total - h_down) * gain
        flow_up = np.maximum(0.0, total - h_up) * gain

        outflow = flow_right + flow_left + flow_down + flow_up
        available = depth / dt
        scale = np.ones_like(depth)
        draining = outflow > available
        scale[draining] = available[draining] / np.maximum(outflow[draining], 1e-9)
        flow_right *= scale
        flow_left *= scale
        flow_down *= scale
        flow_up *= scale

        inflow_from_left = np.pad(flow_right, ((0, 0), (1, 0)))[:, :-1]
        inflow_from_right = np.pad(flow_left, ((0, 0), (0, 1)))[:, 1:]
        inflow_from_up = np.pad(flow_down, ((1, 0), (0, 0)))[:-1, :]
        inflow_from_down = np.pad(flow_up, ((0, 1), (0, 0)))[1:, :]
        inflow = inflow_from_left + inflow_from_right + inflow_from_up + inflow_from_down

        new_depth = depth + dt * (inflow - (flow_right + flow_left + flow_down + flow_up))
        new_depth = np.maximum(0.0, new_depth)
        new_depth[self._obstacle_mask] = 0.0
        new_depth[new_depth < config.FLUID_MIN_DEPTH] = 0.0

        self._depth = new_depth.astype(np.float32)
        self._flow_x = (flow_right - flow_left).astype(np.float32)
        self._flow_z = (flow_down - flow_up).astype(np.float32)

    # ------------------------------------------------------------------ readback
    def sample_for_bodies(self, positions: np.ndarray) -> dict:
        """Sample depth/velocity around each body, not at its exact centre.

        Each registered body carves a dry hole into its own footprint (see
        set_boundaries), so sampling the centre would always read depth=0
        and an object could never float. Sample a small ring just outside
        the body's own obstacle radius and average it instead.
        """
        count = len(positions)
        if count == 0 or self._terrain is None:
            return {"depths": np.zeros(0, dtype=np.float32),
                    "velocities": np.zeros((0, 3), dtype=np.float32),
                    "forces": np.zeros((0, 3), dtype=np.float32)}
        cell = self._terrain.cell_size
        gx = positions[:, 0] / cell + self._terrain.width / 2
        gz = positions[:, 2] / cell + self._terrain.height / 2
        ring = max(1.0, config.FLUID_OBSTACLE_RADIUS_M / cell) + 0.5
        depth_samples, vx_samples, vz_samples = [], [], []
        for dgx, dgz in ((ring, 0.0), (-ring, 0.0), (0.0, ring), (0.0, -ring)):
            depth_samples.append(_bilinear(self._depth, gx + dgx, gz + dgz))
            vx_samples.append(_bilinear(self._flow_x, gx + dgx, gz + dgz))
            vz_samples.append(_bilinear(self._flow_z, gx + dgx, gz + dgz))
        depths = np.mean(depth_samples, axis=0).astype(np.float32)
        vx = np.mean(vx_samples, axis=0).astype(np.float32)
        vz = np.mean(vz_samples, axis=0).astype(np.float32)
        velocities = np.column_stack([vx, np.zeros(count, dtype=np.float32), vz])
        forces = velocities * depths[:, None]
        return {"depths": depths, "velocities": velocities, "forces": forces}

    def reset(self) -> None:
        if self._world is not None:
            self.initialize(self._world)

    def get_water_height(self, x: float = 0.0, z: float = 0.0) -> float:
        if self._terrain is None:
            return 0.0
        gx = np.array([x / self._terrain.cell_size + self._terrain.width / 2])
        gz = np.array([z / self._terrain.cell_size + self._terrain.height / 2])
        terrain_h = float(self._terrain.height_at(x, z))
        depth = float(_bilinear(self._depth, gx, gz)[0])
        return terrain_h + depth

    def get_velocity_field(self) -> Optional[np.ndarray]:
        return np.dstack([self._flow_x, np.zeros_like(self._flow_x), self._flow_z])

    def total_volume(self) -> float:
        """Sum of water depth over all cells; used by conservation tests."""
        return float(self._depth.sum())
