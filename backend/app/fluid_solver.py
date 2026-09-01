"""Fluid solver contract with boundary coupling and internal substeps."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


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
