"""Compute layer abstraction.

ComputeEngine is the interface used by the SimulationManager. WarpEngine
fulfils it via NVIDIA Warp (CUDA when available, CPU otherwise). If Warp is
missing entirely, NumpyEngine keeps the simulation alive so the architecture
stays testable on any machine.
"""
from __future__ import annotations

import platform
from typing import Dict, Optional

import numpy as np

from . import config

try:
    import warp as wp
    WARP_IMPORTED = True
except Exception:  # pragma: no cover - environment without warp
    wp = None
    WARP_IMPORTED = False


class ComputeEngine:
    """Interface: particles integrate on the compute device each fixed step."""

    kind = "none"
    warp_available = False
    cuda = False
    device = "cpu"
    device_name = "unknown"

    def init_particles(self, count: int, water_level: float) -> None: ...
    def step_particles(self, dt: float) -> None: ...
    def positions(self) -> np.ndarray: ...
    def visualization_positions(self, limit: int) -> np.ndarray:
        positions = self.positions()
        if len(positions) <= limit:
            return positions
        stride = max(1, len(positions) // limit)
        return positions[::stride][:limit]
    def selftest(self) -> Dict[str, float]: ...


class NumpyEngine(ComputeEngine):
    kind = "numpy"
    device_name = platform.processor() or "CPU"

    def __init__(self) -> None:
        self._pos: Optional[np.ndarray] = None
        self._vel: Optional[np.ndarray] = None

    def init_particles(self, count: int, water_level: float) -> None:
        rng = np.random.default_rng(42)
        half = config.WORLD_SIZE_M / 2
        self._pos = np.column_stack([
            rng.uniform(-half, half, count),
            np.full(count, water_level + 0.05, dtype=np.float64),
            rng.uniform(-half, half, count),
        ]).astype(np.float64)
        self._vel = np.column_stack([
            rng.uniform(0.5, 2.5, count),
            np.zeros(count),
            rng.uniform(-0.3, 0.3, count),
        ])

    def step_particles(self, dt: float) -> None:
        assert self._pos is not None and self._vel is not None
        self._pos += self._vel * dt
        half = config.WORLD_SIZE_M / 2
        for axis in (0, 2):
            over = self._pos[:, axis] > half
            self._pos[over, axis] -= 2 * half
            under = self._pos[:, axis] < -half
            self._pos[under, axis] += 2 * half

    def positions(self) -> np.ndarray:
        return np.zeros((0, 3), dtype=np.float32) if self._pos is None \
            else self._pos.astype(np.float32, copy=False)

    def visualization_positions(self, limit: int) -> np.ndarray:
        positions = self.positions()
        if len(positions) <= limit:
            return positions
        stride = max(1, len(positions) // limit)
        return positions[::stride][:limit]

    def selftest(self) -> Dict[str, float]:
        pos = np.array([[0.0, 0.0, 0.0]])
        vel = np.array([[1.0, 0.0, 0.0]])
        pos += vel * 0.1
        return {"points": 1, "moved_by": float(pos[0, 0])}


class WarpEngine(ComputeEngine):
    kind = "warp"
    warp_available = True

    def __init__(self) -> None:
        wp.init()
        self.device = self._pick_device()
        dev = wp.get_device(self.device)
        self.cuda = dev.is_cuda
        self.device_name = dev.name if self.cuda else (
            "CPU: " + (platform.processor() or "generic"))
        self._pos: Optional[wp.array] = None
        self._vel: Optional[wp.array] = None
        self._count = 0
        self._define_kernels()

    @staticmethod
    def _pick_device() -> str:
        if "cuda:0" in wp.get_devices():
            return "cuda:0"
        return "cpu"

    def _define_kernels(self) -> None:
        engine = self

        @wp.kernel
        def integrate(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
                      dt: float, bound: float):
            i = wp.tid()
            p = pos[i] + vel[i] * dt
            if p.x > bound:
                p = wp.vec3(p.x - 2.0 * bound, p.y, p.z)
            if p.x < -bound:
                p = wp.vec3(p.x + 2.0 * bound, p.y, p.z)
            if p.z > bound:
                p = wp.vec3(p.x, p.y, p.z - 2.0 * bound)
            if p.z < -bound:
                p = wp.vec3(p.x, p.y, p.z + 2.0 * bound)
            pos[i] = p

        engine._integrate = integrate

    def init_particles(self, count: int, water_level: float) -> None:
        rng = np.random.default_rng(42)
        half = config.WORLD_SIZE_M / 2
        pos = np.column_stack([
            rng.uniform(-half, half, count),
            np.full(count, water_level + 0.05),
            rng.uniform(-half, half, count),
        ]).astype(np.float32)
        vel = np.column_stack([
            rng.uniform(0.5, 2.5, count),
            np.zeros(count, dtype=np.float32),
            rng.uniform(-0.3, 0.3, count),
        ]).astype(np.float32)
        self._pos = wp.array(pos, dtype=wp.vec3, device=self.device)
        self._vel = wp.array(vel, dtype=wp.vec3, device=self.device)
        self._count = count

    def step_particles(self, dt: float) -> None:
        assert self._pos is not None
        wp.launch(self._integrate, dim=self._count,
                  inputs=[self._pos, self._vel, dt, config.WORLD_SIZE_M / 2],
                  device=self.device)

    def positions(self) -> np.ndarray:
        if self._pos is None:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(self._pos.numpy(), dtype=np.float32)

    def visualization_positions(self, limit: int) -> np.ndarray:
        # Foundation 0.2 separates visualization from simulation count. A future
        # CUDA kernel can gather these samples without a full device readback.
        positions = self.positions()
        if len(positions) <= limit:
            return positions
        stride = max(1, len(positions) // limit)
        return positions[::stride][:limit]

    def selftest(self) -> Dict[str, float]:
        """Test kernel required by the spec: 100 000 points, position += velocity."""
        n = 100_000
        pos = wp.array(np.zeros((n, 3), dtype=np.float32), dtype=wp.vec3,
                       device=self.device)
        vel = wp.array(np.tile([1.0, 0.0, 0.0], (n, 1)).astype(np.float32),
                       dtype=wp.vec3, device=self.device)
        wp.launch(self._integrate, dim=n,
                  inputs=[pos, vel, 1.0, 1e9], device=self.device)
        moved = float(np.asarray(pos.numpy(), dtype=np.float32)[0, 0])
        return {"points": n, "moved_by": moved, "device": self.device}


def create_engine() -> ComputeEngine:
    """Pick the best available compute backend, never crash on import."""
    if WARP_IMPORTED:
        try:
            return WarpEngine()
        except Exception as exc:  # pragma: no cover
            print(f"[naturelab] Warp initialisation failed: {exc}; "
                  f"falling back to NumPy engine")
    return NumpyEngine()
