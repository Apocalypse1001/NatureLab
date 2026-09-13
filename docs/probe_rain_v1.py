"""RainLab-1 probe (docs/14_rain_plan.md, "Зонд до кернелов").

Rain wets EVERY cell of the map with films just above FLUID_DRY_DEPTH -- a
regime the substep schedule, the friction floor and the 2dx mode were never
measured in. This drives the shipped path (SimulationManager -> fluid solver),
not a numpy stand-in, and answers four questions:

1. does the substep count blow up (CFL-limited frames) on a fully wet map?
2. does a checkerboard (2dx) grow on the films?
3. does the volume ledger close (added = on the map + removed)?
4. does rain on a river valley actually collect in the channel within a session?

Run from the repo root:  python docs/probe_rain_v1.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import config  # noqa: E402
from app.simulation import SimulationManager  # noqa: E402

N = config.TERRAIN_CELLS + 1


def checkerboard_ratio(depth: np.ndarray) -> float:
    """Energy in the odd-even mode relative to the smooth gradient, wet cells only.

    A smooth sheet gives a second difference much smaller than its first
    difference; an alternating pattern gives a second difference about twice
    the first. Measured along x and z together.
    """
    wet = depth > config.FLUID_DRY_DEPTH
    second = np.abs(depth[1:-1, 2:] - 2 * depth[1:-1, 1:-1] + depth[1:-1, :-2]) \
        + np.abs(depth[2:, 1:-1] - 2 * depth[1:-1, 1:-1] + depth[:-2, 1:-1])
    first = np.abs(depth[1:-1, 2:] - depth[1:-1, :-2]) \
        + np.abs(depth[2:, 1:-1] - depth[:-2, 1:-1])
    mask = wet[1:-1, 1:-1]
    if not mask.any():
        return 0.0
    return float(second[mask].mean() / max(first[mask].mean(), 1.0e-9))


def run(label: str, manager: SimulationManager, seconds: float) -> dict:
    steps = int(seconds / config.FIXED_DT)
    limited = 0
    max_substeps = 0
    started = time.perf_counter()
    for _ in range(steps):
        manager._step_once()
        diag = manager.fluid.diagnostics()
        max_substeps = max(max_substeps, diag["substeps"])
        limited += int(bool(diag["cfl_limited"]))
    wall = time.perf_counter() - started
    diag = manager.fluid.diagnostics()
    depth = np.asarray(manager.fluid._h.numpy(), dtype=np.float32).reshape(N, N)
    result = {
        "label": label, "sim_s": seconds, "wall_s": round(wall, 1),
        "max_substeps": max_substeps, "cfl_limited_frames": limited,
        "volume_m3": round(diag["volume_m3"], 3), "added_m3": round(diag["added_m3"], 3),
        "removed_m3": round(diag["removed_m3"], 3),
        "volume_error_m3": round(diag["volume_error_m3"], 5),
        "max_depth_m": round(diag["max_depth"], 4),
        "max_velocity_m_s": round(diag["max_velocity"], 3),
        "wet_cells": diag["wet_cells"],
        "checkerboard": round(checkerboard_ratio(depth), 3),
    }
    print(result, flush=True)
    return {**result, "depth": depth}


def flat_closed(intensity: float, seconds: float) -> None:
    """Question 1-3 with nothing else going on: a flat closed box."""
    manager = SimulationManager()
    manager.apply_water_level(0.0)
    manager.apply_water_outflow(False)
    manager.apply_edge_inflow(False)
    manager.apply_rain({"intensity_mm_h": intensity})
    manager.start()
    manager.pause()
    out = run(f"flat closed {intensity} mm/h", manager, seconds)
    # rain is poured in RAIN_APPLY_STEP_M portions, so what has FALLEN is the
    # requested depth minus the portion still pending -- comparing against the
    # raw request made 1 mm/h look like a 40% loss (0.673 vs 0.404 m3) when
    # exactly one portion had been poured and nothing was missing
    pending = manager.fluid.diagnostics()["rain_pending_m"]
    expected = (intensity / 3.6e6 * seconds - pending) * N * N * config.TERRAIN_CELL_SIZE ** 2
    print(f"  fallen (request - pending) {expected:.3f} m3, measured {out['volume_m3']:.3f} m3")
    manager.stop()


def valley(intensity: float, seconds: float) -> None:
    """Question 4: rain on a generated valley, no inlet, open outlet."""
    manager = SimulationManager()
    info = manager.apply_terrain_river({})["river"]
    manager.apply_water_level(0.0)
    manager.apply_edge_inflow(False)
    manager.apply_rain({"intensity_mm_h": intensity})
    manager.start()
    manager.pause()
    out = run(f"valley {intensity} mm/h", manager, seconds)
    depth = out["depth"]
    bed = manager.world.terrain.heights
    # river_valley runs the channel west -> east along the centre ROW (z), so
    # the cross-section is taken across rows, not columns
    centre = N // 2
    half_width = max(1, int(info.get("bed_width", 12.0) / config.TERRAIN_CELL_SIZE / 2))
    channel = depth[centre - half_width:centre + half_width + 1, :]
    banks = np.concatenate([depth[:centre - 3 * half_width, :],
                            depth[centre + 3 * half_width + 1:, :]], axis=0)
    print(f"  channel mean depth {channel.mean() * 1000:.2f} mm, "
          f"bank mean depth {banks.mean() * 1000:.2f} mm, "
          f"bed span {float(bed.min()):.2f}..{float(bed.max()):.2f} m")
    manager.stop()


if __name__ == "__main__":
    for mm_h in (1.0, 20.0, 100.0):
        flat_closed(mm_h, 60.0)
    for mm_h in (20.0, 50.0, 100.0):
        valley(mm_h, 300.0)
