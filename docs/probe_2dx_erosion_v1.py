"""Does the 2dx mode still carve the bed? (docs/07_river_plan.md, item 1)

v0.12.0 left 8 cells cut deeper than 1 m at the side ends of the inlet band
after 150 s, which TEST_REPORT.md 13.4 attributes to the odd-even mode locking
in under erosion. This reproduces the release configuration through the
shipped path -- generated valley, local inlet 12 m3/s, 20 m outlet, erosion on
-- and reports how much of the bed change is the period-2 pattern.

Run from the repo root:  python docs/probe_2dx_erosion_v1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import config  # noqa: E402
from app.simulation import SimulationManager  # noqa: E402

N = config.TERRAIN_CELLS + 1


def smooth(field: np.ndarray) -> np.ndarray:
    out = field.copy()
    out[1:-1, :] = 0.25 * field[:-2, :] + 0.5 * field[1:-1, :] + 0.25 * field[2:, :]
    tmp = out.copy()
    out[:, 1:-1] = 0.25 * tmp[:, :-2] + 0.5 * tmp[:, 1:-1] + 0.25 * tmp[:, 2:]
    return out


def bed(manager: SimulationManager) -> np.ndarray:
    return np.asarray(manager.fluid._bed_terrain.numpy(), dtype=np.float64).reshape(N, N)


manager = SimulationManager()
manager.apply_terrain_river({})
manager.apply_water_level(0.0)
manager.apply_edge_inflow(False)
manager.apply_river_inlet({"enabled": True, "width_m": 12.0, "discharge_m3s": 12.0})
manager.apply_river_outlet({"width_m": 20.0})
manager.apply_water_erosion(True)
manager.start()
start = None
for t in (30, 90, 150):
    while manager.sim_time < t:
        manager._step_once()
        if start is None:
            start = bed(manager)
    change = bed(manager) - start
    mode = change - smooth(change)
    inner = (slice(2, -2), slice(2, -2))
    diag = manager.fluid.diagnostics()
    cut = -change
    print(f"t={manager.sim_time:5.0f}  cells cut >1 m {int((cut > 1.0).sum()):4d}  "
          f">0.25 m {int((cut > 0.25).sum()):4d}  max cut {cut.max():.3f} m  "
          f"max deposit {change.max():.3f} m  |change| mean {np.abs(change[inner]).mean() * 1000:.3f} mm  "
          f"period-2 part {np.abs(mode[inner]).mean() * 1000:.3f} mm "
          f"({100 * np.abs(mode[inner]).mean() / max(np.abs(change[inner]).mean(), 1e-12):.0f}%)  "
          f"max|u| {diag['max_velocity']:.2f}  ledger {diag['volume_error_m3']:.4f}", flush=True)
    worst = np.unravel_index(np.argmax(cut), cut.shape)
    r, c = worst
    print(f"   deepest cut at row {r} col {c}; 5x5 around it (m):\n",
          np.array2string(change[max(0, r - 2):r + 3, max(0, c - 2):c + 3], precision=3), flush=True)
manager.stop()
