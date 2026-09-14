"""Does the 2dx mode reach what a storm inlet would actually measure?

docs/14_rain_plan.md found an odd-even (2dx) mode across the channel in rain
films: 20-50% of a few-mm depth, cell to cell. It concluded that the mode
blocks RainLab-3 (storm inlets), because a single-cell gauge there reads +-25%.
That is a claim about ONE instrument. An inlet takes water over a footprint,
and a zero-mean alternating pattern cancels when averaged, so this probe
measures what the claim actually depends on. It drives the shipped path
(SimulationManager), not a stand-in.

Reported per sample time:
- oddeven: the period-2 amplitude across the channel by projection onto
  (-1)^j, as a fraction of the mean depth, and the fraction of cells that are a
  local extremum across the channel (a pure mode gives 1.0);
- jensen3x3: mean(h^1.5) / mean(h)^1.5 - 1 over a 3x3 footprint. A grate
  inlet's intake goes as depth^1.5, so this is the error of computing it from
  the footprint's mean depth, with no reference field involved;
- cell: the centre cell against the [1,2,1]/4-filtered field. The first run of
  this probe used that filter as "truth" for 3x3/5x5 footprints and a weir
  ratio; the filter moves a CURVED profile by h''/4, so those numbers read the
  channel's cross-section shape as mode (docs/15_cgrid_plan.md, 2a). Kept for
  comparison only;
- the discharge through a full channel cross-section, raw vs filtered.

Run from the repo root:  python docs/probe_2dx_gauge_v1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import config  # noqa: E402
from app.simulation import SimulationManager  # noqa: E402

N = config.TERRAIN_CELLS + 1
CENTRE = N // 2
GAUGE_COLUMNS = (50, 100, 150)


def smooth(field: np.ndarray) -> np.ndarray:
    """[1,2,1]/4 in z then in x; edges keep their value."""
    out = field.copy()
    out[1:-1, :] = 0.25 * field[:-2, :] + 0.5 * field[1:-1, :] + 0.25 * field[2:, :]
    tmp = out.copy()
    out[:, 1:-1] = 0.25 * tmp[:, :-2] + 0.5 * tmp[:, 1:-1] + 0.25 * tmp[:, 2:]
    return out


def fields(manager: SimulationManager) -> tuple[np.ndarray, np.ndarray]:
    fluid = manager.fluid
    h = np.asarray(fluid._h.numpy(), dtype=np.float64).reshape(N, N)
    u = np.asarray(fluid._u.numpy(), dtype=np.float64).reshape(N, N)
    return h, u


def patch_mean(field: np.ndarray, row: int, col: int, half: int) -> float:
    return float(field[row - half:row + half + 1, col - half:col + half + 1].mean())


def mode_metrics(h: np.ndarray) -> tuple[float, float]:
    """Period-2 amplitude across the channel by projection, and the extrema fraction.

    Projection onto (-1)^j is orthogonal to a smooth cross-channel profile, so a
    curved channel bed does not leak into it -- unlike the filter difference,
    which moves a curved profile by h''/4 and was found to read that curvature
    as mode (see docs/15_cgrid_plan.md, 2a).
    """
    rows, cols = slice(CENTRE - 5, CENTRE + 6), slice(20, 181)
    patch = h[rows, cols]
    sign = ((-1.0) ** np.arange(patch.shape[0]))[:, None]
    dev = patch - patch.mean(axis=0, keepdims=True)
    amplitude = abs(float((dev * sign).mean()))
    c = patch[1:-1, :]
    extrema = ((c > patch[2:, :]) & (c > patch[:-2, :])) | ((c < patch[2:, :]) & (c < patch[:-2, :]))
    return amplitude / max(float(patch.mean()), 1.0e-12), float(extrema.mean())


def sample(label: str, manager: SimulationManager, rows_half: int) -> dict:
    h, u = fields(manager)
    hs = smooth(h)
    record = {"t": round(manager.sim_time)}
    # "cell" is still referenced to the [1,2,1] filter and so still carries bed
    # curvature; it is kept only to compare with the first run. The decision
    # metrics are "oddeven", "extrema" and "jensen3x3".
    worst = {"cell": 0.0, "jensen3x3": 0.0}
    for col in GAUGE_COLUMNS:
        ref = patch_mean(hs, CENTRE, col, 0)
        if ref <= config.FLUID_DRY_DEPTH:
            continue
        worst["cell"] = max(worst["cell"], abs(float(h[CENTRE, col]) - ref) / ref)
        # the weir error itself: nonlinear intake over the footprint against the
        # intake computed from the footprint's mean depth -- no reference field
        foot = h[CENTRE - 1:CENTRE + 2, col - 1:col + 2]
        jensen = float((foot ** 1.5).mean() / foot.mean() ** 1.5) - 1.0
        worst["jensen3x3"] = max(worst["jensen3x3"], abs(jensen))
    oddeven, extrema = mode_metrics(h)
    record["oddeven"] = round(100 * oddeven, 2)
    record["extrema"] = round(extrema, 2)
    q = h * u
    qs = smooth(h) * smooth(u)
    rows = slice(CENTRE - rows_half, CENTRE + rows_half + 1)
    section = [float(q[rows, c].sum() * config.TERRAIN_CELL_SIZE) for c in GAUGE_COLUMNS]
    section_s = [float(qs[rows, c].sum() * config.TERRAIN_CELL_SIZE) for c in GAUGE_COLUMNS]
    record.update({k: round(100 * v, 2) for k, v in worst.items()})
    record["section_q_m3s"] = [round(v, 4) for v in section]
    record["section_q_smooth"] = [round(v, 4) for v in section_s]
    record["centre_mm"] = round(1000 * float(h[CENTRE, 100]), 3)
    record["substeps"] = manager.fluid.diagnostics()["substeps"]
    print(label, record, flush=True)
    return record


def valley(intensity: float, seconds: float) -> None:
    manager = SimulationManager()
    info = manager.apply_terrain_river({})["river"]
    manager.apply_water_level(0.0)
    manager.apply_edge_inflow(False)
    manager.apply_rain({"intensity_mm_h": intensity})
    manager.start()
    rows_half = int(info.get("bed_width", 12.0) / config.TERRAIN_CELL_SIZE) + 3
    for t in range(60, int(seconds) + 1, 60):
        while manager.sim_time < t:
            manager._step_once()
        sample(f"valley {intensity:g} mm/h", manager, rows_half)
    manager.stop()


def river(discharge: float, seconds: float) -> None:
    manager = SimulationManager()
    manager.apply_terrain_river({})
    manager.apply_water_level(0.0)
    manager.apply_edge_inflow(False)
    manager.apply_river_inlet({"enabled": True, "width_m": 12.0, "discharge_m3s": discharge})
    manager.apply_river_outlet({"width_m": 20.0})
    manager.start()
    for t in range(60, int(seconds) + 1, 60):
        while manager.sim_time < t:
            manager._step_once()
        sample(f"river {discharge:g} m3/s", manager, 10)
    manager.stop()


def flat_box(intensity: float, seconds: float) -> None:
    """No bed gradient: the exact answer is a uniform sheet, so any mode is numerical."""
    manager = SimulationManager()
    manager.apply_water_level(0.0)
    manager.apply_water_outflow(False)
    manager.apply_edge_inflow(False)
    manager.apply_rain({"intensity_mm_h": intensity})
    manager.start()
    while manager.sim_time < seconds:
        manager._step_once()
    h, _ = fields(manager)
    wet = h[2:-2, 2:-2]
    print(f"flat box {intensity:g} mm/h t={manager.sim_time:.0f}: mean {wet.mean() * 1000:.4f} mm "
          f"std {wet.std() * 1000:.6f} mm  max|h - smooth| "
          f"{np.abs(h - smooth(h))[2:-2, 2:-2].max() * 1000:.6f} mm", flush=True)
    manager.stop()


if __name__ == "__main__":
    flat_box(50.0, 120.0)
    valley(50.0, 300.0)
    valley(20.0, 300.0)
    river(12.0, 180.0)
