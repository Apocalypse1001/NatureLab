"""Probe: does the river's bed live on a long run? (docs/07_river_plan.md, erosion)

The river plan's calibration question is not "is the capacity constant right"
-- the v0.12.0 measurement showed it is not what limits erosion -- but whether
the channel behaves like a river over minutes: does it cut evenly or dig holes,
where does it deposit, does it silt itself up, does the sediment ledger close,
and does the 2dx mode come back now the grid is staggered.

Shipped path: generated valley, local inlet, 20 m river outlet, erosion on,
primed channel. Every `step` seconds it reports, for the INTERIOR (|x| <= 85 m)
and for the two boundary zones (|x| > 90 m) separately:

- bed change over the channel band |z| <= 8 m, in thirds of the interior: mean
  and the deepest cut / highest fill;
- cells cut or filled by more than 0.1 m and 0.5 m;
- cells that changed at the SEDIMENT_MAX_BED_CHANGE clamp over the interval --
  a number read off a clamp-pinned cell measures the limiter, not erosion;
- the share of the bed change that is the period-2 (odd-even) pattern;
- sediment ledger: bed lost + load the inlet assigned = suspended + carried
  out + what the semi-Lagrangian transport did not keep.

`shield` holds the bed in the boundary zones at its starting height every tick,
to see whether the interior changes when the boundaries stop digging.

    python docs/probe_erosion_long_v1.py 900 150 12
    python docs/probe_erosion_long_v1.py 900 150 12 shield
"""
import sys

import numpy as np

sys.path.insert(0, "backend")
from app import config  # noqa: E402
from app.simulation import SimulationManager  # noqa: E402

N = config.TERRAIN_CELLS + 1
total = float(sys.argv[1]) if len(sys.argv) > 1 else 900.0
step = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
Q = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
shield = len(sys.argv) > 4 and sys.argv[4] == "shield"

m = SimulationManager()
m.apply_terrain_river({})
m.apply_water_level(0.0)
m.apply_edge_inflow(False)
m.apply_river_inlet({"enabled": True, "width_m": 12.0, "discharge_m3s": Q})
m.apply_river_outlet({"width_m": 20.0, "kind": "river"})
m.apply_water_erosion(True)
m.start()
cell = m.world.terrain.cell_size
area = cell * cell
x = (np.arange(N) - (N - 1) * 0.5) * cell
X, Z = np.meshgrid(x, x)
bed0 = np.asarray(m.fluid.get_terrain_heights(), dtype=np.float64).reshape(N, N)
m.fluid._h.assign(np.maximum(bed0[N // 2, :][None, :] + 0.7 - bed0, 0.0).astype(np.float32).ravel())
m._step_once()


def bed() -> np.ndarray:
    return np.asarray(m.fluid._bed_terrain.numpy(), dtype=np.float64).reshape(N, N)


def odd_even_share(change: np.ndarray) -> float:
    inner = change[2:-2, 2:-2]
    sign = np.fromfunction(lambda j, i: (-1.0) ** (i + j), inner.shape)
    projection = float(np.sum(inner * sign)) / inner.size
    rms = float(np.sqrt(np.mean(inner ** 2)))
    return abs(projection) / rms if rms > 1e-9 else 0.0


start_bed = bed()
boundary = (np.abs(x) > 90.0)[None, :] & np.ones((N, 1), bool)
d0 = m.fluid.diagnostics()
start_sed = float(np.asarray(m.fluid._sediment.numpy()).sum() * area)
band = np.abs(Z) <= 8.0
interior_cols = np.abs(x) <= 85.0
thirds = [(x >= lo) & (x < hi) for lo, hi in ((-85.0, -28.0), (-28.0, 28.0), (28.0, 85.1))]
clamp = config.SEDIMENT_MAX_BED_CHANGE * step
print(f"Q = {Q:g} m3/s, erosion on, primed, boundary zones {'HELD' if shield else 'free'}; "
      f"every {step:g} s (clamp over an interval = {clamp:.2f} m)")
previous = np.zeros_like(start_bed)
t = step
while t <= total + 1e-9:
    while m.sim_time < t:
        m._step_once()
        if shield:
            current = bed()
            current[boundary] = start_bed[boundary]
            m.fluid._bed_terrain.assign(current.astype(np.float32).ravel())
            m.fluid._recombine_bed()
    change = bed() - start_bed
    pinned = np.abs(change - previous) >= 0.95 * clamp
    previous = change
    interior = interior_cols[None, :] & np.ones((N, 1), bool)
    parts = []
    for name, cols in zip(("upper", "middle", "lower"), thirds):
        c = change[band & cols[None, :]]
        parts.append(f"{name} {c.mean() * 100:+6.2f} cm ({c.min():+.2f}/{c.max():+.2f})")
    ci, cb = change[interior], change[boundary]
    d = m.fluid.diagnostics()
    lost = -float(change.sum() * area)
    brought = d.get("sediment_in_m3", 0.0) - d0.get("sediment_in_m3", 0.0)
    suspended = float(np.asarray(m.fluid._sediment.numpy()).sum() * area) - start_sed
    out = d.get("sediment_out_m3", 0.0) - d0.get("sediment_out_m3", 0.0)
    print(f"t={m.sim_time:5.0f}  interior channel: " + "  ".join(parts))
    print(f"        interior: cut>0.1 {int((ci < -0.1).sum()):4d} >0.5 {int((ci < -0.5).sum()):3d}  "
          f"fill>0.1 {int((ci > 0.1).sum()):4d} >0.5 {int((ci > 0.5).sum()):3d}  "
          f"at clamp {int(pinned[interior].sum()):3d}   "
          f"boundaries: deepest {cb.min():+.2f} m, at clamp {int(pinned[boundary].sum()):3d}   "
          f"odd-even {odd_even_share(change) * 100:4.1f}%")
    print(f"        ledger: bed lost {lost:8.2f} + inlet brought {brought:8.2f} = suspended "
          f"{suspended:7.2f} + out {out:7.2f} + transport gap {lost + brought - suspended - out:+8.2f} m3",
          flush=True)
    t += step
