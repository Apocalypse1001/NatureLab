"""Probe: a flood wave down the valley, read by gauging lines.

The same valley and inlet as SectionTests (base Q 12 m3/s, primed channel),
with a hydrograph on the inlet: peak, start, rise and fall in simulation
seconds. Lines at the inlet face, x = -50 and x = +80. Reports the flood volume
through the inlet line against (peak - base)(rise + fall)/2, and at each line
the peak discharge, its time and the peak level.

    python docs/probe_hydrograph_v1.py 30 20 30 60 220
"""
import sys

import numpy as np

sys.path.insert(0, "backend")
from app import config, hydrograph  # noqa: E402
from app.simulation import SimulationManager  # noqa: E402

N = config.TERRAIN_CELLS + 1
peak, start, rise, fall, total = (float(v) for v in (sys.argv[1:6] or (30, 20, 30, 60, 220)))
BASE = 12.0

m = SimulationManager()
m.apply_terrain_river({})
m.apply_water_level(0.0)
m.apply_river_inlet({"enabled": True, "width_m": 12.0, "discharge_m3s": BASE,
                     "hydrograph": {"enabled": True, "peak_m3s": peak, "start_s": start,
                                    "rise_s": rise, "fall_s": fall}})
m.apply_river_outlet({"width_m": 20.0})
cell = m.world.terrain.cell_size
west = -(N - 1) * 0.5 * cell
lines = {}
for name, x in (("inlet", west + 0.5 * cell), ("x-50", -50.0), ("x+80", 80.0)):
    oid = m.apply_object_add({"type": "SECTION", "position": [x, 0.0, 0.0]})["id"]
    m.apply_object_update(oid, {"metadata": {"section_width_m": 60.0}})
    lines[name] = oid
m.start()
bed = m.fluid.get_terrain_heights().reshape(N, N)
m.fluid._h.assign(np.maximum(bed[N // 2, :][None, :] + 0.7 - bed, 0.0).astype(np.float32).ravel())
series = {name: [] for name in lines}
for k in range(int(total * 60)):
    m._step_once()
    state = {s["id"]: s["latest"] for s in m.section_state()}
    for name, oid in lines.items():
        r = state[oid]
        series[name].append((m.sim_time, r["flow_m3s"], r["level_m"] or 0.0, r["volume_m3"]))
expected = hydrograph.flood_volume_m3(BASE, peak, rise, fall)
inlet = np.array(series["inlet"])
# from 5 s before the flood: the first seconds after priming the inlet's edge
# cell runs short and would be booked against the flood
k0 = int(np.searchsorted(inlet[:, 0], start - 5.0))
flood = (inlet[-1, 3] - inlet[k0, 3]) - BASE * (inlet[-1, 0] - inlet[k0, 0])
print(f"peak {peak:g}, start {start:g}, rise {rise:g}, fall {fall:g}; flood volume through the "
      f"inlet line {flood:.1f} m3, expected {expected:.1f} ({(flood / expected - 1) * 100:+.2f}%)")
for name, rows in series.items():
    a = np.array(rows)
    k = int(np.argmax(a[:, 1]))
    j = int(np.argmax(a[:, 2]))
    before = a[int(np.searchsorted(a[:, 0], start)), 1]
    print(f"{name:6s} Q before {before:6.2f}  peak Q {a[k, 1]:6.2f} (+{a[k, 1] - before:5.2f}) "
          f"at {a[k, 0]:6.1f} s   peak level {a[j, 2]:.3f} m at {a[j, 0]:6.1f} s   Q at end {a[-1, 1]:.2f}")
