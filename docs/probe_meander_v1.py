"""Probe: does a meandering channel carry the river the straight one does?

Meanders are generated (terrain_gen.river_valley, meander_amplitude) but have been
off by default. This runs the shipped path twice -- straight, then winding --
with the same inlet, and reads gauging lines across the whole valley at five
stations: discharge, water width, section Froude number; plus how many cells
OUTSIDE the channel are wet (the river leaving its bed at a bend) and the
deepest water anywhere.

The channel is primed per column (its own lowest bed + 0.7 m), not along the
map's centre row: a winding channel is not on that row.

    python docs/probe_meander_v1.py 300 20 120
"""
import sys

import numpy as np

sys.path.insert(0, "backend")
from app import config  # noqa: E402
from app.simulation import SimulationManager  # noqa: E402

N = config.TERRAIN_CELLS + 1
total = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
amp = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
wave = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
STATIONS = (-80.0, -40.0, 0.0, 40.0, 80.0)


def run(amplitude: float) -> dict:
    m = SimulationManager()
    info = m.apply_terrain_river({"meander_amplitude": amplitude, "meander_wavelength": wave})["river"]
    m.apply_water_level(0.0)
    m.apply_river_inlet({"enabled": True, "width_m": 12.0, "discharge_m3s": 12.0})
    m.apply_river_outlet({"width_m": 20.0, "kind": "river"})
    ids = {}
    for x in STATIONS:
        oid = m.apply_object_add({"type": "SECTION", "position": [x, 0.0, 0.0]})["id"]
        m.apply_object_update(oid, {"metadata": {"section_width_m": 120.0}})
        ids[x] = oid
    m.start()
    bed = m.fluid.get_terrain_heights().reshape(N, N)
    surface = bed.min(axis=0) + 0.7
    m.fluid._h.assign(np.maximum(surface[None, :] - bed, 0.0).astype(np.float32).ravel())
    for _ in range(int(total * 60)):
        m._step_once()
    sums = {x: np.zeros(3) for x in STATIONS}
    for _ in range(300):
        m._step_once()
        state = {s["id"]: s["latest"] for s in m.section_state()}
        for x, oid in ids.items():
            r = state[oid]
            sums[x] += (r["flow_m3s"], r["wetted_width_m"], r["froude_section"])
    h = np.asarray(m.fluid._h.numpy()).reshape(N, N)
    # outside the channel: more than half a bed width plus the bank from the centreline
    cell = m.world.terrain.cell_size
    xs = np.arange(N) * cell
    zs = (np.arange(N) - (N - 1) * 0.5) * cell
    centre = amplitude * np.sin(2.0 * np.pi * xs / wave) if amplitude > 0 else np.zeros(N)
    outside = np.abs(zs[:, None] - centre[None, :]) > info["bed_width"] * 0.5 + info["bank_run"]
    wet_out = int(((h > 0.01) & outside).sum())
    d = m.fluid.diagnostics()
    m.stop()
    return {"lines": {x: v / 300.0 for x, v in sums.items()}, "wet_outside": wet_out,
            "max_depth": float(h.max()), "outlet_z": info["outlet_centre_z"],
            "volume_error": d.get("volume_error_m3", 0.0)}


straight, winding = run(0.0), run(amp)
print(f"t = {total:.0f}-{total + 5:.0f} s, Q = 12; meander amplitude {amp:g} m, wavelength {wave:g} m, "
      f"outlet band at z = {winding['outlet_z']:+.1f}")
print("    x   Q straight/winding   width s/w     Fr s/w")
for x in STATIONS:
    a, b = straight["lines"][x], winding["lines"][x]
    print(f"{x:5.0f}   {a[0]:6.2f} {b[0]:6.2f}      {a[1]:5.1f} {b[1]:5.1f}   {a[2]:5.2f} {b[2]:5.2f}")
for name, r in (("straight", straight), ("winding", winding)):
    print(f"{name:8s} wet cells outside the channel {r['wet_outside']:5d}   deepest {r['max_depth']:.2f} m   "
          f"volume error {r['volume_error']:+.2f} m3")
