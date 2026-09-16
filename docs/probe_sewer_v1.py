"""Probe: how much does each storm inlet in the Sewer scenario actually take?

v0.17.0 WIP measured 2-3 L/s per pipe against 37-64 L/s capacity after 180 s
at 50 mm/h -- too little to see. Before changing anything, this separates the
candidate causes, one change per run:

  base       the scenario as shipped
  long       the same, run long enough to tell "still filling" from "that is all"
  d100       100 mm pipes instead of 200 mm (capacity down ~6x)
  crossfall  the ground graded toward each inlet, 2% over 12 m
  crossfall_d150  both

Each line prints per-pipe flow, capacity, and the peak depth within 3 m of each
grate, so "the grate is starved" and "the grate is overloaded" read apart.

    python docs/probe_sewer_v1.py base 600
"""
import sys

import numpy as np

sys.path.insert(0, "backend")
from app.simulation import SimulationManager  # noqa: E402

variant = sys.argv[1] if len(sys.argv) > 1 else "base"
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
rain = float(sys.argv[3]) if len(sys.argv) > 3 else None

m = SimulationManager()
m.load("scenario_sewer")
world = m.world
if rain is not None:
    world.water.rain_intensity_mm_h = rain
if "_d" in variant or variant.startswith("d"):
    # e.g. d100, crossfall_d150: pipe diameter in mm
    mm = float(variant.rsplit("d", 1)[1])
    for obj in world.objects.values():
        if obj.type == "PIPE":
            obj.metadata["diameter_m"] = mm / 1000.0
if variant.startswith("crossfall"):
    sys.path.insert(0, "tools")
    import make_scenarios as ms  # noqa: E402
    ms.grade_sewer_catchment(world)
    m.terrain_revision += 1
    m.fluid.initialize(world)
    m.fluid.set_boundaries(world.terrain, m._obstacle_snapshot,
                           m.terrain_revision, m.obstacle_revision)

m.start()
t = world.terrain
n, c = m.fluid._width, t.cell_size
coords = (np.arange(n) - (n - 1) * 0.5) * c
X, Z = np.meshgrid(coords, coords)
inlets = {o.id: o for o in world.objects.values() if o.type == "STORM_INLET"}
masks = {k: np.hypot(X - o.position[0], Z - o.position[2]) <= 3.0 for k, o in inlets.items()}
print(f"variant={variant} rain={world.water.rain_intensity_mm_h} mm/h")
for step in range(1, int(seconds * 60) + 1):
    m._step_once()
    if step % (60 * 60) == 0:
        sewer = m.sewer_state()
        h = np.asarray(m.fluid._h.numpy()).reshape(n, n)
        parts = []
        for link in sewer:
            depth = h[masks[link["inlet_id"]]].max() if link["inlet_id"] in masks else 0.0
            parts.append(f"{link['flow_m3s'] * 1000:5.1f}/{link['capacity_m3s'] * 1000:4.1f} L/s "
                         f"pit {depth * 100:4.1f} cm")
        print(f"t={step / 60:4.0f}s  " + "  |  ".join(parts), flush=True)
m.stop()
