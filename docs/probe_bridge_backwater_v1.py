"""Probe: does a bridge back the river up, measured with gauging lines?

The river plan's acceptance scene (docs/07_river_plan.md): piers narrow the
channel, so upstream the level rises, between the piers the flow speeds up,
and the discharge through a line above and below is the same. Before writing a
test, this measures it -- the same valley and inlet as SectionTests, with and
without a BRIDGE across the channel at x = 0, lines at several x.

    python docs/probe_bridge_backwater_v1.py 300
    python docs/probe_bridge_backwater_v1.py 240 5 1.2 12   # piers, radius m, Q m3/s
"""
import math
import sys

import numpy as np

sys.path.insert(0, "backend")
from app import config  # noqa: E402
from app.simulation import SimulationManager  # noqa: E402

N = config.TERRAIN_CELLS + 1
seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
PIERS = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
RADIUS = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
Q = float(sys.argv[4]) if len(sys.argv) > 4 else 12.0
LINES = (-60.0, -20.0, -6.0, 0.0, 6.0, 20.0, 60.0)


def run(bridge: bool, discharge: float = 12.0) -> dict:
    m = SimulationManager()
    m.apply_terrain_river({})
    m.apply_water_level(0.0)
    m.apply_river_inlet({"enabled": True, "width_m": 12.0, "discharge_m3s": discharge})
    m.apply_river_outlet({"width_m": 20.0})
    ids = {}
    for x in LINES:
        oid = m.apply_object_add({"type": "SECTION", "position": [x, 0.0, 0.0]})["id"]
        m.apply_object_update(oid, {"metadata": {"section_width_m": 60.0}})
        ids[x] = oid
    if bridge:
        oid = m.apply_object_add({"type": "BRIDGE", "position": [0.0, 0.0, 0.0]})["id"]
        m.apply_object_update(oid, {"metadata": {"pier_count": PIERS, "pier_radius": RADIUS}})
    m.start()
    bed = m.fluid.get_terrain_heights().reshape(N, N)
    h = np.maximum(bed[N // 2, :][None, :] + 0.7 - bed, 0.0).astype(np.float32)
    m.fluid._h.assign(h.ravel())
    for _ in range(int(seconds * 60)):
        m._step_once()
    # average the last 10 s of readings, so a sloshing mode does not decide it
    sums = {x: np.zeros(4) for x in LINES}
    for _ in range(600):
        m._step_once()
        state = {s["id"]: s["latest"] for s in m.section_state()}
        for x, oid in ids.items():
            r = state[oid]
            sums[x] += (r["flow_m3s"], r["level_m"] or 0.0, r["froude_section"],
                        r["wetted_width_m"])
    solid = int((np.asarray(m.fluid._obstacle_host) != 0).sum())
    m.stop()
    return {x: sums[x] / 600.0 for x in LINES} | {"solid": solid}


with_bridge, without = run(True, Q), run(False, Q)
print(f"t = {seconds:.0f}-{seconds + 10:.0f} s, Q = {Q:g} m3/s, {PIERS:g} piers r {RADIUS:g} m; solid cells "
      f"{without['solid']} -> {with_bridge['solid']}")
print("   x     Q no/br      level no/br   (afflux)    Fr no/br     width no/br")
for x in LINES:
    a, b = without[x], with_bridge[x]
    print(f"{x:5.0f}  {a[0]:6.2f} {b[0]:6.2f}   {a[1]:6.3f} {b[1]:6.3f} ({(b[1] - a[1]) * 100:+5.1f} cm)"
          f"   {a[2]:5.2f} {b[2]:5.2f}   {a[3]:5.1f} {b[3]:5.1f}")
