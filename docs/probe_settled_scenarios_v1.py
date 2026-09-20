"""Probe: does a settled scenario still open in equilibrium after the friction change?

v0.18.1 shipped River, Bridge and Sewer with `water.initial_flow` -- a depth and
face-velocity field settled by `apply_water_settle` and saved with the world, so
the scene opens on a river already flowing instead of a dry bed filling up.

That water was settled under `FLUID_FRICTION_MIN_DEPTH = 0.02`. v0.19.0 retires
the floor to 0.0005 (docs/17_film_friction_plan.md), so the stored field is now
the equilibrium of a solver that no longer exists: the promise "starts from
equilibrium" is only as true as the drift measured here.

Measured 2026-09-20, and the answer was that it held: re-settling under the new
floor moved the stored depth by at most 0.12 mm, and the RELEASED field -- the
one settled under the old floor -- already read 11.92-12.00 m3/s at five lines
from the first second. The channel is ~0.7 m deep and never came near the floor.

**Read the Sewer numbers with its rain in mind.** That scene ships raining at
50 mm/h, which is 0.561 m3/s over a 201 x 201 m map, so its volume climbs about
1.7% a minute and 4.7% over three minutes. That is the storm arriving, not the
scene re-settling: `apply_water_settle` runs with rain off, so what is stored is
the dry-weather equilibrium and PLAY is supposed to start the rain. Compare the
climb against the rain rate before calling anything drift.

What it reports, from the first tick after PLAY:

  * discharge at five lines across the channel (River and Bridge). A settled
    river reads its base 12 m3/s at every line from the first second; a scene
    that has to re-settle shows a front travelling down the reach instead.
  * the water on the map, as a volume and as the largest single-cell change
    since load. Drift here is the scene rearranging itself while someone
    watches, which is exactly what settling was meant to remove.

    python docs/probe_settled_scenarios_v1.py river 60
    python docs/probe_settled_scenarios_v1.py sewer 60
"""
import sys

import numpy as np

sys.path.insert(0, "backend")
from app import config  # noqa: E402
from app.simulation import SimulationManager  # noqa: E402

# Across the channel, which runs along z = 0: upstream of the town, through it,
# and down to the outlet. 60 m spans the trapezoid and both banks, so a line
# catches the whole flow rather than the middle of it.
LINE_X = (-80.0, -40.0, 0.0, 40.0, 80.0)
LINE_WIDTH_M = 60.0
SAMPLE_AT_S = (1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 180.0)
CHANNEL_SCENES = {"river", "bridge"}

scene = sys.argv[1] if len(sys.argv) > 1 else "river"
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0

manager = SimulationManager()
manager.load(f"scenario_{scene}")
water = manager.world.water
print(f"scenario_{scene}: initial_flow {'present' if water.initial_flow else 'ABSENT'}, "
      f"inlet {'on' if water.inlet_enabled else 'off'} at "
      f"{water.inlet_discharge_m3s:g} m3/s, friction floor "
      f"{config.FLUID_FRICTION_MIN_DEPTH:g} m")
# Printed next to the drift because it is the first thing the drift has to be
# judged against: a raining scene is meant to gain water.
rain_m3s = (water.rain_intensity_mm_h / 1000.0 / 3600.0
            * len(np.asarray(manager.world.terrain.heights).ravel())
            * float(manager.world.terrain.cell_size) ** 2)
print(f"  rain {water.rain_intensity_mm_h:g} mm/h = {rain_m3s:.3f} m3/s onto the map")

lines = {}
if scene in CHANNEL_SCENES:
    for x in LINE_X:
        oid = manager.apply_object_add({"type": "SECTION", "position": [x, 0.0, 0.0]})["id"]
        manager.apply_object_update(oid, {"metadata": {"section_width_m": LINE_WIDTH_M}})
        lines[x] = oid

manager.start()
cell_area = float(manager.world.terrain.cell_size) ** 2


def depth() -> np.ndarray:
    return np.asarray(manager.fluid._h.numpy(), dtype=np.float64)


# The field as the scene opens: PLAY has restored initial_flow but nothing has
# been stepped yet, so this is what the saved world promised.
opening = depth()
print(f"on load: {opening.sum() * cell_area:8.1f} m3 of water, "
      f"{int((opening > config.FLUID_DRY_DEPTH).sum())} wet cells")

header = "     t      volume     drift   max|dh|"
if lines:
    header += "".join(f"{x:>10.0f}" for x in LINE_X)
print(header)

samples = [s for s in SAMPLE_AT_S if s <= seconds] or [seconds]
next_sample = 0
for _ in range(int(round(seconds / config.FIXED_DT))):
    manager._step_once()
    if next_sample < len(samples) and manager.sim_time + 1e-9 >= samples[next_sample]:
        now = depth()
        volume = now.sum() * cell_area
        row = (f"{manager.sim_time:6.1f}  {volume:9.1f}  "
               f"{100.0 * (volume - opening.sum() * cell_area) / max(opening.sum() * cell_area, 1e-9):+7.2f}%  "
               f"{np.abs(now - opening).max():7.4f}")
        if lines:
            state = {s["id"]: s["latest"] for s in manager.section_state()}
            for x in LINE_X:
                latest = state.get(lines[x])
                row += (f"{latest['flow_m3s']:10.2f}" if latest else f"{'--':>10}")
        print(row)
        next_sample += 1

manager.stop()
