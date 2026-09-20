"""Probe: what holds the sheet of water that has to reach a storm grate?

Measured 2026-09-19 on the shipped Sewer scenario: in the graded ring 3-12 m
around each grate -- the street that must CARRY the rain to the inlet -- the
median depth is 0.4-0.6 mm and 100% of the wet cells are thinner than 2 cm.
Every one of them therefore runs inside `FLUID_FRICTION_MIN_DEPTH = 0.02`, so
the drag on that film is computed at 2 cm instead of half a millimetre:
(0.02/0.0005)^(4/3) ~ 137x too weak. The headline number of the sewer work is
a TIME -- when does the drain overflow -- and right now that time is set by a
constant rather than by the depth.

Two candidate changes pull in opposite directions, so they are measured one at
a time and never together until each is known alone:

  base        as shipped: floor 0.02 m, Manning 0.03 everywhere
  floor<x>    the friction floor alone, e.g. floor0.002
  road        the roughness map alone: ROAD footprints get pavement n
  both<x>     floor <x> and the road map together

Lowering the floor BRAKES the film (real drag at 0.5 mm is far stronger than
the clamped drag). Giving the asphalt its own n ~ 0.013 against the 0.03 of a
"clean natural channel" SPEEDS it. Only the pair of runs says which wins.

The drag is semi-implicit -- u /= (1 + g n^2 |u| dt / hf^(4/3)) -- so the floor
is not a stability guard and can be lowered without the step blowing up. That
is what `config.py` means by calling it a physics knob.

    python docs/probe_film_friction_v1.py base 420
    python docs/probe_film_friction_v1.py floor0.002 420
"""
import sys

import numpy as np

sys.path.insert(0, "backend")
from app import config  # noqa: E402
from app.simulation import SimulationManager  # noqa: E402

PAVEMENT_N = 0.013      # asphalt in good repair; Chow's tables give 0.012-0.016
variant = sys.argv[1] if len(sys.argv) > 1 else "base"
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 420.0

floor = config.FLUID_FRICTION_MIN_DEPTH
if variant.startswith("floor") or variant.startswith("both"):
    floor = float(variant.lstrip("floorbth"))
    # The kernel reads this value from config at every launch, so patching it
    # here drives the shipped path rather than a stand-in.
    config.FLUID_FRICTION_MIN_DEPTH = floor

if variant == "noroad":
    # The solver now builds the roughness map itself from ROAD footprints, so
    # the control is the run WITHOUT it, not the run with it.
    from app import fluid_solver  # noqa: E402
    fluid_solver.SURFACE_MANNING_TYPES = {}

m = SimulationManager()
m.load("scenario_sewer")
world = m.world
m.start()

t = world.terrain
n, c = m.fluid._width, t.cell_size
coords = (np.arange(n) - (n - 1) * 0.5) * c
X, Z = np.meshgrid(coords, coords)

road_cells = 0
if variant.startswith("road") or variant.startswith("both"):
    # Same footprints the rigid system reports to the solver (half extents
    # 6.0 x 3.5 for a ROAD), rasterised in the rotated frame of each slab.
    mask = np.zeros((n, n), dtype=bool)
    for obj in world.objects.values():
        if obj.type != "ROAD":
            continue
        yaw = float(obj.rotation[1]) if len(obj.rotation) > 1 else 0.0
        dx, dz = X - obj.position[0], Z - obj.position[2]
        lx = dx * np.cos(-yaw) - dz * np.sin(-yaw)
        lz = dx * np.sin(-yaw) + dz * np.cos(-yaw)
        mask |= (np.abs(lx) <= 6.0) & (np.abs(lz) <= 3.5)
    road_cells = int(mask.sum())
    field = np.full((n, n), config.FLUID_MANNING_N, dtype=np.float32)
    field[mask] = PAVEMENT_N
    m.fluid._manning.assign(field.ravel())

inlets = [o for o in world.objects.values() if o.type == "STORM_INLET"]
dist = np.full((n, n), 1e9)
for o in inlets:
    dist = np.minimum(dist, np.hypot(X - o.position[0], Z - o.position[2]))
street = (dist > 3.0) & (dist <= 12.0)
near_grate = dist <= 3.0
# A ring one cell thick at the ring's MEDIAN radius -- half the street cells lie
# inside it. Comparing a median depth against an analytic depth is only fair at
# the median cell, and measuring the discharge that actually crosses this circle
# removes the one unknown in the analytic sum: how much ground outside R = 12 m
# also drains through here.
R_MID = ((3.0 ** 2 + 12.0 ** 2) / 2.0) ** 0.5
annulus = (dist > R_MID - 0.5 * c) & (dist <= R_MID + 0.5 * c)
nearest = np.zeros((n, n, 2))
for o in inlets:
    d = np.hypot(X - o.position[0], Z - o.position[2])
    closer = d < np.hypot(X - nearest[:, :, 0], Z - nearest[:, :, 1]) if nearest.any() else np.ones((n, n), bool)
    nearest[closer] = [o.position[0], o.position[2]]
# unit vector pointing INWARD, toward the grate each cell drains to
rx, rz = nearest[:, :, 0] - X, nearest[:, :, 1] - Z
rlen = np.maximum(np.hypot(rx, rz), 1e-9)
rx, rz = rx / rlen, rz / rlen

print(f"variant={variant}  floor={floor * 1000:.1f} mm  road_cells={road_cells}  "
      f"rain={world.water.rain_intensity_mm_h} mm/h")
print("    t | street p50  p90 |  pit  |   q_meas  h(q) | pipes L/s")
arrival = {}
for step in range(1, int(seconds * 60) + 1):
    m._step_once()
    if step % 600:      # every 10 s
        continue
    now = step / 60.0
    h = np.asarray(m.fluid._h.numpy()).reshape(n, n)
    pit = float(h[near_grate].max())
    flows = [link["flow_m3s"] * 1000.0 for link in m.sewer_state()]
    # "The water has arrived" is the grate taking, not the street being wet.
    for level, key in ((0.01, "pit 1 cm"), (0.05, "pit 5 cm")):
        if key not in arrival and pit >= level:
            arrival[key] = now
    if "pipes 5 L/s" not in arrival and flows and max(flows) >= 5.0:
        arrival["pipes 5 L/s"] = now
    if step % 3600 == 0:      # print every 60 s
        wet = h[street][h[street] > 1e-4]
        p50, p90 = (np.percentile(wet, (50, 90)) if wet.size else (0.0, 0.0))
        uc = np.asarray(m.fluid._uc.numpy()).reshape(n, n)
        vc = np.asarray(m.fluid._vc.numpy()).reshape(n, n)
        # q per unit width crossing the median circle, inward-positive
        q = float(np.mean((h * (uc * rx + vc * rz))[annulus]))
        slope = 0.25 / 12.0
        ha = (max(q, 0.0) * config.FLUID_MANNING_N / slope ** 0.5) ** 0.6
        print(f"{now:5.0f} | {p50 * 1000:9.2f} {p90 * 1000:4.1f} mm | "
              f"{pit * 100:4.1f} cm | {q:9.2e} {ha * 1000:5.2f} mm | "
              + " ".join(f"{f:5.1f}" for f in flows), flush=True)
m.stop()
print("arrival: " + ("  ".join(f"{k} at {v:.0f} s" for k, v in arrival.items())
                     or "none of the marks reached"))
