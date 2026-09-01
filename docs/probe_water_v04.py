"""Diagnostic probe for the v0.4 water model.

Read-only: imports the real solver from backend/app, changes nothing on disk.
Run from the repo root:   python docs/probe_water_v04.py
Referenced by docs/05_audit_v0.4_water.md.
"""
import sys
import numpy as np

sys.path.insert(0, "backend")
from app.world_state import WorldState          # noqa: E402
from app.fluid_solver import ShallowWaterFluidSolver  # noqa: E402
from app import config                          # noqa: E402

CELL = config.TERRAIN_CELL_SIZE
N = config.TERRAIN_CELLS + 1


def make(level=1.0, flow=False, terrain=None):
    w = WorldState()
    if terrain is not None:
        w.terrain.heights = terrain.astype(np.float32)
    w.water.level = level
    w.water.flow_enabled = flow
    s = ShallowWaterFluidSolver()
    s.initialize(w)
    s.set_river_flow(flow)
    return w, s


def step(s, seconds):
    for _ in range(int(round(seconds * 60))):
        s.advance(config.FIXED_DT, config.FLUID_MAX_SUBSTEPS, config.FLUID_STABILITY_DT)


def stats(s):
    d = s.get_depth_grid()
    flux = np.sqrt(s._flow_x ** 2 + s._flow_z ** 2)
    vel = np.where(d > 0.01, flux * CELL / np.maximum(d, 1e-6), 0.0)
    return dict(total=round(float(d.sum()), 1), wet=int((d > 1e-3).sum()),
                max_flux=round(float(flux.max()), 4),
                max_true_vel_mps=round(float(vel.max()), 3))


def slope(grade):
    xs = np.arange(N, dtype=np.float32)
    return np.tile(grade * (N - 1 - xs), (N, 1))


def timeline(s, marks):
    prev, out = 0.0, []
    for t in marks:
        step(s, t - prev)
        prev = t
        out.append((t, stats(s)))
    return out


print("== A. LAKE MODE (flow off) -- does water ever move? ==")
for name, terr in (("flat", None), ("2% slope", slope(0.02)), ("6% slope", slope(0.06))):
    _, s = make(level=1.0, flow=False, terrain=terr)
    row = timeline(s, [1.0, 10.0, 30.0])
    print(f"  {name:9s} max_flux at t=1/10/30s: "
          + "  ".join(str(st['max_flux']) for _, st in row))

print("\n== B. RIVER MODE (flow on) -- sustained current, or a decaying transient? ==")
_, s = make(level=1.0, flow=True)
for t, st in timeline(s, [1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]):
    print(f"  t={int(t):4d}s  {st}")

print("\n== C. Water Level slider: is it an elevation or a depth? ==")
w, s = make(level=1.0, flow=True, terrain=slope(0.02))
step(s, 2)
d = s.get_depth_grid()
print("  user set Water Level = 1.0 m")
print(f"  west-edge terrain = {float(w.terrain.heights[50, 0]):.2f} m, "
      f"depth = {float(d[50, 0]):.2f} m "
      f"-> water surface = {float(w.terrain.heights[50, 0] + d[50, 0]):.2f} m")

print("\n== D. drag force on bodies is linear in FLUID_FLOW_GAIN ==")
print("  sample_for_bodies() returns velocities = flow_x = (flow_right - flow_left),")
print("  and flow_* = (delta h) * FLUID_FLOW_GAIN, so drag scales with the constant.")
print(f"  gain 1.6 -> 10.0 (commit a6f53b3) = {10.0 / 1.6:.2f}x on every hydrodynamic force;")
print(f"  RIGID_WATER_DRAG_SCALE = {config.RIGID_WATER_DRAG_SCALE} unchanged since 3c47a7c,")
print("  where it was tuned against gain 1.6.")

print("\n== E. Does the source respect terrain height at the west edge? ==")
terr = np.zeros((N, N), dtype=np.float32)
terr[:50, 0:3] = 10.0                      # north half of the inlet is a 10 m plateau
_, s = make(level=1.0, flow=True, terrain=terr)
step(s, 3)
d = s.get_depth_grid()
print(f"  inlet cell ON the 10 m plateau (row 10, x=0): terrain=10.00 m  depth={float(d[10, 0]):.3f} m")
print(f"  inlet cell on flat ground      (row 90, x=0): terrain= 0.00 m  depth={float(d[90, 0]):.3f} m")

print("\n== F. Constriction: does narrowing the channel speed the water up? ==")
mid = N // 2
for label, half in (("wide  (36 cells)", 18), ("narrow (18 cells)", 9)):
    terr = np.full((N, N), 5.0, dtype=np.float32)
    terr[mid - 18:mid + 18, :] = 0.0
    terr[:, 0:2] = 0.0                     # full-width inlet, so E above cannot contaminate this
    terr[mid - 18:mid + 18, 40:60] = 5.0
    terr[mid - half:mid + half, 40:60] = 0.0
    _, s = make(level=1.0, flow=True, terrain=terr)
    step(s, 60)
    d, fx = s.get_depth_grid(), s._flow_x
    wet = (d[:, 50] > 0.01) & (terr[:, 50] < 1.0)
    if not wet.any():
        print(f"  {label} x=50: DRY after 60 s")
        continue
    flux = float(np.abs(fx[wet, 50]).mean())
    dep = float(d[wet, 50].mean())
    print(f"  {label} x=50: wet={int(wet.sum()):2d}  mean_depth={dep:.3f} m  "
          f"reported_flux={flux:.4f}  true_vel={flux * CELL / dep:.3f} m/s  "
          f"discharge={flux * CELL * CELL * int(wet.sum()):.3f} m3/s")
