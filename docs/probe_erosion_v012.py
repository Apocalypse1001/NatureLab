"""Diagnostic probe: is river erosion limited by the flow or by the valve?

Blocking measurement for v0.12.0, specified in docs/07_river_plan.md, item 1.
The v0.11.0 datum was taken on a *flat* map, where the only motion is the
inflow spreading out; a river runs on a slope, where Manning sets the speed and
the capacity law reads that speed directly. So the question is re-asked on the
geometry the river will actually have.

Read-only: imports the real solver from backend/app, changes nothing on disk.
Run from the repo root:   python docs/probe_erosion_v012.py

What is compared, and why in this form: the kernel writes

    demand = gap * SEDIMENT_ERODE_RATE * dt        (m of bed, this substep)
    limit  = SEDIMENT_MAX_BED_CHANGE * dt          (m of bed, this substep)

Both carry dt, so the honest comparison is rate against rate --
`gap * ERODE_RATE` (m/s) against `SEDIMENT_MAX_BED_CHANGE` (m/s). Comparing a
per-substep amount against a per-second rate would manufacture a factor of 60
out of the units alone.

And it is measured as a distribution over cells and over time rather than as one
number, because this is a relaxation system: gap = capacity - sediment, so once
the suspended load reaches capacity the flow stops being hungry and erosion
self-arrests whatever the clamp says. "How far over the clamp" only means
something together with "in how many cells, where, and for how long".
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import config                              # noqa: E402
from app.compute_engine import create_engine        # noqa: E402
from app.fluid_solver import create_fluid_solver    # noqa: E402
from app.world_state import WorldState              # noqa: E402

N = config.TERRAIN_CELLS + 1
CELL = config.TERRAIN_CELL_SIZE
DT = config.FIXED_DT
CLAMP = config.SEDIMENT_MAX_BED_CHANGE
SCALE = config.SEDIMENT_CAPACITY_SCALE
ERODE = config.SEDIMENT_ERODE_RATE


def build(slope, depth, edge_source=False, channel=False):
    """Uniform sheet (or trapezoidal channel) let go on a constant bed slope.

    The sheet case is FrictionLawTests._terminal_speed_on_slope with erosion
    switched on, so the velocities it reaches are already checked against the
    Manning value elsewhere in the suite.
    """
    world = WorldState()
    world.water.level = depth
    fall = np.arange(N, dtype=np.float32) * CELL * slope
    bed = np.tile((fall[-1] - fall)[None, :], (N, 1))
    if channel:
        # 12 m flat bottom, 1.2 m incision, 3 m banks -- the geometry the
        # generator in item 2 will produce, so the answer transfers.
        centre = N // 2
        rows = np.arange(N)[:, None]
        dist = np.abs(rows - centre).astype(np.float32)
        cut = np.clip((6.0 + 3.0 - dist) / 3.0, 0.0, 1.0) * 1.2
        bed = bed - cut
    world.terrain.heights[:, :] = bed
    solver = create_fluid_solver(create_engine().device)
    solver.initialize(world)
    solver._source_enabled = edge_source
    solver.set_boundaries(world.terrain, {}, 0, 0)
    solver.set_outflow(config.FLUID_OUTFLOW_COLUMNS)
    solver.set_erosion(True)
    if channel:
        # fill the channel to `depth` over its bottom, dry banks
        surface = bed[N // 2, :] + depth
        h = np.maximum(surface[None, :] - bed, 0.0).astype(np.float32)
    else:
        h = np.full((N, N), depth, dtype=np.float32)
    solver._h.assign(h.ravel())
    return world, solver


def sample(solver):
    h = np.asarray(solver._h.numpy()).reshape(N, N)
    u = np.asarray(solver._u.numpy()).reshape(N, N)
    v = np.asarray(solver._v.numpy()).reshape(N, N)
    sed = np.asarray(solver._sediment.numpy()).reshape(N, N)
    speed = np.hypot(u, v)
    capacity = SCALE * speed * h
    gap = capacity - sed
    wet = h > 0.05
    hungry = wet & (gap > 0.0)
    demand = gap * ERODE                      # m/s, same units as the clamp
    return h, speed, capacity, sed, demand, wet, hungry


def run(label, slope, depth, edge_source=False, channel=False, seconds=20.0):
    world, solver = build(slope, depth, edge_source, channel)
    bed0 = solver.get_terrain_heights().reshape(N, N).copy()
    steps = int(seconds * 60)
    interior = slice(4, N - 4)                # off the inlet strip and the outlet
    trace = []
    for step in range(steps):
        solver.advance(DT, config.FLUID_MAX_SUBSTEPS, config.FLUID_STABILITY_DT)
        if step % 60 != 59:
            continue
        h, speed, capacity, sed, demand, wet, hungry = sample(solver)
        mean_u = float(speed[wet].mean()) if wet.any() else 0.0
        if not hungry.any():
            trace.append((solver._time, 0.0, 0.0, 0.0, mean_u))
            continue
        binding = demand > CLAMP
        inner = np.zeros_like(binding)
        inner[:, interior] = binding[:, interior]
        ratio = demand[hungry] / CLAMP
        sat = np.divide(sed, capacity, out=np.zeros_like(sed), where=capacity > 1e-9)
        trace.append((solver._time,
                      float(inner[hungry].mean()),
                      float(np.percentile(demand[hungry] / CLAMP, 99)),
                      float(sat[wet].mean()) if wet.any() else 0.0,
                      mean_u))
    h, speed, capacity, sed, demand, wet, hungry = sample(solver)
    bed1 = solver.get_terrain_heights().reshape(N, N)
    loss = bed0 - bed1
    diag = solver.diagnostics()
    src = slice(0, config.FLUID_SOURCE_COLUMNS + 2)

    print("")
    print("=== %s ===" % label)
    print("  slope=%.2f%%  depth=%.2f m  edge_source=%s  channel=%s  t=%.1f s"
          % (slope * 100, depth, edge_source, channel, solver._time))
    print("  substeps=%d  cfl_limited=%s  max|u|=%.2f m/s  mean|u| wet=%.2f m/s"
          % (diag["substeps"], diag["cfl_limited"], diag["max_velocity"],
             float(speed[wet].mean()) if wet.any() else 0.0))
    if hungry.any():
        ratio = demand[hungry] / CLAMP
        print("  demand/clamp over hungry wet cells: median=%.2f  p90=%.2f  max=%.2f"
              % (np.median(ratio), np.percentile(ratio, 90), ratio.max()))
        binding = demand > CLAMP
        print("  cells at the clamp: %.1f%% of hungry cells, %d cells"
              % (float(binding[hungry].mean()) * 100, int(binding.sum())))
        print("    of those, in the inlet strip (i<%d): %d;  interior: %d"
              % (src.stop, int(binding[:, src].sum()), int(binding[:, interior].sum())))
    else:
        print("  no hungry cells left: the flow is carrying its capacity")
    sat = np.divide(sed, capacity, out=np.zeros_like(sed), where=capacity > 1e-9)
    if wet.any():
        print("  saturation sed/capacity over wet cells: mean=%.2f  median=%.2f"
              % (float(sat[wet].mean()), float(np.median(sat[wet]))))
    ceiling = CLAMP * solver._time
    print("  bed change: max cut=%.4f m of a %.3f m ceiling (%.1f%%)  interior max=%.4f m"
          % (float(loss.max()), ceiling, float(loss.max()) / ceiling * 100,
             float(loss[:, interior].max())))
    print("   t(s)  interior binding%  p99(demand/clamp)  saturation  mean|u|")
    for t, frac, med, satm, spd in trace:
        print("  %5.1f  %7.1f  %19.2f  %10.2f  %6.2f" % (t, frac * 100, med, satm, spd))
    solver.reset()
    return float(loss.max()), diag


def run_river(label, seconds=180.0, discharge=12.0, slope=0.002):
    """The same question, asked of the finished v0.12.0 river.

    The 120 s runaway further up was measured with the edge-level source feeding
    clean water into a sheet -- which is exactly the artifact that was then
    fixed. So the number has to be taken again on the configuration that ships:
    generated valley, local discharge inlet, local outlet, erosion on.
    """
    from app.simulation import SimulationManager    # noqa: E402  (probe-only)

    manager = SimulationManager()
    info = manager.apply_terrain_river({"slope": slope})["river"]
    manager.apply_water_level(0.0)
    manager.apply_river_inlet({"enabled": True, "width_m": 12.0,
                               "discharge_m3s": discharge})
    manager.apply_river_outlet({"width_m": 20.0})
    manager.apply_water_erosion(True)
    manager.start()
    bed0 = manager.fluid.get_terrain_heights().reshape(N, N).copy()
    print("")
    print("=== %s ===" % label)
    print("  Q=%.1f m3/s  slope=%.2f%%  incision=%.1f m" %
          (discharge, slope * 100, info["incision"]))
    print("   t(s)   Qin   Qout  binding%  max|u|  max cut  substeps cfl_limited")
    step = 0
    while manager.sim_time < seconds:
        manager._step_once()
        step += 1
        if step % 1800:
            continue
        d = manager.fluid.diagnostics()
        h = np.asarray(manager.fluid._h.numpy()).reshape(N, N)
        u = np.asarray(manager.fluid._u.numpy()).reshape(N, N)
        v = np.asarray(manager.fluid._v.numpy()).reshape(N, N)
        sed = np.asarray(manager.fluid._sediment.numpy()).reshape(N, N)
        speed = np.hypot(u, v)
        wet = h > 0.05
        hungry = wet & (SCALE * speed * h - sed > 0.0)
        demand = (SCALE * speed * h - sed) * ERODE
        binding = float((demand[hungry] > CLAMP).mean() * 100) if hungry.any() else 0.0
        cut = float((bed0 - manager.fluid.get_terrain_heights().reshape(N, N)).max())
        print("  %5.1f %6.2f %6.2f  %7.2f  %6.2f  %7.3f  %6d   %s" %
              (manager.sim_time, d["inlet_discharge_m3s"],
               d["removed_m3"] / max(manager.sim_time, 1e-9), binding,
               d["max_velocity"], cut, d["substeps"], d["cfl_limited"]))
    manager.stop()


if __name__ == "__main__":
    print("config: SEDIMENT_CAPACITY_SCALE=%s  ERODE_RATE=%s  MAX_BED_CHANGE=%s m/s  "
          "MANNING_N=%s" % (SCALE, ERODE, CLAMP, config.FLUID_MANNING_N))
    # Manning's own prediction for the sheet cases, as a check on the probe:
    for s in (0.002, 0.005):
        for d in (0.5, 1.5):
            u = d ** (2.0 / 3.0) * np.sqrt(s) / config.FLUID_MANNING_N
            print("  hand check  S=%.1f%%  h=%.1f m -> u~%.2f m/s  capacity~%.4f m  "
                  "demand~%.4f m/s = %.1fx clamp"
                  % (s * 100, d, u, SCALE * u * d, SCALE * u * d * ERODE,
                     SCALE * u * d * ERODE / CLAMP))

    run("sheet, gentle slope, shallow", 0.002, 0.5)
    run("sheet, gentle slope, deep", 0.002, 1.5)
    run("sheet, river slope, mid", 0.005, 1.0)
    run("channel, river slope, mid", 0.005, 1.0, channel=True)
    run("channel + edge inflow (clean water at the inlet)", 0.005, 1.0,
        channel=True, edge_source=True)
