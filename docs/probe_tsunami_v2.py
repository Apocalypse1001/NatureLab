"""Diagnostic probe and calibration record: a coast that actually withdraws.

Answer, measured below: not in a 200 m world -- and the fix is not a constant.
This probe both diagnosed that and calibrated the kilometre-scale coast that
replaced it (sea out 62 m, then 258 m of flood inland).

The v1 probe (`probe_tsunami_v1.py`) asked "does depth at a fixed point go down
and then up". It does, and that answer was accepted -- but it is a much weaker
question than the one the scenario actually claims to answer, which is the one
a person watching asks: **does the sea visibly leave the beach, and does the
wave then flood the land?** Measured as waterline movement rather than depth at
a point, what v1 shipped does neither:

    sea retreated 1.4 m, flooded 4.6 m inland, on a beach sloping 1:2.3 (24 deg)

Two things are wrong with v1, and only one of them is a constant.

1. **The beach is a cliff.** 24 degrees. Real tsunami-vulnerable coast is 1:50
   to 1:1000. On a 1:2.3 face a 1.2 m drawdown moves the waterline 2.7 m, so
   the drawback is invisible no matter how the wave is tuned.

2. **The wave is far too short, and short enough to be outside the solver's own
   validity.** Shallow-water equations assume wavelength >> depth. v1 seeds a
   pulse of half-width 15 m into water 15 m deep: wavelength/depth ~ 4, where
   the theory needs that ratio to be large. A tsunami's defining property is
   the opposite extreme -- wavelength of 100+ km, period of 10-60 minutes --
   and THAT is why the real sea withdraws for many minutes and goes far out.
   A 15 m pulse draws the water down for about two seconds.

   Sweeping the half-width confirms it is the dominant term, not the amplitude:
   at half-width 15 m the retreat is 0.1-1.4 m across every geometry tried; at
   50 m it reaches 8-18 m.

The second point is fatal to v1's whole approach, not just its numbers. A wave
long enough to draw the sea back does not FIT in a 200 m box together with the
shore and the land behind it: pushing the pulse far enough offshore to start
calm clips its crest off the east edge, and making it long enough clips it
anyway. v1 seeded the wave inside the domain, so the domain had to hold the
whole wave.

**So this probe tests a different mechanism: the wave ARRIVES from outside.**
The east edge is driven as a prescribed sea level over time -- exactly the
mechanism `_apply_source` already implements on the west edge, only time-varying
and on the other side. The open ocean beyond the map draws down and then surges;
the map is a window onto a coast, not a container for a whole wave. The wave
period is then a free parameter in TIME and is no longer limited by how much
ocean fits on screen, which is the entire point.

    eta(t) = A * ((t - t0) / T) * exp(-0.5 * ((t - t0) / T)^2),   t0 = 2T

The same N-shape v1 used, in time instead of space: negative (trough, the sea
withdrawing) before t0, positive (crest, the wave) after it. Trough first is
again the whole mechanism, not a special case in the code. Peak magnitude is
A*exp(-0.5) = 0.607*A, not A -- worth stating, since v1's plan doc quotes A as
though it were the wave height.

Read-only: imports the real solver, drives it exactly as a kernel would, and
writes nothing. Run from the repo root:
    python docs/probe_tsunami_v2.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import config                              # noqa: E402
from app.compute_engine import create_engine        # noqa: E402
from app.fluid_solver import create_fluid_solver    # noqa: E402
from app.terrain_gen import coastline               # noqa: E402
from app.world_state import WorldState              # noqa: E402

N = config.TERRAIN_CELLS + 1
CELL = config.TERRAIN_CELL_SIZE
DT = config.FIXED_DT
DRY = config.FLUID_DRY_DEPTH
XS = (np.arange(N) - (N - 1) * 0.5) * CELL
EDGE_COLUMNS = config.FLUID_OUTFLOW_COLUMNS


def run(ocean_depth, land_edge_x, beach_run, amplitude, period,
        seconds=900.0, verbose=False, cell_size=10.0, inland_height=None):
    """Drive the REAL shipped path: no hand-rolled boundary here.

    An earlier version of this probe applied its own boundary by writing `h`
    into the east columns from Python. That measured the wrong system, and it
    took a wrong answer to notice: `water.outflow_enabled` defaults True, so
    `_apply_outflow` was draining the very columns the probe was driving, and
    every drawback it reported was fighting an open outlet. The shipped kernel
    suppresses the outlet while it owns those columns; the only way to be sure
    a probe measures that is to let the solver do it.
    """
    world = WorldState()
    world.terrain.cell_size = cell_size
    effective = coastline(world.terrain, {"ocean_depth_m": ocean_depth,
                                          "inland_height_m": float(
                                              inland_height if inland_height
                                              else max(2.0, ocean_depth * 0.3)),
                                          "land_edge_x": land_edge_x,
                                          "beach_run_m": beach_run})
    shore_x = effective["shore_x"]
    world.water.level = 0.0
    world.water.outflow_enabled = False
    world.water.tsunami_enabled = True
    world.water.tsunami_amplitude_m = amplitude
    world.water.tsunami_period_s = period

    solver = create_fluid_solver(create_engine().device)
    solver.initialize(world)
    solver.set_boundaries(world.terrain, {}, 0, 0)

    xs = (np.arange(N) - (N - 1) * 0.5) * cell_size
    bed_row = np.asarray(world.terrain.heights, dtype=np.float64)[N // 2]
    i_shore = int(np.argmin(np.abs(xs - shore_x)))
    slope = abs((bed_row[i_shore + 1] - bed_row[i_shore - 1]) / (2.0 * cell_size))

    retreat = flood = 0.0
    t_retreat = t_flood = None
    vmax = 0.0
    trace = []
    for step in range(int(seconds / DT)):
        solver.set_boundaries(world.terrain, {}, 0, 0)
        solver.advance(DT, config.FLUID_MAX_SUBSTEPS, config.FLUID_STABILITY_DT)
        if step % 120 != 119:
            continue
        t = (step + 1) * DT
        vmax = max(vmax, float(solver.diagnostics()["max_velocity"]))
        row = np.asarray(solver._h.numpy()).reshape(N, N)[N // 2]
        wet = np.flatnonzero(row > DRY)
        if not len(wet):
            continue
        waterline = float(xs[wet[0]])
        if waterline - shore_x > retreat:
            retreat, t_retreat = waterline - shore_x, round(t)
        if shore_x - waterline > flood:
            flood, t_flood = shore_x - waterline, round(t)
        trace.append((round(t), round(waterline)))

    if verbose:
        print("      t(s)   waterline_x(m)   [shore at %.0f]" % shore_x)
        for row in trace[::max(1, len(trace) // 14)]:
            print("  %8d   %14d" % row)
    return dict(shore_x=shore_x, slope=slope, retreat=retreat, flood=flood,
                t_retreat=t_retreat, t_flood=t_flood, vmax=vmax)


def main():
    print("grid %dx%d at %.0f m cells -> world %.0f m across"
          % (N, N, 10.0, N * 10.0))
    print("")
    print("  depth  land_x  beach   amp  period  shore  slope     RETREAT(t)"
          "        FLOOD(t)   max|u|")
    best = None
    for ocean_depth, land_edge_x, beach_run in ((20.0, -950.0, 1900.0),
                                                (20.0, -950.0, 1300.0),
                                                (12.0, -950.0, 1900.0)):
        for amplitude in (6.0, 9.0):
            for period in (100.0, 200.0):
                r = run(ocean_depth, land_edge_x, beach_run, amplitude, period)
                print("  %5.1f  %6.0f  %5.0f  %4.1f  %6.0f %6.0f  1:%-6.0f "
                      "%5.0f(%4ss) %6.0f(%4ss)  %6.2f"
                      % (ocean_depth, land_edge_x, beach_run, amplitude, period,
                         r["shore_x"], 1.0 / max(r["slope"], 1e-9),
                         r["retreat"], r["t_retreat"], r["flood"], r["t_flood"],
                         r["vmax"]))
                sys.stdout.flush()
                score = min(r["retreat"], r["flood"])
                if best is None or score > best[0]:
                    best = (score, ocean_depth, land_edge_x, beach_run,
                            amplitude, period)
    if best:
        print("")
        print("best (largest smaller-of-retreat-and-flood): depth=%.0f land_x=%.0f "
              "beach=%.0f amp=%.1f period=%.0f" % best[1:])
        r = run(best[1], best[2], best[3], best[4], best[5], verbose=True)
        print("  shore_x=%.0f  slope 1:%.0f  SEA OUT %.0f m at t=%ss  "
              "FLOOD %.0f m inland at t=%ss  max|u|=%.2f m/s"
              % (r["shore_x"], 1.0 / r["slope"], r["retreat"], r["t_retreat"],
                 r["flood"], r["t_flood"], r["vmax"]))


if __name__ == "__main__":
    main()
